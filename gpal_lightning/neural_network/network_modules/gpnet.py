import copy
import logging
import time
from collections import defaultdict
from typing import Callable, List, Union

import torch
from pytorch_lightning.core.module import LightningModule
from pytorch_lightning.utilities import rank_zero_info
from torch import distributed, nn
from torch.cuda.amp import autocast
from torch.distributed.distributed_c10d import ReduceOp
from torch.nn.utils import clip_grad_norm_

from gpal_lightning import const
from gpal_lightning.data.dataloader_helpers.build_val_or_infer_dataloaders import build_val_or_infer_dataloaders
from gpal_lightning.data.dataloader_helpers.ratio_dataloader import RatioDataloader
from gpal_lightning.data.dataloader_helpers.gpal_collate import gpal_collate
from gpal_lightning.data.dataset_identifier import DatasetIdentifier
from gpal_lightning.monitoring.logging import is_logging
from gpal_lightning.neural_network.global_config import GlobalConfig

from gpal_lightning.neural_network.gradient_operations import (
    grads_norm,
    reset_saved_grads,
    restore_saved_grads,
    save_grads,
    scale_grads,
)
from gpal_lightning.neural_network.gradient_scalers.builder import build_grad_scaler
from gpal_lightning.neural_network.lr_schedulers.builder import build_lr_scheduler
from gpal_lightning.neural_network.lr_schedulers.warmup_lr import warmup_lr
from gpal_lightning.neural_network.network_modules.backbones.builder import build
from gpal_lightning.neural_network.network_modules.base_module import StateResetable
from gpal_lightning.neural_network.network_modules.necks import builder as neck_builder
from gpal_lightning.neural_network.network_modules.transformers import builder as transformer_builder
from gpal_lightning.neural_network.optimizers.builder import build_optimizer
from gpal_lightning.neural_network.tasks.build_task import build_tasks_datasets
from gpal_lightning.utils.distributed import all_gather_by_chunks
from gpal_lightning.utils.datatype_convert import convert_tensor_to_fp32, convert_half_to_single_precision
from tools_scripts.data_format_cvt import ShowDataStruct
from gpal_lightning.utils.profiling import GetMemInfo, TrainSpeedRec, PrintTopProcesses, DetailProf


class GpNet(LightningModule):
    def __init__(
            self,
            global_config: GlobalConfig,
            tasks: List,
            automatic_optimization: bool = False,
            collate_fn: Callable = gpal_collate,
    ):
        super().__init__()
        self.global_config = global_config
        self.tasks = {task.name: task for task in tasks}
        self.collate_fn = collate_fn
        self.tasks_to_run = self.tasks
        self.automatic_optimization: bool = automatic_optimization
        self._training_parameters: list = []
#         # used to control when to update model's param
        self._curr_iteration: int = 0
#         # None means no warmup, otherwise this warmup is based on the epoch number
        self.warmup_epochs: Union[int, float, None] = None
#         # To indicate how many epochs later LR is decayed every time
#         self.lr_default_decay_epoch: int = 1
#         # This learning rate is used to track the process of lr warm-up, should be updated at each on_train_epoch_end
#         self.learning_rate: float = 0
        self.model = nn.ModuleDict()
#         self._batch_size: int = -1
#         self._camera_name: str = ""
        self._backbone_feature_id: dict = {}
        self._group_feature_id: dict = {}
        self._neck_feature_id: dict = {}
        self._curr_tasks: list = []
        self._fp_module_dict: dict = {}
        self._groups: dict = {}
        self._necks: dict = {}
        self._backbones: dict = {}
        self._transformers: dict = {}
        self._freeze_module_dict: dict = {}
        self._load_module_dict: dict = {}
        self.grad_scalers = self.build_grad_scaler()
        self._setup()
        self._training_parameters.extend(self.model.parameters())
        self._saved_gradients = [None for _ in range(
            len(self._training_parameters))]
        self.speed_rec = TrainSpeedRec(1000)
        self.sync_dt = 0.0

    def get_freeze_module_names(self, module_name):
        return self._freeze_module_dict.get(module_name, None)

    def get_load_module_names(self, module_name):
        return self._load_module_dict.get(module_name, None)

    def build_grad_scaler(self):
        # Mixed Precision Training
        grad_scalers = {}
        for task_name, task in self.tasks.items():
            grad_scalers[task_name] = build_grad_scaler(
                task.task_config.amp_scaler)
        return grad_scalers

#     @property
#     def batch_size(self):
#         """batch size is used to control the onnx model batch"""
#         return self._batch_size

#     @batch_size.setter
#     def batch_size(self, batch_size: int):
#         self._batch_size = batch_size
#         for module in self.model.values():
#             module.batch_size = batch_size

#     @property
#     def camera_name(self):
#         """camera id is used to control the switable batch norm"""
#         return self._camera_name

#     @camera_name.setter
#     def camera_name(self, camera_name: str):
#         self._camera_name = camera_name
#         for module in self.model.values():
#             module.camera_name = camera_name

    @property
    def curr_tasks(self):
        """curr_tasks is used to decide which task heads to use for model forwarding"""
        return self._curr_tasks

    @curr_tasks.setter
    def curr_tasks(self, curr_tasks: List[str]):
        self._curr_tasks = curr_tasks

    @property
    def curr_iteration(self):
        """This iteration is used to fetch data from correct task dataloader and optimizer updates"""
        return self._curr_iteration

    @curr_iteration.setter
    def curr_iteration(self, curr_interation):
        self._curr_iteration = curr_interation

    def get_progress_bar_dict(self):
        tqdm_dict = super().get_progress_bar_dict()
        tqdm_dict.pop("v_num", None)
        curr_task = self.curr_tasks[0] if self.curr_tasks else ""
        tqdm_dict["curr_task"] = curr_task
        return tqdm_dict

    def on_save_checkpoint(self, checkpoint) -> None:
        checkpoint["curr_iteration"] = self.curr_iteration
        checkpoint["grad_scalers"] = {key: grad_scaler.state_dict(
        ) for key, grad_scaler in self.grad_scalers.items()}
        super().on_save_checkpoint(checkpoint)

    def _setup(self):
        """Call module initialization methods here"""
        if self.global_config.Backbones:
            self._make_backbones()
        if self.global_config.Groups:
            self._make_groups()
        if self.global_config.Necks:
            self._make_necks()
        if self.global_config.Transformer:
            self._make_transformer()
        self._make_heads()

    def _make_backbones(self):
        # print(self.global_config.Backbones)
        self.backbone_camera_mapping = {}
        for backbone_name, backbone_config in self.global_config.Backbones.items():
            backbone_type = backbone_config.pop(
                "type") if "type" in backbone_config else ""
            # print(backbone_type)
            if "freeze" in backbone_config:
                self._freeze_module_dict[backbone_name] = backbone_config.pop(
                    "freeze")
            if "load" in backbone_config:
                self._load_module_dict[backbone_name] = backbone_config.pop(
                    "load")
            self._fp_module_dict[backbone_name] = backbone_config.pop(
                "ampfp16", False)
            self.backbone_camera_mapping[backbone_name] = backbone_config.pop(
                'input_source')
            backbone = build(self.global_config,
                             backbone_type, backbone_config)
            self.model[backbone_name] = backbone
        for task in self.tasks.values():
            if hasattr(task.task_config, "backbone"):
                self._backbones[task.name] = task.task_config.backbone

    def _make_necks(self):
        self.group_neck_mapping = {}
        for neck_name, neck_config in self.global_config.Necks.items():
            neck_type = neck_config.pop("type")
            if "freeze" in neck_config:
                self._freeze_module_dict[neck_name] = neck_config.pop("freeze")
            if "load" in neck_config:
                self._load_module_dict[neck_name] = neck_config.pop("load")
            self._fp_module_dict[neck_name] = neck_config.pop("ampfp16", False)
            self.group_neck_mapping[neck_config.pop(
                'input_source')] = neck_name
            neck_object = neck_builder.build(
                self.global_config, neck_type, neck_config)
            self.model[neck_name] = neck_object
        for task in self.tasks.values():
            if getattr(task.task_config, 'neck', False):
                self._necks[task.name] = task.task_config.neck

    def _make_groups(self):
        self.backbone_group_mapping = defaultdict(list)
        self.group_camera_mapping = {}
        for group_name, group_config in self.global_config.Groups.items():
            group_type = group_config.pop("type")
            if "freeze" in group_config:
                self._freeze_module_dict[group_name] = group_config.pop(
                    "freeze")
            if "load" in group_config:
                self._load_module_dict[group_name] = group_config.pop("load")
            self._fp_module_dict[group_name] = group_config.pop(
                "ampfp16", False)
            if "camera_source" in group_config:
                self.group_camera_mapping[group_name] = group_config.pop(
                    'camera_source')
            self.backbone_group_mapping[group_config.pop(
                'input_source')].append(group_name)

            group_obj = build(self.global_config, group_type, group_config)
            self.model[group_name] = group_obj
        for task in self.tasks.values():
            if getattr(task.task_config, 'group', False):
                self._groups[task.name] = task.task_config.group

    def val_dataloader(self):
        return self._build_val_or_infer_dataloaders(const.PHASE_VALIDATION)

    def predict_dataloader(self):
        return self._build_val_or_infer_dataloaders(const.PHASE_INFERENCE)

    def _make_transformer(self):
        transformer_type = self.global_config.Transformer.pop("type")
        transformer_config = self.global_config.Transformer.copy()
        if "freeze" in transformer_config:
            self._freeze_module_dict["transformer"] = transformer_config.pop(
                "freeze")
        if "load" in transformer_config:
            self._load_module_dict["transformer"] = transformer_config.pop(
                "load")
        self._fp_module_dict["transformer"] = transformer_config.pop(
            "ampfp16", False)
        transformer_object = transformer_builder.build(
            self.global_config, transformer_type, transformer_config)
        self.model["transformer"] = transformer_object
        for task in self.tasks.values():
            if getattr(task.task_config, 'transformer', False):
                self._transformers[task.name] = task.task_config.transformer

    def _make_heads(self):
        for task in self.tasks.values():
            if hasattr(task.task_config, "freeze"):
                self._freeze_module_dict[task.name] = task.task_config.freeze
            if hasattr(task.task_config, "load"):
                self._load_module_dict[task.name] = task.task_config.load
            self._fp_module_dict[task.name] = getattr(
                task.task_config, "ampfp16", False)
            self._group_feature_id[task.name] = task.task_config.group_feat_id
            self._backbone_feature_id[task.name] = task.task_config.backbone_feat_id
            self._neck_feature_id[task.name] = task.task_config.neck_feat_id
            if task.name in self._groups:
                if not self._group_feature_id[task.name]:
                    self._group_feature_id[task.name] = self._backbone_feature_id[task.name]
                    # [-1] means the last layer output
                    self._backbone_feature_id[task.name] = [-1]
            self.model[task.name] = task.head

    def configure_optimizers(self):
        optimizer_config = self.global_config.optimizer
        # print("optimizer_config", optimizer_config)
        optimizer = build_optimizer(
            optimizer_config, self._training_parameters)
        lr_scheduler_config = self.global_config.lr_scheduler
        if "warmup_epochs" in lr_scheduler_config:
            self.warmup_epochs = lr_scheduler_config.pop("warmup_epochs")
        if "lr_default_decay_epoch" in lr_scheduler_config:
            self.lr_default_decay_epoch = lr_scheduler_config.pop(
                "lr_default_decay_epoch")
        lr_scheduler = build_lr_scheduler(lr_scheduler_config, optimizer)
        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler}

    def train_dataloader(self):
        build_tasks_datasets(self.global_config, list(
            self.tasks_to_run.values()), "training")
        ratio_dataloader = RatioDataloader(
            self.global_config, list(self.tasks_to_run.values()), curr_iteration=self.curr_iteration
        )
        return ratio_dataloader

    def _build_val_or_infer_dataloaders(self, phase: str):
        build_tasks_datasets(self.global_config, list(
            self.tasks_to_run.values()), phase)
        dataloaders, self._dataloader_start_idx = build_val_or_infer_dataloaders(
            self.tasks_to_run, phase, self.global_config.image_per_gpu, collate_fn=self.collate_fn
        )
        return dataloaders

    def reset_all_states(self):
        for _, module in self.model.items():
            if isinstance(module, StateResetable):
                module.reset_all_states()

    def on_train_batch_start(self, batch, batch_idx, dataloader_idx=0):
        assert "image" in batch
        # assert "mask" in batch
        # assert "label" in batch
        # assert "camera_name" in batch
        assert "curr_task" in batch
        batch["image"] = convert_tensor_to_fp32(batch["image"])
        self.camera_name = batch["camera_name"]
        self.curr_tasks = [batch["curr_task"]]

        self.reset_all_states()

    def on_validation_start(self):
        if distributed.is_initialized():
            const.sync(distributed)

    def on_predict_start(self):
        if distributed.is_initialized():
            const.sync(distributed)

    def on_train_start(self):
        if distributed.is_initialized():
            const.sync(distributed)

    def _on_val_or_pred_batch_start_common(self, batch, batch_idx, phase, dataloader_idx=0):
        assert "image" in batch
        # assert "mask" in batch
        # assert "label" in batch
        # assert "camera_name" in batch
        assert "meta" in batch
        batch["image"] = convert_tensor_to_fp32(batch["image"])

        metadata = batch["meta"]

        camera_name = metadata[0]["camera_name"]
        curr_task = metadata[0]["task_name"]
        self.camera_name = camera_name
        self.curr_tasks = [curr_task]

        self.reset_all_states()
        # if const.JOBNAME != -1:
        #     rank_zero_info(
        #         f"{phase}: task_name: {curr_task}, batch_idx: {batch_idx}")

    def on_validation_batch_start(self, batch, batch_idx, dataloader_idx=0):
        self._on_val_or_pred_batch_start_common(
            batch, batch_idx, const.PHASE_VALIDATION, dataloader_idx=dataloader_idx)

#     def on_predict_batch_start(self, batch, batch_idx, dataloader_idx=0):
#         if "curr_task" in batch and hasattr(self.global_config, "image_path"):
#             curr_task = batch["curr_task"]
#             self.camera_name = self.global_config.camera_name
#             self.curr_tasks = [curr_task]
#             if const.JOBNAME != -1:
#                 rank_zero_info(f"{const.PHASE_INFERENCE}: task_name: {curr_task}, " f"batch_idx: {batch_idx}")
#         else:
#             self._on_val_or_pred_batch_start_common(
#                 batch, batch_idx, const.PHASE_INFERENCE, dataloader_idx=dataloader_idx
#             )

    def training_step(self, batch, batch_idx):
        time_dp = DetailProf()
        time_dp.Tic("begin")
        if "curr_iteration" not in batch:
            logging.warning(
                "No curr_iteration in current batch on rank:{}".format(self.global_rank))
            return

        optimizer = self.optimizers()
        curr_iteration = batch["curr_iteration"]
        epoch_percentage = (batch_idx + 1) / \
            self.trainer.num_training_batches + self.current_epoch

        # print(
        #     f"curr_iteration = {curr_iteration}  epoch_percentage = {epoch_percentage}")
        # We do optimizer step when the incoming iteration > self._curr_iteration
        # TODO: last iteration won't call this step, maybe consider fixing it
        if curr_iteration > self._curr_iteration:
            self.speed_rec.Rec(curr_iteration)
            self._curr_iteration = curr_iteration
            restore_saved_grads(self._training_parameters,
                                self._saved_gradients)
            # # nccl通信会忙等,实践发现这里可能是造成资源阻塞的原因,一旦出现阻塞无法自愈,
            # # 这里根据上一次同步时间sleep一个小时间片,能够概率上缓解资源冲突
            # # 由于目前训练数据是瓶颈,所以这个sleep并且不会造成实质上变慢
            # time.sleep(self.sync_dt * 0.3)
            self.sync_dt = self.SyncParamsGrad()
            optimizer.step()
            reset_saved_grads(self._training_parameters, self._saved_gradients)

            if not self.is_warmup_stage(epoch_percentage) and self.global_config.iteration_based_lr_schedulers:
                self.update_lr_schedulers()

        if self.global_config.max_iterations is not None and iteration_should_stop(
                self.global_config.max_iterations, curr_iteration
        ):
            self.trainer.should_stop = True

        # Update actual learning rate
        self.learning_rate = self.lr_schedulers().get_last_lr()[0]
        if self.is_warmup_stage(epoch_percentage):
            actual_lr = warmup_lr(
                self.warmup_epochs, epoch_percentage, optimizer, self.learning_rate)
        else:
            actual_lr = self.learning_rate

        # print(ShowDataStruct("batch", batch))

        time_dp.Duration("prepare", "begin")

        optimizer.zero_grad()
        data = batch['image']
        masks = batch["mask"] if 'mask' in batch else None
        trues = batch["label"]
        calib = batch.get('calib', None)
        metadata = batch['meta']
        curr_task = self.curr_tasks[0]
        camera_name = self.camera_name

        time_dp.Duration("data", "prepare")

        # print(f"curr_task = {curr_task}, camera_name = {camera_name}")
        preds = self.model_forward(data, calib,  phase=const.PHASE_TRAINING)[0]

        time_dp.Duration("model_forward", "data")

        losses = self.loss_forward(curr_task, preds, trues, masks)
        total_loss = losses.pop("total_loss")
        time_dp.Duration("loss_forward", "model_forward")

        if any(self._fp_module_dict.values()):
            # FP16 backward and scaler update
            scaler = self.grad_scalers[curr_task]
            scaler.scale(total_loss).backward()
            scaler.unscale_(optimizer)
            scaler.update()
        else:
            total_loss.backward()

        time_dp.Duration("backward", "loss_forward")

        norm = grads_norm(self._training_parameters)
        scale = self.tasks[curr_task].weight * 0.01 / norm if norm != 0 else 0
        scale_grads(self._training_parameters, scale)
        assert self.tasks[curr_task].task_config.grad_bounds != 0, f"{curr_task} grad_bounds shouldn't be 0"
        clip_grad_norm_(self._training_parameters,
                        self.tasks[curr_task].task_config.grad_bounds)
        save_grads(self._training_parameters, self._saved_gradients)
        time_dp.Duration("save_grads", "backward")

        if ((curr_iteration % self.global_config.log_every) == 0) and ("dataloader_time" in batch):
            all_rank_dataloader_time = self.all_gather(
                batch["dataloader_time"], sync_grads=False)
            sync_dt = self.all_gather(self.sync_dt, sync_grads=False)
            sync_dt = [float(ele) for ele in sync_dt]
            # if (self.global_rank % 8 == 0):
            #     PrintTopProcesses()

        time_dp.Duration("sync_dt", "save_grads")

        if is_logging(self.global_rank, curr_iteration, self.global_config.log_every):
            with torch.no_grad():
                self.log_scalar(
                    f"{curr_task}/training_total_loss", total_loss, curr_iteration)
                for loss_name, loss_value in losses.items():
                    if "loss" not in loss_name:
                        continue
                    self.log_scalar(
                        f"{curr_task}/training_{loss_name}", loss_value, curr_iteration)
                self.log_scalar("lr", actual_lr, curr_iteration)

                meminfo = GetMemInfo()
                for t, v in meminfo.items():
                    self.log_scalar(
                        "memory/" + t.replace('/', "_"), v, curr_iteration)

                train_spd10 = self.speed_rec.GetAvg(10)
                train_spd500 = self.speed_rec.GetAvg(500)
                if train_spd10 is not None:
                    self.log_scalar(
                        "train_speed/avg_10iter(iter_per_h)", train_spd10, curr_iteration)
                if train_spd500 is not None:
                    self.log_scalar(
                        "train_speed/avg_500iter(iter_per_h)", train_spd500, curr_iteration)

                if "dataloader_time" in batch:
                    all_rank_dataloader_time_sum = [
                        float(ele.sum()) for ele in all_rank_dataloader_time]
                    bad_rank_idx = torch.tensor(
                        all_rank_dataloader_time_sum).argmax()
                    # for rank_idx in range(0, len(all_rank_dataloader_time_sum), 8):
                    #     logging.warning(
                    #         f"{curr_task} rank{rank_idx}-{rank_idx+8} {all_rank_dataloader_time_sum[rank_idx:rank_idx+8]}")

                    self.log_scalar(
                        f"data_time/{curr_task}_bad_rank", int(bad_rank_idx), curr_iteration)
                    self.log_scalar(
                        f"data_time/{curr_task}_batch_sum", all_rank_dataloader_time[bad_rank_idx].sum(), curr_iteration)
                    self.log_scalar(
                        f"data_time/{curr_task}_batch_avg", all_rank_dataloader_time[bad_rank_idx].mean(), curr_iteration)
                    self.log_scalar(
                        f"data_time/{curr_task}_batch_max", all_rank_dataloader_time[bad_rank_idx].max(), curr_iteration)

                for rank_idx in range(0, len(all_rank_dataloader_time_sum), 8):
                    # logging.warning(
                    #     f"{curr_task} sync dt rank{rank_idx}-{rank_idx+8} {sync_dt[rank_idx:rank_idx+8]}")
                    self.log_scalar(f"sync_dt_mean/rank{rank_idx}-{rank_idx + 8}", torch.tensor(
                        sync_dt[rank_idx:rank_idx+8]).mean(), curr_iteration)

        # TODO We don't support logging by task-defined function now
        # if is_logging(self.global_rank, curr_iteration, self.global_config.log_every):
        #     self.scalar_log(curr_task, curr_iteration, total_loss)

        # image_log is used for visualizing images in tensorboard drawn by task-defined visualization functions
        if is_logging(self.global_rank, curr_iteration, self.global_config.visualize_every) and (curr_task in ["DRIVING_BEV_STA"]):
            self.image_log(curr_task, curr_iteration, data,
                           preds, masks, trues, metadata, total_loss)
        if const.JOBNAME != -1:
            rank_zero_info(
                f"iteration: {curr_iteration}, "
                f"current_epoch: {round(epoch_percentage, 3)}, "
                f"task_name: {curr_task}, "
                f"total_loss: {round(total_loss.item(), 3)}, "
                f"lr: {round(actual_lr, 5)}, "
                f"grad: {round(norm, 3)}"
            )
        self.log("total_loss", total_loss, prog_bar=True, logger=False)
        self.log("local_rank", self.local_rank, prog_bar=True, logger=False)
        self.log("global_rank", self.global_rank, prog_bar=True, logger=False)
        for loss_name, loss_value in losses.items():
            if "loss" not in loss_name:
                continue
            self.log(loss_name, loss_value, prog_bar=True, logger=False)
        self.log("grad", norm, prog_bar=True, logger=False)
        self.log("lr", actual_lr, prog_bar=True, logger=False)
        time_dp.Duration("log", "sync_dt")
        time_dp.Duration(f"{self.local_rank} {self.global_rank} all", "begin")
        # time_dp.Print()

        return {"loss": total_loss, "epoch_percentage": epoch_percentage}

    def is_warmup_stage(self, epoch_percentage):
        return self.warmup_epochs and epoch_percentage < self.warmup_epochs

    def update_lr_schedulers(self):
        lr_schedulers = self.lr_schedulers()
        lr_schedulers.step()

    def on_train_epoch_end(self) -> None:
        if (
                not self.is_warmup_stage(self.current_epoch)
                and not self.global_config.iteration_based_lr_schedulers
                and self.current_epoch % self.lr_default_decay_epoch == 0
        ):
            self.update_lr_schedulers()

    def on_predict_epoch_end(self, results) -> None:
        self.ddp_process_sync()

    def build_dataset_identifiers(self):
        dataset_identifiers: defaultdict = defaultdict(list)
        tasks = self.tasks.values()
        for task in tasks:
            datasets = task.val_datasets
            for idx, dataset in enumerate(datasets):
                if len(dataset) == 0:
                    continue
                dataset_identifier = DatasetIdentifier(
                    dataset.camera_name,
                    dataset.root_dir,
                    dataset.dataset_name,
                    dataset.sql_filter,
                    idx,
                    len(dataset),
                )
                dataset_identifiers[task.name].append(dataset_identifier)
        return dataset_identifiers

    def _validation_step_json_lists_and_output(self, batch, batch_idx, task_dataloader_idx):
        curr_task = self.curr_tasks[0]
        metadata = batch["meta"]

        preds, pred_json_list = self._gpal_predict_step(
            batch, batch_idx, task_dataloader_idx)

        data = batch["image"]
        if "points" in batch:
            data["points"] = batch.get("points", None)
        if "intrins" in batch:  # monodetpth
            data = batch
        trues = batch["label"]
        true_json_list, metadata_list = self.tasks[curr_task].vectors_to_json(
            metadata, data, task_dataloader_idx, trues, True
        )
        assert len(true_json_list) == len(metadata_list) == len(pred_json_list)
        masks = batch["mask"] if 'mask' in batch else None

        if isinstance(preds, tuple) and preds[1] is not None:
            losses = preds[1]
        else:
            preds = preds[0]
            losses = self.loss_forward(curr_task, preds, trues, masks)
            for key, val in losses.items():
                # Data that are intended to be synced through all_gather_object() must be in CPU.
                # Otherwise, the resulting gathered tensor will reside in different GPU devices
                # corresponding to different ranks, causing tensor arithmetics to fail.
                if isinstance(val, torch.Tensor):
                    losses[key] = val.item()
        output = {
            **losses,
            # "metadata": metadata,
            "camera_name": self.camera_name,
            "curr_task": curr_task,
        }
        return pred_json_list, true_json_list, metadata_list, output

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        curr_task = self.curr_tasks[0]
        task_dataloader_idx = dataloader_idx - \
            self._dataloader_start_idx[curr_task]
        # print(ShowDataStruct("batch", batch))
        pred_json_list, true_json_list, metadata_list, output = self._validation_step_json_lists_and_output(
            batch, batch_idx, task_dataloader_idx
        )
        if not self.global_config.dump_json_during_validation:
            # A hack to force all ddp processes to collect/sync.
            self.ddp_process_sync()
            all_preds = all_gather_by_chunks(
                pred_json_list, self.global_config.dist_val_split_n_chunks, "pred_json")
            all_trues = all_gather_by_chunks(
                true_json_list, self.global_config.dist_val_split_n_chunks, "true_json")
            all_metas = all_gather_by_chunks(
                metadata_list, self.global_config.dist_val_split_n_chunks, "metadata")
            if distributed.get_rank() != 0:
                return output
            if task_dataloader_idx not in self.tasks[curr_task].evaluators:
                self.tasks[curr_task].evaluators[task_dataloader_idx] = self.tasks[curr_task]._build_evaluator()
                self.tasks[curr_task].kpi_formatters[task_dataloader_idx] = self.tasks[curr_task]._build_kpi_formatter()
            evaluator = self.tasks[curr_task].evaluators[task_dataloader_idx]
            for pred_json, true_json, metadata in zip(all_preds, all_trues, all_metas):
                evaluator.process(pred_json, true_json, metadata)

        return output

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        self._gpal_predict_step(batch, batch_idx, dataloader_idx)

    def _gpal_predict_step(self, batch, batch_idx, dataloader_idx=0):
        metadata = batch["meta"]

        data = batch['image']
        # masks = batch["mask"] if 'mask' in batch else None
        # trues = batch["label"]
        calib = batch.get('calib', None)
        metadata = batch['meta']
        curr_task = self.curr_tasks[0]
        # camera_name = self.camera_name

        preds = self.model_forward(
            data, calib, metadata, phase=const.PHASE_VALIDATION)

        curr_task = self.curr_tasks[0]
        json_list, _ = self.tasks[curr_task].vectors_to_json(
            metadata, data, dataloader_idx, preds[0], False)

        # print(ShowDataStruct("json_list", json_list))
        # print(preds[0][0]["all_pts_preds"][-1, 0])
        # exit(1)
        return preds, json_list

    def forward_fp16(self, fp16_curr_module, fp16_next_module, curr_module, *args, **kwargs):
        with autocast(fp16_curr_module):
            result = self.model[curr_module](*args, **kwargs)
        if not fp16_curr_module or fp16_next_module:
            return result
        return convert_half_to_single_precision(result)

    def slice_forward(self, x, calib):
        if 'image' in x.keys():
            for key, value in x.items():
                x.update({key: value})
            del x

        assert len(self.curr_tasks) == 1

        outputs = []
        cam_feats = {}
        for key in x.keys():
            cam_feats[key] = x[key].clone().detach()

        cam_feats_bkp = copy.deepcopy(cam_feats)

        radar_point_feature = None
        for curr_task in self.curr_tasks:
            for backbone_name, camera_list in self.backbone_camera_mapping.items():
                backbone_input = [cam_feats[key]
                                  for key in camera_list if key in cam_feats.keys()]
                if backbone_input:
                    bs = len(backbone_input[0]) if isinstance(
                        backbone_input[0], list) else backbone_input[0].shape[0]
                    map_key_list = [
                        key for key in camera_list if key in cam_feats.keys()]
                    # 将bev的backbone运行拿出来运算，避免因为多个group计算多次
                    self.model[backbone_name].feature_id = self._backbone_feature_id[curr_task]
                    backbone_output = self.forward_fp16(self._fp_module_dict[backbone_name],
                                                        self._fp_module_dict[self.backbone_group_mapping[backbone_name][0]],
                                                        backbone_name, torch.cat(backbone_input))

                    # print(ShowDataStruct("backbone_output", backbone_output))

                    for group_name in self.backbone_group_mapping[backbone_name]:
                        if group_name in ['group2', 'group6']:
                            continue
                        # TODO: bev默认用当前group的最后一个neck
                        neck_name = self.group_neck_mapping[group_name]
                        if len(self.group_camera_mapping):
                            camera_source = self.group_camera_mapping[group_name]
                            group_input = []
                            for i, key in enumerate(map_key_list):
                                if key in camera_source:
                                    for j in range(len(backbone_output)):
                                        group_input.append(
                                            backbone_output[j][i * bs:(i + 1) * bs, ...])
                            output = torch.cat(group_input)
                        else:
                            camera_source = map_key_list
                            output = backbone_output

                        output = self.forward_fp16(self._fp_module_dict[group_name],
                                                   self._fp_module_dict[neck_name], group_name, output)

                        # print(self._neck_feature_id[curr_task])
                        self.model[neck_name].feature_id = self._neck_feature_id[curr_task]
                        output = self.forward_fp16(self._fp_module_dict[neck_name],
                                                   False, neck_name, output)
                        # print(ShowDataStruct("neck", output))
                        # exit(1)
                        for i, key in enumerate(camera_source):
                            if key not in cam_feats:
                                continue
                            for j in range(len(output)):
                                cur_output = output[j][i *
                                                       bs:(i + 1) * bs, ...]
                                if j == 0:
                                    cam_feats[key] = [cur_output]
                                else:
                                    cam_feats[key].append(cur_output)

            # print(f"**************backbone outputs**************")
            # for k in cam_feats:
            #     for fea in cam_feats[k]:
            #         print(k, fea.shape)

            if curr_task in self._transformers:
                curr_transformer = self._transformers[curr_task]
                output = {"img_bev_feat": self.model[curr_transformer](
                    cam_feats, calib)}
            else:
                output = {"cam_feats": cam_feats}
            # print(f"img_bev_feat = {output["img_bev_feat"].shape}")

            # import pickle as pkl
            # bev_feats, _ = pkl.load(open("../wangtong_bev_feats.pkl", 'rb'))
            # output = {"img_bev_feat": bev_feats}

            # print(self.model[curr_task])
            # exit(1)
            output = self.model[curr_task](output["img_bev_feat"], calib)
            outputs.append(output)

            # print(ShowDataStruct("output", output))

            # print(output[0]["all_pts_preds"][:,-1,0])
            # print(output[0]["all_bbox_preds"][:,-1,0])

        # exit(1)
        return outputs

    def image_forward(self, x):
        outputs = []
        for curr_task in self.curr_tasks:
            x0 = x
            if curr_task in self._backbones:
                curr_backbone = self._backbones[curr_task]
                self.model[curr_backbone].feature_id = self._backbone_feature_id[curr_task]
                x0 = self.model[curr_backbone](x0)

            if curr_task in self._groups:
                curr_group = self._groups[curr_task]
                self.model[curr_group].feature_id = self._group_feature_id[curr_task]

                x0 = self.model[curr_group](x0)

            if curr_task in self._necks:
                curr_neck = self._necks[curr_task]
                self.model[curr_neck].feature_id = self._neck_feature_id[curr_task]
                x0 = self.model[curr_neck](x0)

            output = self.model[curr_task](x0)
            outputs.append(output)
        return outputs

    def forward(self, x, calib=None, metadata=None, phase=const.PHASE_TRAINING):
        """logic for model forwarding
        Please set camera id, curr_tasks, curr_group before forwarding
        If doing onnx conversion, please set batch size
        """
        if isinstance(x, dict):
            outputs = self.slice_forward(x, calib)
        elif isinstance(x, torch.cuda.FloatTensor):
            outputs = self.image_forward(x)
        else:
            assert isinstance(x, (dict, torch.cuda.FloatTensor)
                              ), "[Data] data is neither a dicitonary or a torch.cuda.FloatTensor"
        return outputs

    def model_forward(self, data, calib=None, metadata=None, phase=const.PHASE_TRAINING):
        preds = self.forward(data, calib, metadata, phase=phase)
        return preds

    def loss_forward(self, curr_task, preds, trues, masks, **kwargs):
        losses = self.tasks[curr_task].head.loss(preds, trues, masks, **kwargs)
        return losses

    def image_log(self, curr_task, curr_iteration, data, preds, masks, trues, metadata, loss=None):
        self.tasks[curr_task].heavy_log(
            curr_iteration,
            "training",
            self.logger.experiment,
            data, preds, masks, trues, metadata, loss_info=loss

        )

    def log_scalar(self, log_key, log_val, iteration, epoch=None):
        """Log variables during the training process"""
        self.logger.experiment.add_scalar(log_key, log_val, iteration)

    def ddp_process_sync(self):
        """Use the all gather op to force all ranks to sync."""
        rank = self.global_rank
        rank = self.all_gather(rank)

    def SyncParamsGrad(self):
        world_size = distributed.get_world_size()
        grad_list = []
        shape_list = {}
        cnt = 0
        for i in range(len(self._training_parameters)):
            if self._training_parameters[i].grad is not None:
                grad_list.append(self._training_parameters[i].grad.reshape(-1))
                shape_list[i] = (cnt, cnt + grad_list[-1].shape[0],
                                 self._training_parameters[i].grad.shape)
                cnt = cnt + grad_list[-1].shape[0]
        grad_list_flatten = torch.cat(grad_list)
        all_reduce_t1 = time.time()
        distributed.barrier()
        distributed.all_reduce(
            grad_list_flatten, op=ReduceOp.SUM, async_op=False)
        all_reduce_t2 = time.time()
        for k, v in shape_list.items():
            self._training_parameters[k].grad = grad_list_flatten[v[0]: v[1]].reshape(
                v[2]) / world_size

        return all_reduce_t2 - all_reduce_t1


def iteration_should_stop(max_iteration: int, curr_iteration: int):
    if curr_iteration >= max_iteration:
        return True
    return False
