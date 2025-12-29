import logging
import os
from functools import partial

from pytorch_lightning import Trainer
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.profilers import AdvancedProfiler, SimpleProfiler
from pytorch_lightning.callbacks import Callback

from gpal_lightning import const
from gpal_lightning.neural_network.global_config import GlobalConfig
from gpal_lightning.neural_network.plugins.finetune import Finetune
from gpal_lightning.neural_network.plugins.model_checkpoint import (
    EpochModelCheckpoint,
    IterationModelCheckpoint,
)
from gpal_lightning.neural_network.plugins.nn_env import NNTCPEnvironment
from gpal_lightning.utils.copytree_with_progressbar import copytree_with_progressbar
from gpal_lightning.utils.evaluation_helpers.evaluation_pipeline import EvaluationPipeline
from torch.utils.tensorboard import SummaryWriter


class DatasetEpochCallback(Callback):
    """在每个epoch开始前调用DRIVING_BEV_DYNDataset的函数"""
    
    def __init__(self):
        super().__init__()
        print("=== DatasetEpochCallback initialized ===")
    
    def on_train_epoch_start(self, trainer, pl_module):
        """在每个训练epoch开始时调用"""
        try:
            dataloaders = trainer.train_dataloader.dataloaders
            for task_name, dataloader in dataloaders.items():
                if task_name == "DRIVING_BEV_DYN":
                    datasets = dataloader[0]
                    datasets.dataset.datasets[0].set_current_epoch(trainer.current_epoch)
                    return
        except Exception as e:
            print(f"Error accessing trainer.train_dataloader: {e}")
        


class Runner:
    #     """Runner wrapper for Gpnet, it should provide runner.train(), runner.eval(),
    #     runner.test() for user"""

    def __init__(self, global_config: GlobalConfig, net, mode=const.PHASE_TRAINING):
        self.global_config = global_config
        self.net = net
        self._plugins = None
        self._training_callbacks = None
        self._evaluation_callbacks = None
        self._prediction_callbacks = None
        self.tuned_system_config = {}
        self.mode = mode
        self.global_step = 0
        self._setup()

    def _setup_plugins(self):
        plugins = []
        if const.JOBNAME != -1:
            # means JOB is deployed on Model Factory
            plugins.append(NNTCPEnvironment())

        if plugins:
            self._plugins = plugins

    def _setup_callbacks(self):
        # seperate training/evalution/prediction callbacks
        training_callbacks = [Finetune(), DatasetEpochCallback()]
        evaluation_callbacks = []
        prediction_callbacks = []

        # add model saving callback for training
        checkpoint_path = os.path.join(
            self.global_config.save, const.CHECKPOINT_PATH)
        model_checkpoint_config = {
            "save_last": True,
            "dirpath": checkpoint_path,
            "save_top_k": -1,
            "save_on_train_epoch_end": True,
            "mode": self.mode
        }
        if self.global_config.max_iterations is not None:
            model_checkpoint = partial(
                IterationModelCheckpoint, load_from=self.global_config.load_from, to_resume=self.global_config.to_resume
            )
            model_checkpoint_config["save_every_n_iterations"] = self.global_config.save_every_n_iterations
            model_checkpoint_config["filename"] = "{epoch}-{step}"
        else:
            model_checkpoint = partial(
                EpochModelCheckpoint, load_from=self.global_config.load_from, to_resume=self.global_config.to_resume
            )
            model_checkpoint_config[
                "save_checkpoint_before_validation"
            ] = self.global_config.save_checkpoint_before_validation
            model_checkpoint_config["every_n_epochs"] = self.global_config.save_every_n_epochs
            model_checkpoint_config["filename"] = "{epoch}"
        checkpoint_callback = model_checkpoint(**model_checkpoint_config)
        checkpoint_callback.CHECKPOINT_NAME_LAST = const.CHECKPOINT_NAME_LAST
        checkpoint_callback.FILE_EXTENSION = const.FILE_EXTENSION
        training_callbacks.append(checkpoint_callback)
        evaluation_callbacks.append(checkpoint_callback)
        prediction_callbacks.append(checkpoint_callback)

        # Add network callbacks for all three phases
        for task in self.net.tasks.values():
            training_callbacks.extend(task.task_config.callbacks.values())
            evaluation_callbacks.extend(task.task_config.callbacks.values())
            prediction_callbacks.extend(task.task_config.callbacks.values())

        if training_callbacks:
            self._training_callbacks = training_callbacks
        if evaluation_callbacks:
            self._evaluation_callbacks = evaluation_callbacks
        if prediction_callbacks:
            self._prediction_callbacks = prediction_callbacks

    def _setup_logging(self):
        logging.getLogger().setLevel(self.global_config.logging_level)

    def _setup(self):
        self._setup_plugins()
        self._setup_callbacks()
        self._setup_logging()

    def _get_train_step(self):
        """Get Max training steps for the runner.
        Runners will choose the minimum limit if both "max_epoch" and "max_steps" are set.
        """
        max_train_steps = self.global_config.max_steps
        print(f"self.global_config.max_steps = {self.global_config.max_steps}")

        return max_train_steps

    def train(self):
        self._train()

    def _train(self):
        zpilot_lightning_logger = pl_loggers.TensorBoardLogger(
            save_dir=self.global_config.save, name=const.LOG_PATH, version=""
        )
        # profiler = AdvancedProfiler(dirpath=self.global_config.save, filename="advanced_perf_logs")
        profiler = SimpleProfiler(
            dirpath=self.global_config.save, filename="simple_perf_logs")
        trainer = Trainer(
            max_epochs=self.global_config.max_epochs,
            max_steps=self._get_train_step(),
            accelerator="gpu",
            devices=self.global_config.gpus,
            logger=zpilot_lightning_logger,
            default_root_dir=self.global_config.save,
            num_nodes=self.global_config.num_nodes,
            precision=self.global_config.precision,
            strategy="ddp",
            check_val_every_n_epoch=self.global_config.check_val_every_n_epoch,
            # replace_sampler_ddp=False,
            callbacks=self._training_callbacks,
            # progress_bar_refresh_rate=progress_bar_refresh_rate,
            enable_progress_bar=True,
            plugins=self._plugins,
            num_sanity_val_steps=self.global_config.num_sanity_val_steps,
            profiler=profiler
        )

        trainer.fit(self.net)

    def eval(self) -> None:
        """
        This method will handle validation related logic,
        """
        # True means Tensorboard logger
        gpal_lightning_logger = True

        evaluator = Trainer(
            accelerator="gpu",
            devices=self.global_config.gpus,
            logger=gpal_lightning_logger,
            num_nodes=self.global_config.num_nodes,
            precision=self.global_config.precision,
            strategy="ddp",
            # replace_sampler_ddp=False,
            plugins=self._plugins,
            callbacks=self._evaluation_callbacks,
        )

        evaluator.validate(self.net)

        if not evaluator.is_global_zero:
            return

        print("evaluator done")

        dataset_identifiers = self.net.build_dataset_identifiers()
        print(self.global_config.dump_path)
        if self.global_config.dump_json_during_validation:
            if self.global_config.dump_path:
                dump_path = os.path.join(
                    self.global_config.dump_path, const.CURRENT_TIME)
            else:
                NotImplementedError("self.global_config.dump_path")
            print(dump_path)

            # 打开tensorboard
            tensorboard_path = os.path.join(
                self.global_config.dump_path, "log")
            writer = SummaryWriter(tensorboard_path)

            combined_kpi_dict = {}
            for task in self.net.tasks.values():
                print(task)
                # continue
                for dataset_idx, dataset_identifier in enumerate(dataset_identifiers[task.name]):
                    kpi, formatted_kpi = EvaluationPipeline.run(
                        task, dump_path, dataset_idx, dataset_identifier
                    )
                    # dict_to_json(
                    #     os.path.join(
                    #         dump_path, f"{task.name}_kpi_{dataset_idx}{const.EVALUATION_FILES_EXTENSION}"),
                    #     kpi,
                    # )

                    # if self.global_config.vis_badcase:  # visualize eval badcase images
                    #     task.visualize_badcase(dataset_identifier)

            writer.close()