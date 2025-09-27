from collections import defaultdict
from typing import Callable, List, Union

import torch
from tqdm import tqdm
import numpy as np
from gpal_lightning import const
from gpal_lightning.data.dataloader_helpers.gpal_collate import gpal_collate
from gpal_lightning.neural_network.global_config import GlobalConfig

class GpNetDeploy(GpNet):
    def __init__(
            self,
            global_config: GlobalConfig,
            tasks: List,
            automatic_optimization: bool = False,
            collate_fn: Callable = gpal_collate,
    ):
        super().__init__(global_config, tasks, automatic_optimization, collate_fn)
        self.session = ort.InferenceSession(global_config.onnx_path)

    def forward_one_DRIVING_BEV_DYN(self, x, calib, metadata):
        batch_ret = {
            'with_postprocess': True,
            'Points_Loss': {
                'estimation_cen': [],
                'estimation_z': [],
                'estimation_dim': [],
                'estimation_dir': [],
                'estimation_vel': [],
                'estimation_score': [],
                'estimation_score_cls': [],
            }
        }
        batch_size = len(metadata)
        for i in tqdm(range(batch_size)):
            img_slice = torch.stack(
                [x[k][i] for k in x], dim=0).float().detach().cpu().numpy()
            # print(ShowDataStruct("img_slice", img_slice))
            ego2imgs = calib["ego2imgs"][i].unsqueeze(
                0).float().detach().cpu().numpy()
            outputs = self.session.run(
                None, {"image": img_slice, "ego2imgs": ego2imgs})
            for k, o in zip(batch_ret["Points_Loss"], outputs):
                batch_ret["Points_Loss"][k].append(o)

        for k in batch_ret["Points_Loss"]:
            batch_ret["Points_Loss"][k] = torch.from_numpy(np.stack(
                batch_ret["Points_Loss"][k], axis = 0)).cuda()
        # print(ShowDataStruct("batch_ret", batch_ret))
        return [batch_ret]

    def forward(self, x, calib=None, metadata=None, phase=const.PHASE_TRAINING):
        forward_outputs = []
        for task in self.tasks:
            # print(task)
            # print(ShowDataStruct("x", x))
            # print(ShowDataStruct("calib", calib))
            if "DRIVING_BEV_DYN" == task:
                output = self.forward_one_DRIVING_BEV_DYN(x, calib, metadata)
                forward_outputs.append(output)

        return forward_outputs
        
