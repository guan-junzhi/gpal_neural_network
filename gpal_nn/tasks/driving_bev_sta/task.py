import numpy as np
from gpal_lightning.neural_network.tasks.base.task import BaseTask
from gpal_lightning.neural_network.tasks.builder import TASKS
import torch
import random


@TASKS.register_module()
class DRIVING_BEV_STATask(BaseTask):
    def __init__(self, global_config, task_config, name):
        super().__init__(global_config, task_config, name, None)
        pass

    def GetVis(self, preds, gts, idx):

        from tools_scripts.vis_2d import Vis2D
        vis1 = Vis2D([-30, 100], [-20, 20], 0.1)
        try:
            for l in gts[idx]['edges']['points']:
                vis1.DrawPolyline(l, [0, 255, 255], 2)
            for l in gts[idx]['polylines']['points']:
                vis1.DrawPolyline(l, [0, 255, 0], 2)
        except:
            pass
        vis_draw1 = vis1.Draw()

        pre_pts = preds['all_pts_preds']
        pre_pts_denorm = torch.stack(
            [(1-pre_pts[..., 1]) * 96, ((1-pre_pts[..., 0])-0.5) * 32], dim=-1)

        vis2 = Vis2D([-30, 100], [-20, 20], 0.1)
        for l, ln, s in zip(pre_pts_denorm[-1, idx], pre_pts[-1, idx], preds['all_cls_scores'][-1, idx]):
            if s[1:].sigmoid().max() > 0.1:
                # print(f"ln \n{s.sigmoid().max()}")

                color = [random.randint(0, 255), random.randint(
                    0, 255), random.randint(0, 255)]
                vis2.DrawPolyline(l.detach().cpu().numpy(), color, 2)
        vis_draw2 = vis2.Draw()

        return np.concatenate([vis_draw1, vis_draw2], axis=1)

    def heavy_log(self, iteration, phase, log_writer, data, preds, masks, trues, metadata, loss_info=None):
        imgs = []
        for idx in range(4):
            vis = self.GetVis(preds[0], trues, idx)
            imgs.append(vis)

        imgs = np.concatenate(imgs, axis=1)
        self.logger.image_log(iteration, phase, log_writer,
                              0, torch.from_numpy(imgs).permute(2, 0, 1).flip(0))
