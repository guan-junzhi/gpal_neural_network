from gpal_lightning.neural_network.tasks.builder import POSTPROCESSES
from gpal_lightning.neural_network.tasks.base.postprocesses.postprocess import (
    BasePostProcess,
)
from gpal_nn.tasks.radar4d_nn_sdk.postprocess.heatmap_instance_p3 import HeatMap

from tools_scripts.data_format_cvt import ShowDataStruct
import numpy as np


@POSTPROCESSES.register_module()
class RADAR4D_NN_SDKPostProcessing(BasePostProcess):
    def __init__(self, global_config, task_config):
        super().__init__(global_config, task_config)

    def process(self, vectors, metadata: dict, is_gt: bool = False) -> dict:

        if is_gt:
            return vectors
        else:
            # print(ShowDataStruct("vectors", vectors, 2, 3))
            b, _, h, w = vectors[0].shape

            vertexElements_batch = []

            for i in range(b):
                heatmapValue = vectors[0][i, 0, :, :].cpu().detach().numpy()
                vecmapValue = vectors[1][i, 0, :, :].cpu().detach().numpy()
                SlotDetInstance = HeatMap(w, h)
                vertexElements = SlotDetInstance.doProc(
                    heatmapValue, vecmapValue)
                vertexElements_batch.append(vertexElements)
            # print(ShowDataStruct("vertexElements_batch", vertexElements_batch, 2, 3))

            return vertexElements_batch

        pass
