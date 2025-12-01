from gpal_lightning import const
from gpal_lightning.neural_network.runners.runner import Runner
from gpal_lightning.neural_network.tasks.build_task import build_tasks
from gpal_lightning.neural_network.network_modules.gpnet import GpNet
# from gpal_lightning.neural_network.network_modules.gpnet_deploy import GpNetDeploy

from gpal_lightning.utils.load_global_config import load_global_config
from gpal_lightning.utils.args_parser import ArgumentParserHelper

# from gpal_nn.models import necks, backbones, transformers
from gpal_nn.models import backbones, transformers
import torch
import logging
import argparse
import os
import numpy as np
import cv2


def parse_args():
    parser = argparse.ArgumentParser(description="slot_pth")
    parser.add_argument("--onnx_path", default="/home/jovyan/gpal_neural_network/20251022_log/20251022_int16_avm_original_float_model.onnx", type=str)
    parser.add_argument("--img_path", default="/data/ai_group/datasets/bev_park/park_slot_jira/static_avm/static_j6_avm", type=str)
    # parser.add_argument("--calib_data_path", default="/data/ai_group/datasets/bev_park/park_slot_jira/1114_park_bad/calib", type=str)
    # parser.add_argument("--label_path", default="/data/ai_group/datasets/bev_park/train_test_data/PointLineData_Test_fisheye/pointline_txt/", type=str)
    parser.add_argument("--save_path", default="/data/ai_group/datasets/bev_park/park_slot_jira/1114_static_j6_avm_20251022_int16_avm_original_float_onnx_model_res", type=str)
    args = parser.parse_args()
    return args

# import torch.multiprocessing as mp
# if mp.current_process().name == 'MainProcess':
#     mp.set_start_method('spawn')

def preprocess_img(avm_img):
    img_tensor = torch.tensor(avm_img) / 255.0
    img_tensor = img_tensor.permute(2, 0, 1)
    # mean = torch.tensor([0.481093804, 0.457524588, 0.407870549]).view(3, 1, 1)
    # std = torch.tensor([1.0, 1.0, 1.0]).view(3, 1, 1)
    # normalized_tensor = (img_tensor - mean) / std  # 应用归一化公式
  
    return img_tensor

def evaluate():
    args = ArgumentParserHelper.parse()
    dump_config = "save" in args and args.save != args.load_from
    global_config = load_global_config(
        args, override=False, dump_config=dump_config)

    global_config.validation = True
    tasks = build_tasks(global_config,
                        phase="validation",
                        tasks_root="gpal_nn.tasks")
    print(tasks)
    # if global_config.onnx_path != None:
    #     net = GpNetDeploy(global_config, tasks)
    # else:
    net = GpNet(global_config, tasks)

    if global_config.load_from:
        # checkpoint_path = get_checkpoint_path(global_config.load_from)
        checkpoint_path = global_config.load_from
        logging.info("Loading from {}".format(checkpoint_path))
        checkpoint = torch.load(checkpoint_path)
        try:
            net.load_state_dict(checkpoint["state_dict"], strict=False)
        except Exception as e:
            print(e)
        # PytorchToOnnx.reset_weight(net)
        logging.info("Model was loaded successfully")
    pic_path_list = os.listdir(args.img_path)
    for test_path in pic_path_list:
        print("path ", test_path)
        avm_img = cv2.imread(test_path)
        model_h = 768
        model_w = 768
        # vertexElements = detect_fisheye(fisheye_img, img_file, onnx_mode_path)
        img_name = os.path.basename(test_path)
        img = preprocess_img(avm_img)
        # if do_save_img:
        #     cv2.imwrite('./img.jpg', img)
        net.eval()
        with torch.no_grad():
            heatmapValueTensor, vecmapValueTensor, = net(img)

        #get heatmap w h ch 
        _, _, h, w = heatmapValueTensor.shape
        heatmapValue = heatmapValueTensor[0,0,:,:].cpu().detach().numpy()
        vecmapValue = vecmapValueTensor[0,0,:,:].cpu().detach().numpy()

        save_txt_path = os.path.join(args.save_txt, img_name.replace(".jpg", ".txt"))
        line_pt_info = []
        #         print("img_file ", front_img_path)
        # print("vertexElements ", vertexElements)
        label_img_raww = 768
        label_img_rawh = 768
        # sw, sh = getImageSizeScale(label_img_raww, label_img_rawh, model_w, model_h)
        # img_name_ext = os.path.splitext(img_name)[0]
        # for json_path in txt_path_list:
        #     if img_name_ext in json_path:
        #         #do Heatmap Statistics 
        #         txtPath = os.path.join(ann_dir, json_path)
        #         if not os.path.exists(txtPath):
        #             print("!!!!{} json path is not exist ".format(txtPath))
        #         # print("txtPath is: {}".format(txtPath))
        #         labelInstance = TXTLabelLoader(sw, sh)
        #         # heatmapResultPack = labelInstance.doHeatmapStatistics_json_anno(jsonPath, vertexElements)
        #         heatmapResultPack = labelInstance.doHeatmapStatistics(txtPath, vertexElements)
        #         updatePack(StatPackage, heatmapResultPack)
        
        # outputStat(StatPackage, weight_name=model_name)



if __name__ == '__main__':
    evaluate()
