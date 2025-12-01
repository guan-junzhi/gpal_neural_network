
import os
import cv2
import numpy as np
import PIL.Image as Image
from pathlib import Path
import datetime
import shutil
import argparse
import time
import pandas as pd
import glob
import json
# 获取当前脚本所在的目录
current_path = "/home/jovyan/gpal_neural_network"
import sys
sys.path.append(current_path)
from gpal_nn.tasks.parking_ipm_sta.postprocess.heatmap_instance_p3 import HeatMap
# 获取当前脚本所在的目录
current_path = Path(__file__).resolve().parent
parent_path = current_path.parent
import sys
sys.path.append(os.path.join(parent_path))

import torch
from torchvision import transforms
import numpy as np
from PIL import Image
import torch.nn as nn
import onnxruntime as ort

from gpal_nn.tasks.parking_ipm_sta.postprocess.heatmap_instance_p3 import HeatMap
from gpal_nn.tasks.parking_ipm_sta.datasets.txtlabel_instance_p3 import TXTLabelLoader
from gpal_nn.tasks.parking_ipm_sta.evaluators.evaluator import initStatPack, updatePack, outputStat

def parse_args():
    parser = argparse.ArgumentParser(description="slot_onnx")
    parser.add_argument("--onnx_path", default="/home/jovyan/gpal_neural_network/20251022_log/20251022_int16_avm_original_float_model.onnx", type=str)
    parser.add_argument("--img_path", default="/data/ai_group/datasets/bev_park/park_slot_jira/jira3716/all_2025_11_22_11_51_24_avm", type=str)
    # parser.add_argument("--calib_data_path", default="/data/ai_group/datasets/bev_park/park_slot_jira/1114_park_bad/calib", type=str)
    # parser.add_argument("--label_path", default="/data/ai_group/datasets/bev_park/train_test_data/PointLineData_Test_fisheye/pointline_txt/", type=str)
    parser.add_argument("--save_path", default="/data/ai_group/datasets/bev_park/park_slot_jira/jira3716/all_2025_11_22_11_51_24_avm_20251022_int16_avm_original_float_onnx_model_res", type=str)
    args = parser.parse_args()
    return args

args = parse_args()

def preprocess_img(avm_img):
    img_tensor = torch.tensor(avm_img) / 255.0
    img_tensor = img_tensor.permute(2, 0, 1)
    # mean = torch.tensor([0.481093804, 0.457524588, 0.407870549]).view(3, 1, 1)
    # std = torch.tensor([1.0, 1.0, 1.0]).view(3, 1, 1)
    # normalized_tensor = (img_tensor - mean) / std  # 应用归一化公式
  
    return img_tensor

def preprocess(original_img, modelH, modelW):
    original_img = cv2.resize(original_img, (modelH, modelW))
    # original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
    # data_transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean=(122.67892/255, 116.66877/255, 104.00699/255),std=(1, 1, 1))])
    
    img = preprocess_img(original_img)
    img_np = img.to("cpu").numpy()
    img = torch.unsqueeze(img, dim=0)
    return img

def getFeatureMap(point_out, line_out, w, h):
    point_map = np.zeros((h, w, 1), np.uint8)
    line_map = np.zeros((h, w, 1), np.uint8)
    for j in range (h):
        for i in range (w):
            p_value = point_out[j,i]
            p_value = max(0,min(255, p_value*255))
            l_value = line_out[j,i]
            l_value = max(0,min(255, l_value*255))
            point_map[j][i]=int(p_value)
            line_map[j][i]=int(l_value)
    return point_map, line_map

def detect_avm(avm_img, img_name, onnx_mode_path, args):
    ort_session = ort.InferenceSession(onnx_mode_path)
    input_names = [input.name for input in ort_session.get_inputs()]
    input_shapes = [input.shape for input in ort_session.get_inputs()]
    input_types = [input.type for input in ort_session.get_inputs()]
    # for i, (name, shape, type) in enumerate(zip(input_names, input_shapes, input_types)):
    #     print("{} , name {}, shape {}, type {}".format(i+1, name, shape, type))
    
    avm_img = avm_img.astype(np.float32)
    # img_tensor = preprocess_img(avm_img)
    img_tensor = np.array(avm_img)
    test_inputs = {}
    test_inputs['img_avm'] = img_tensor[None, :, :, :]
   
    outputs = ort_session.run(None, test_inputs)

    # 处理输出
    # print("\n模型输出信息:")
    # for i, output in enumerate(outputs):
    #     print(f"输出 {i+1}: 形状={output.shape}, 类型={output.dtype}")

    output_pt = outputs[0]
    output_line = outputs[1]
    _, _, h, w = output_pt.shape
    heatmapValue = output_pt[0,0,:,:]
    linemapValue = output_line[0,0,:,:]
    point_img, line_img = getFeatureMap(heatmapValue, linemapValue, w, h)
    # print(line_img)
    save_folder = onnx_mode_path.split('/')[-1].split('.')[0] + "_onnx_res" 
    savePath = os.path.join(args.save_path, save_folder)
    if not os.path.exists(savePath):
        os.makedirs(savePath)
    savename = savePath + '/' + img_name
  
    # avm_img = np.array(outputs[0]).astype(np.uint8)
    # avm_img = cv2.cvtColor(avm_img, cv2.COLOR_RGB2BGR)

    # cv2.imwrite(savename, avm_img)
    # savePath = "/home/gpal/gpal_work/ParkingSlot/parking_slot/master_v3/code_train_segmentation/connvert_onnx/slot/{}.jpg".format(img_name[0:-4])
    # cv2.imwrite(savename.replace('.jpg','_point_map.jpg'), point_img)
    # cv2.imwrite(savename.replace('.jpg','_line_map.jpg'), line_img)

    SlotDetInstance = HeatMap(w, h)
    vertexElements = SlotDetInstance.doProc(heatmapValue, linemapValue)
 
    show_img = avm_img.copy()
    SlotDetInstance.drawVE(show_img, savename.replace('.jpg','_draw.jpg'))
    return vertexElements


def main(args):
    onnx_mode_path = args.onnx_path
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))


    test_dir = args.img_path #"oneIMG" #"miniBatch" #"test_img"
    avm_path_list = glob.glob(os.path.join(test_dir, "*.jpg"))
    print("process file ", len(avm_path_list))
    # channel_fisheye_foler = ["img_front_fisheye", "img_left_fisheye", "img_rear_fisheye", "img_right_fisheye"]
    # ann_dir = args.label_path
    # do_save_img = False
    do_save_img = True
    # StatPackage = initStatPack()
    point_num = 0
    line_num = 0

    for avm_path in avm_path_list:
        avm_img = cv2.imread(avm_path)
        img_name = os.path.basename(avm_path)
        # vertexElements, slotsList = detect_point(orig_img, model, model_h, model_w, device)
        # vertexElements, slotsList = detect(orig_img, model, model_h, model_w, device)
        model_h = 768
        model_w = 768
        vertexElements = detect_avm(avm_img, img_name, onnx_mode_path, args)

        for i in range(len(vertexElements)):
            orients = vertexElements[i][1]
            line_num = line_num + len(orients)
            print("cur frame line ", len(orients))
        print("cur frame point ", len(vertexElements))
        point_num = point_num + len(vertexElements)
    
    print("sum line ", line_num)
    print("sum_point ", point_num)
    
if __name__ == '__main__':
    main(args)
