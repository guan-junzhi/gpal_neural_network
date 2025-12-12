
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
    parser.add_argument("--onnx_path", default="/home/jovyan/gpal_neural_network/20251022_log/20251022_int16_avm_200w_quat_original_float_model.onnx", type=str)
    parser.add_argument("--label_path", default="/data/ai_group/datasets/bev_park/train_test_dataset/300w_camera_label_jx_c5_001/pointline.txt", type=str)
    parser.add_argument("--ann_dir", default="/data/ai_group/datasets/bev_park/train_test_dataset/300w_camera_label_jx_c5_001", type=str)
    parser.add_argument("--save_path", default="/data/ai_group/datasets/bev_park/park_slot_jira/1114_jira/res/300w_camera_label_jx_c5_001_det_res_static_thr5_onnx", type=str)
    parser.add_argument("--save_txt", default="300w_camera_label_jx_c5_001_det_res_static_thr5_onnx", type=str)
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
    avm_img = np.array(avm_img)
  
    test_inputs = {}
    test_inputs['img_avm'] = avm_img[None, :, :, :]
   
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

def getImageSizeScale(img_w, img_h, model_w, model_h):
    # sw = 240.0 / img_w
    # sh = 288.0 / img_h
    sw = float(model_w) / img_w
    sh = float(model_h) / img_h
    return sw, sh
def main(args):
    onnx_mode_path = args.onnx_path
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))
    label_path = args.label_path
    with open(label_path, "r") as f:
        txt_lines = f.readlines()
    txt_path_list = []
    pic_path_list = []
    for line in txt_lines:
        txt_path = line.split(" ")[0]
        pic_path = line.split(" ")[1].strip("\n")
        txt_path_list.append(txt_path)
        pic_path_list.append(pic_path)

    # test_dir = args.img_path #"oneIMG" #"miniBatch" #"test_img"
    # avm_path_list = glob.glob(os.path.join(test_dir, "*.jpg"))
    # print("process file ", len(avm_path_list))
    # channel_fisheye_foler = ["img_front_fisheye", "img_left_fisheye", "img_rear_fisheye", "img_right_fisheye"] if not os.path.exists(args.save_txt):
    if not os.path.exists(args.save_txt):
        os.makedirs(args.save_txt)
   
    StatPackage = initStatPack()
    model_name = os.path.basename(args.onnx_path)
    ann_dir = args.ann_dir
    # do_save_img = False
    do_save_img = True
    # StatPackage = initStatPack()
    point_num = 0
    line_num = 0

    for test_path in pic_path_list:
        print("path ", test_path)
        avm_img = cv2.imread(os.path.join(ann_dir,test_path))
        label_img_rawh, label_img_raww, _= avm_img.shape
        # vertexElements, slotsList = detect_point(orig_img, model, model_h, model_w, device)
        # vertexElements, slotsList = detect(orig_img, model, model_h, model_w, device)
        model_h = 768
        model_w = 768
        img_name = os.path.basename(test_path)
        avm_img = cv2.resize(avm_img, (model_h, model_w))
        vertexElements = detect_avm(avm_img, img_name, onnx_mode_path, args)

        # label_img_raww = 768
        # label_img_rawh = 768
        sw, sh = getImageSizeScale(label_img_raww, label_img_rawh, model_w, model_h)
        img_name_ext = os.path.splitext(img_name)[0]
        for json_path in txt_path_list:
            if img_name_ext in json_path:
                #do Heatmap Statistics 
                txtPath = os.path.join(ann_dir, json_path)
                if not os.path.exists(txtPath):
                    print("!!!!{} json path is not exist ".format(txtPath))
                # print("txtPath is: {}".format(txtPath))
                labelInstance = TXTLabelLoader(sw, sh)
                # heatmapResultPack = labelInstance.doHeatmapStatistics_json_anno(jsonPath, vertexElements)
                heatmapResultPack = labelInstance.doHeatmapStatistics(txtPath, vertexElements)
                updatePack(StatPackage, heatmapResultPack)
        
        outputStat(StatPackage, weight_name=model_name)
    
          
    

if __name__ == '__main__':
    main(args)
