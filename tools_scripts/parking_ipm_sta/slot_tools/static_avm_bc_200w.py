
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

import torch
from torchvision import transforms
import numpy as np
from PIL import Image
import torch.nn as nn
import onnxruntime as ort
from horizon_tc_ui import HBRuntime
from gpal_nn.tasks.parking_ipm_sta.postprocess.heatmap_instance_p3 import HeatMap
from gpal_nn.tasks.parking_ipm_sta.datasets.txtlabel_instance_p3 import TXTLabelLoader
from gpal_nn.tasks.parking_ipm_sta.evaluators.evaluator import initStatPack, updatePack, outputStat
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(description="slot_onnx")
    parser.add_argument("--onnx_path", default="/home/jovyan/gpal_neural_network/20251201_log/20251201_int16_avm_200w_quat_1000num_quantized_model.bc", type=str)
    parser.add_argument("--label_path", default="/data/ai_group/datasets/bev_park/train_test_dataset/PointLineData_Test/pointline.txt", type=str)
    parser.add_argument("--ann_dir", default="/data/ai_group/datasets/bev_park/train_test_dataset/PointLineData_Test/", type=str)
    parser.add_argument("--save_dir", default="/home/jovyan/gpal_neural_network/20251201_log/0928_test_res", type=str)

    args = parser.parse_args()
    return args

args = parse_args()

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

def bgr_to_nv12_split(img_rgb):
    # 转换为YUV420 (I420)
    yuv_i420 = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2YUV_I420)
    h, w = img_rgb.shape[:2]
    
    # 提取Y平面
    y_plane = yuv_i420[:h, :]
    
    uv_start = h
    uv_height = h // 4 
    u_plane = yuv_i420[uv_start:uv_start+uv_height, :]
    v_plane = yuv_i420[uv_start+uv_height:uv_start+2*uv_height, :]
    
    uv_interleaved = np.zeros((h//2 * w//2 * 2), dtype=np.uint8)
    uv_interleaved[0::2] = u_plane.flatten()
    uv_interleaved[1::2] = v_plane.flatten()

    y_plane = y_plane.reshape(1, h, w, 1)   #.transpose(0, 3, 1, 2)
    uv_interleaved = uv_interleaved.reshape(1, h//2, w//2, 2)   #.transpose(0, 3, 1, 2)
    return y_plane, uv_interleaved

def detect_avm_nv12(avm_img, save_path, ort_session):

    avm_y, avm_uv = bgr_to_nv12_split(avm_img)
    
    avm_w = 768
    avm_h = 768

    input_names = ort_session.input_names
    output_names = ort_session.output_names
    # print("input_names ", input_names)
    # print("output_names ", output_names)

    test_inputs = {}
    test_inputs['img_avm_y'] = avm_y
    test_inputs['img_avm_uv'] = avm_uv

    outputs = ort_session.run(output_names, input_feed=test_inputs)
    
    if 0:
        avm = np.loadtxt("ptq/aarch64_hbm_outputs/model_infer_output_0_avm.txt").reshape(768, 768, 3)
        slot_point = np.loadtxt("ptq/aarch64_hbm_outputs/model_infer_output_1_slot_point.txt").reshape(1, 1, 768, 768)
        slot_line = np.loadtxt("ptq/aarch64_hbm_outputs/model_infer_output_2_slot_line.txt").reshape(1, 1, 768, 768)
        outputs = (avm, slot_point, slot_line)
    # 处理输出
    # print("\n模型输出信息:")
    # for i, output in enumerate(outputs):
    #     print(f"输出 {i+1}: 形状={output.shape}, 类型={output.dtype}")

    output_pt = outputs[0]
    output_line = outputs[1]
    _, _, h, w = output_pt.shape
    heatmapValue = output_pt[0,0,:,:]
    linemapValue = output_line[0,0,:,:]
    # point_img, line_img = getFeatureMap(heatmapValue, linemapValue, w, h)
    # print(line_img)
    # avm_img = np.array(np.clip(outputs[0], 0, 255)).astype(np.uint8)

    # cv2.imwrite(savename.replace('.jpg','_avm.jpg'), avm_img)
    # savePath = "/home/gpal/gpal_work/ParkingSlot/parking_slot/master_v3/code_train_segmentation/connvert_onnx/slot/{}.jpg".format(img_name[0:-4])
    # cv2.imwrite(savename.replace('.jpg','_point_map.jpg'), point_img)
    # cv2.imwrite(savename.replace('.jpg','_line_map.jpg'), line_img)

    SlotDetInstance = HeatMap(w, h)
    vertexElements = SlotDetInstance.doProc(heatmapValue, linemapValue)
    # print("return vertextElement ", vertexElements)
    show_img = avm_img.copy()
    # savename = os.path.join(save_dir, img_name)
    SlotDetInstance.drawVE(show_img, save_path)
    return vertexElements
############################################################ 
    
def getImageSizeScale(img_w, img_h, model_w, model_h):
    # sw = 240.0 / img_w
    # sh = 288.0 / img_h
    sw = float(model_w) / img_w
    sh = float(model_h) / img_h
    return sw, sh

def printTitleInfo():
    print ("")
    print ("--------------Slot Det Model Auto Test On Image DataSet---------------")
    print ("Algorithom vrsion : ", 'torch v0')
    print ("NetWork Description : ", " (ddrnet_23_slim)")
    #print "" 

def save_csv(d, out_data,result_path):
    data_pd = pd.DataFrame(data=out_data, index=None, columns=None)
    dst_path = result_path + f'/eval_landmark_slot_{d}.csv'
    data_pd.to_csv(dst_path, index=False)
    print(f'Save -> {dst_path}')
        
def read_txt(record_txt):
    with open(record_txt, "r", encoding='utf-8') as f:  #打开文本
        batch = f.read()   #读取文本
        batch = batch.split('\n')[0]
        assert batch, 'You do not have files to work on'
    # weights_path_new = onnx_path + batch + '/model_last.pth'
    return batch 

def main(args):
    onnx_mode_path = args.onnx_path
    print(onnx_mode_path)
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

    # img_valid_path = []
    # for img_path in test_pic_paths:
    #     img_name = os.path.basename(img_path)
    #     for file_path in file_paths:
    #         valid_img_path = file_path.split(" ")[1]
    #         if img_name in valid_img_path:
    #             img_valid_path.append(img_path
    ann_dir = args.ann_dir
    StatPackage = initStatPack()
    model_name = os.path.basename(args.onnx_path)
        
    ort_session = HBRuntime(onnx_mode_path)
    save_root = args.save_dir
    if not os.path.exists(save_root):
        os.makedirs(save_root)
    for test_path, json_path in tqdm(zip(pic_path_list, txt_path_list), desc="处理文件", 
        total=len(pic_path_list), unit="文件", colour="green"):
        # print("path ", test_path)
        # print("txt ", json_path)
        avm_img = cv2.imread(os.path.join(ann_dir,test_path))
        ori_imgh, ori_imgw, _= avm_img.shape
        if ori_imgh != 768:
            avm_img = cv2.resize(avm_img, (768, 768))
        model_h = 768
        model_w = 768
        # vertexElements = detect_fisheye(fisheye_img, img_file, onnx_mode_path)
        img_name = os.path.basename(test_path)
        save_path = os.path.join(save_root, img_name)
        vertexElements = detect_avm_nv12(avm_img, save_path, ort_session)
        # print("img_file ", front_img_path)
        # print("vertexElements ", vertexElements)
        label_img_raww = ori_imgw
        label_img_rawh = ori_imgh
        sw, sh = getImageSizeScale(label_img_raww, label_img_rawh, model_w, model_h)
        # img_name_ext = os.path.splitext(img_name)[0]
        # for json_path in txt_path_list:
        #     if img_name_ext in json_path:
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