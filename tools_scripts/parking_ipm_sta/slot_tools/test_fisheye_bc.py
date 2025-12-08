
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


def parse_args():
    parser = argparse.ArgumentParser(description="slot_onnx")
    parser.add_argument("--onnx_path", default="/home/jovyan/gpal_neural_network/20251022_log/20251022_int16_quantized_model.bc", type=str)
    parser.add_argument("--img_path", default="/data/ai_group/datasets/bev_park/park_slot_jira/1114_park_bad/all_2025_11_14_13_57_28/", type=str)
    parser.add_argument("--label_path", default="/data/ai_group/datasets/bev_park/jira/jira3319/EKART_GPAL-MIFA7-004_tgr_2025-10-23_15-01-41/tgr_2025-10-23_15-01-41/tgr_2025-10-23_15-01-41/avm_image", type=str)
    parser.add_argument("--save_path", default="/data/ai_group/datasets/bev_park/park_slot_jira/1114_park_bad/all_2025_11_14_13_57_28_int16_quantized_bc_res", type=str)
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

def detect_fisheye_nv12(fisheye_img, img_name, onnx_mode_path):
    fisheye_config = '/data/ai_group/datasets/bev_park/park_slot_jira/1114_park_bad/calib'
    mask_front_path = os.path.join(fisheye_config, "mask", 'mask_front_89.jpg')
    mask_left_path = os.path.join(fisheye_config, "mask", 'mask_left_89.jpg')
    mask_back_path = os.path.join(fisheye_config, "mask", 'mask_back_89.jpg')
    mask_right_path = os.path.join(fisheye_config, "mask", 'mask_right_89.jpg')
  
    img_front = fisheye_img[0]
    img_front_y, img_front_uv = bgr_to_nv12_split(img_front)
    img_left = fisheye_img[1]
    img_left_y, img_left_uv = bgr_to_nv12_split(img_left)
    img_rear = fisheye_img[2]
    img_rear_y, img_rear_uv = bgr_to_nv12_split(img_rear)
    img_right = fisheye_img[3]
    img_right_y, img_right_uv = bgr_to_nv12_split(img_right)
    
    avm_w = 768
    avm_h = 768
    mask_back = cv2.imread(mask_back_path,0)
    if mask_back is None:
        print("mask back is None")
        return
    mask_back = cv2.resize(mask_back, (avm_w,avm_h))
    mask_front = cv2.imread(mask_front_path,0)
    mask_front = cv2.resize(mask_front, (avm_w,avm_h))
    mask_left = cv2.imread(mask_left_path,0)
    mask_left = cv2.resize(mask_left, (avm_w,avm_h))
    mask_right = cv2.imread(mask_right_path,0)
    mask_right = cv2.resize(mask_right, (avm_w,avm_h))

    rear_front_path = os.path.join(fisheye_config, 'mask', 'grid_rear_and_front.npy')
    left_right_path = os.path.join(fisheye_config, 'mask', 'grid_left_and_right.npy')
    grid_rear_and_front = np.load(rear_front_path)
    grid_left_and_right = np.load(left_right_path)

    grid_rear_and_front = torch.tensor(grid_rear_and_front).type(torch.float32)
    grid_left_and_right = torch.tensor(grid_left_and_right).type(torch.float32)
    
    ort_session = HBRuntime(onnx_mode_path)
    input_names = ort_session.input_names
    output_names = ort_session.output_names
    print("input_names ", input_names)
    print("output_names ", output_names)
    
    img_rear = img_rear.astype(np.float32)
    img_front = img_front.astype(np.float32)
    img_left = img_left.astype(np.float32)
    img_right = img_right.astype(np.float32)

    test_inputs = {}
    test_inputs['img_rear_y'] = img_rear_y
    test_inputs['img_rear_uv'] = img_rear_uv
    test_inputs['img_front_y'] = img_front_y
    test_inputs['img_front_uv'] = img_front_uv
    test_inputs['grid_rear_and_front'] = np.array(grid_rear_and_front)
    test_inputs['img_left_y'] = img_left_y
    test_inputs['img_left_uv'] = img_left_uv
    test_inputs['img_right_y'] = img_right_y
    test_inputs['img_right_uv'] = img_right_uv
    test_inputs['grid_left_and_right'] = np.array(grid_left_and_right)
    test_inputs['mask_rear'] = mask_back[None, :, :, None].astype(np.float32)
    test_inputs['mask_front'] = mask_front[None, :, :, None].astype(np.float32) 
    test_inputs['mask_left'] = mask_left[None, :, :, None].astype(np.float32)
    test_inputs['mask_right'] = mask_right[None, :, :, None].astype(np.float32) 
    if 0:
        save_calib_data_dir = "ptq/nv12_input_data_20250819-094336_300w"
        for k, v in test_inputs.items():
            save_path = os.path.join(save_calib_data_dir)
            os.makedirs(save_path, exist_ok=True)
            np.save(save_path + f"/{k}.npy", v)
    outputs = ort_session.run(output_names, input_feed=test_inputs)
    
    if 0:
        avm = np.loadtxt("ptq/aarch64_hbm_outputs/model_infer_output_0_avm.txt").reshape(768, 768, 3)
        slot_point = np.loadtxt("ptq/aarch64_hbm_outputs/model_infer_output_1_slot_point.txt").reshape(1, 1, 768, 768)
        slot_line = np.loadtxt("ptq/aarch64_hbm_outputs/model_infer_output_2_slot_line.txt").reshape(1, 1, 768, 768)
        outputs = (avm, slot_point, slot_line)
    # 处理输出
    print("\n模型输出信息:")
    for i, output in enumerate(outputs):
        print(f"输出 {i+1}: 形状={output.shape}, 类型={output.dtype}")

    output_pt = outputs[1]
    output_line = outputs[2]
    _, _, h, w = output_pt.shape
    heatmapValue = output_pt[0,0,:,:]
    linemapValue = output_line[0,0,:,:]
    point_img, line_img = getFeatureMap(heatmapValue, linemapValue, w, h)
    # print(line_img)
    save_folder = onnx_mode_path.split('/')[-1].split('.')[0] + "_quantize_nv12_aarch64" 
    savePath = os.path.join(args.save_path, save_folder)
    if not os.path.exists(savePath):
        os.makedirs(savePath)
    savename = savePath + '/' + img_name
    avm_img = np.array(np.clip(outputs[0], 0, 255)).astype(np.uint8)

    # cv2.imwrite(savename.replace('.jpg','_avm.jpg'), avm_img)
    # savePath = "/home/gpal/gpal_work/ParkingSlot/parking_slot/master_v3/code_train_segmentation/connvert_onnx/slot/{}.jpg".format(img_name[0:-4])
    # cv2.imwrite(savename.replace('.jpg','_point_map.jpg'), point_img)
    # cv2.imwrite(savename.replace('.jpg','_line_map.jpg'), line_img)

    SlotDetInstance = HeatMap(w, h)
    vertexElements = SlotDetInstance.doProc(heatmapValue, linemapValue)
    show_img = avm_img.copy()
    SlotDetInstance.drawVE(show_img, savename.replace('.jpg','_draw.jpg'))
    return vertexElements

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


    test_dir = args.img_path #"oneIMG" #"miniBatch" #"test_img"
    test_pic_folder = glob.glob(os.path.join(test_dir, "*/"))
    channel_fisheye_foler = ["img_front_fisheye", "img_left_fisheye", "img_rear_fisheye", "img_right_fisheye"]
    # do_save_img = False
    do_save_img = True
    frame_idx = 0
    # StatPackage = initStatPack()
    for pic_folder in test_pic_folder:
        
        front_Path = os.path.join(pic_folder, channel_fisheye_foler[0])
        left_Path = os.path.join(pic_folder, channel_fisheye_foler[1])
        rear_Path = os.path.join(pic_folder, channel_fisheye_foler[2])
        right_Path = os.path.join(pic_folder, channel_fisheye_foler[3])
        img_file_list = os.listdir(front_Path)
        for img_file in img_file_list:
            if frame_idx >= 320:
                continue
            print(f"frame_idx: {frame_idx}")
            frame_idx += 1
            
            front_img_path = os.path.join(front_Path, img_file)
            left_img_path = os.path.join(left_Path, img_file)
            rear_img_path = os.path.join(rear_Path, img_file)
            right_img_path = os.path.join(right_Path, img_file)
            fisheye_img = []
            fisheye_img.append(cv2.imread(front_img_path))
            fisheye_img.append(cv2.imread(left_img_path))
            fisheye_img.append(cv2.imread(rear_img_path))
            fisheye_img.append(cv2.imread(right_img_path))
            
            model_h = 768
            model_w = 768
            # vertexElements = detect_fisheye(fisheye_img, img_file, onnx_mode_path)
            detect_fisheye_nv12(fisheye_img, img_file, onnx_mode_path)

if __name__ == '__main__':
    main(args)