
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
from horizon_tc_ui import HBRuntime

# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
#300w quat
# def parse_args():
#     parser = argparse.ArgumentParser(description="slot_onnx")
#     parser.add_argument("--onnx_path", default="/home/jovyan/gpal_neural_network/workspace/20251022_08_03_02_onnx_slot/checkpoint/epoch=146-step=150000_checkpoint_sim.onnx", type=str)
#     parser.add_argument("--img_path", default="/data/ai_group/datasets/bev_park/train_test_dataset/300w_camera_fisheye_img_for_onnx/datas/label/2025-08-26_16-36-26-310/avm", type=str)
#     parser.add_argument("--save_path", default="/data/ai_group/datasets/bev_park/train_test_dataset/300w_camera_avm_for_bc_quat", type=str)
#     args = parser.parse_args()
#     return args

def parse_args():
    parser = argparse.ArgumentParser(description="slot_onnx")
    parser.add_argument("--img_path", default="/data/ai_group/datasets/bev_park/train_test_dataset/test_new_add", type=str)
    parser.add_argument("--avm_path_save", default="/data/ai_group/datasets/bev_park/train_test_dataset/PointLineData_Test_quat_avm_100", type=str)
    parser.add_argument("--save_path", default="/data/ai_group/datasets/bev_park/train_test_dataset/PointLineData_Test_quat_npy_100", type=str)
    args = parser.parse_args()
    return args

def create_qat_datas(args):
    test_dir = args.img_path #"oneIMG" #"miniBatch" #"test_img"
    save_path = args.save_path
    avm_path_root = args.avm_path_save
    avm_path_list = glob.glob(os.path.join(test_dir, "*/*/*/*/*.jpg"))
    print("cur num is ", len(avm_path_list))
    avm_w = 768
    avm_h = 768
    do_save_img = True
    frame_idx = 1000
    # StatPackage = initStatPack()
    save_avm_npy_path = save_path
    if not os.path.exists(save_avm_npy_path):
        os.makedirs(save_avm_npy_path)
    
    np.random.shuffle(avm_path_list)

    for avm_path in avm_path_list:
        if frame_idx >1100:
            continue
        img_avm = cv2.imread(avm_path)

        img_avm = img_avm[None, ...].astype(np.float32)
    
        npy_name = str(frame_idx) + '.npy'
        
        np.save(os.path.join(save_avm_npy_path, npy_name), img_avm)
        vis_avm_path = avm_path.replace(test_dir, avm_path_root)
        vis_avm_path_root = os.path.dirname(vis_avm_path)
        if not os.path.exists(vis_avm_path_root):
            os.makedirs(vis_avm_path_root)
            shutil.copy2(avm_path, vis_avm_path)

        # np.save(os.path.join(save_grid_left_and_right, npy_name), grid_left_and_right)
        # np.save(os.path.join(save_grid_rear_and_front, npy_name), grid_rear_and_front)
        frame_idx += 1

def main(args):
    onnx_mode_path = args.onnx_path
    print(onnx_mode_path)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))



if __name__ == '__main__':
    args = parse_args()
    create_qat_datas(args)