
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

def parse_args():
    parser = argparse.ArgumentParser(description="slot_onnx")
    parser.add_argument("--onnx_path", default="/home/jovyan/gpal_neural_network/workspace/20251022_08_03_02_onnx_slot/checkpoint/epoch=146-step=150000_checkpoint_sim.onnx", type=str)
    parser.add_argument("--img_path", default="/data/ai_group/datasets/bev_park/train_test_dataset/300w_camera_fisheye_img_for_onnx/datas", type=str)
    parser.add_argument("--label_path", default="/data/ai_group/datasets/bev_park/train_test_dataset/300w_camera_fisheye_img_for_onnx/pointline_txt", type=str)
    parser.add_argument("--save_path", default="experiments/test_fisheye/20250819-094336_ddrnet_slim23_model_last_fisheye_300w", type=str)
    args = parser.parse_args()
    return args

args = parse_args()


def create_qat_datas(args):
    test_dir = args.img_path #"oneIMG" #"miniBatch" #"test_img"
    test_pic_folder = glob.glob(os.path.join(test_dir, "*/"))

    fisheye_config = '/data/ai_group/datasets/bev_park/train_test_dataset/300w_camera_fisheye_img_for_onnx/calib'
    mask_front_path = os.path.join(fisheye_config, "mask", 'mask_front_89.jpg')
    mask_left_path = os.path.join(fisheye_config, "mask", 'mask_left_89.jpg')
    mask_back_path = os.path.join(fisheye_config, "mask", 'mask_back_89.jpg')
    mask_right_path = os.path.join(fisheye_config, "mask", 'mask_right_89.jpg')
    
    avm_w = 768
    avm_h = 768
    mask_back = cv2.imread(mask_back_path,0)
    if mask_back is None:
        print("mask back is None")
        return
    mask_rear = cv2.resize(mask_back, (avm_w,avm_h))
    mask_front = cv2.imread(mask_front_path,0)
    mask_front = cv2.resize(mask_front, (avm_w,avm_h))
    mask_left = cv2.imread(mask_left_path,0)
    mask_left = cv2.resize(mask_left, (avm_w,avm_h))
    mask_right = cv2.imread(mask_right_path,0)
    mask_right = cv2.resize(mask_right, (avm_w,avm_h))

    rear_front_path = os.path.join(fisheye_config, 'mask','grid_rear_and_front.npy')
    left_right_path = os.path.join(fisheye_config, 'mask', 'grid_left_and_right.npy')
    grid_rear_and_front = np.load(rear_front_path)
    grid_left_and_right = np.load(left_right_path)
    # print(grid_left_and_right.shape)
    grid_rear_and_front = torch.tensor(grid_rear_and_front).type(torch.float32)
    grid_left_and_right = torch.tensor(grid_left_and_right).type(torch.float32)

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
            if frame_idx >= 50:
                continue
            # print(f"frame_idx: {frame_idx}")
            front_img_path = os.path.join(front_Path, img_file)
            left_img_path = os.path.join(left_Path, img_file)
            rear_img_path = os.path.join(rear_Path, img_file)
            right_img_path = os.path.join(right_Path, img_file)
            # fisheye_img = []
            # fisheye_img.append(cv2.imread(front_img_path))
            # fisheye_img.append(cv2.imread(left_img_path))
            # fisheye_img.append(cv2.imread(rear_img_path))
            # fisheye_img.append(cv2.imread(right_img_path))
            img_rear = cv2.imread(rear_img_path)
            img_front = cv2.imread(front_img_path)
            img_left = cv2.imread(left_img_path)
            img_right = cv2.imread(right_img_path)
            model_h = 768
            model_w = 768
            calib_path = "/data/ai_group/datasets/bev_park/train_test_dataset/300w_camera_fisheye_img_for_onnx/calib"
            save_path = "/data/ai_group/datasets/bev_park/train_test_dataset/300w_camera_fisheye_img_for_quat_tiny50"
            img_name = img_file.split('.')[0]

            img_rear = img_rear[None, ...].astype(np.float32)
            img_front = img_front[None, ...].astype(np.float32)
            img_left = img_left[None, ...].astype(np.float32)
            img_right = img_right[None, ...].astype(np.float32)
            print(img_rear.shape)
            print(mask_rear.shape)
            mask_rear_= mask_rear[None, :, :, None].astype(np.float32)
            mask_front_ = mask_front[None, :, :, None].astype(np.float32)
            mask_left_ = mask_left[None, :, :, None].astype(np.float32)
            mask_right_ = mask_right[None, :, :, None].astype(np.float32)

            save_rear_path = os.path.join(save_path, "img_rear")
            save_front_path = os.path.join(save_path, "img_front")
            save_left_path = os.path.join(save_path, "img_left")
            save_right_path = os.path.join(save_path, "img_right")

            save_mask_rear = os.path.join(save_path, "mask_rear")
            save_mask_front = os.path.join(save_path, "mask_front")
            save_mask_left = os.path.join(save_path, "mask_left")
            save_mask_right = os.path.join(save_path, "mask_right")

            save_grid_left_and_right = os.path.join(save_path, "grid_left_and_right")
            save_grid_rear_and_front = os.path.join(save_path, "grid_rear_and_front")

            npy_name = str(frame_idx) + '.npy'
            if not os.path.exists(save_rear_path):
                os.makedirs(save_rear_path)
            
            if not os.path.exists(save_front_path):
                os.makedirs(save_front_path)

            if not os.path.exists(save_left_path):
                os.makedirs(save_left_path)
            
            if not os.path.exists(save_right_path):
                os.makedirs(save_right_path)

            
            if not os.path.exists(save_mask_rear):
                os.makedirs(save_mask_rear)
            
            if not os.path.exists(save_mask_front):
                os.makedirs(save_mask_front)

            if not os.path.exists(save_mask_left):
                os.makedirs(save_mask_left)
            
            if not os.path.exists(save_mask_right):
                os.makedirs(save_mask_right)

            if not os.path.exists(save_grid_left_and_right):
                os.makedirs(save_grid_left_and_right)
            
            if not os.path.exists(save_grid_rear_and_front):
                os.makedirs(save_grid_rear_and_front)
            
            np.save(os.path.join(save_rear_path, npy_name), img_rear)
            np.save(os.path.join(save_front_path, npy_name), img_front)
            np.save(os.path.join(save_left_path, npy_name), img_left)
            np.save(os.path.join(save_right_path, npy_name), img_right)
            # np.save(os.path.join(save_grid_left_and_right, npy_name), grid_left_and_right)
            # np.save(os.path.join(save_grid_rear_and_front, npy_name), grid_rear_and_front)

            np.save(os.path.join(save_mask_rear, npy_name), mask_rear_)
            np.save(os.path.join(save_mask_front, npy_name), mask_front_)
            np.save(os.path.join(save_mask_left, npy_name), mask_left_)
            np.save(os.path.join(save_mask_right, npy_name), mask_right_)

            frame_idx += 1

def main(args):
    onnx_mode_path = args.onnx_path
    print(onnx_mode_path)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))



if __name__ == '__main__':
    create_qat_datas(args)