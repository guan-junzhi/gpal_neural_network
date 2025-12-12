
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


def parse_args():
    parser = argparse.ArgumentParser(description="slot_onnx")
    parser.add_argument("--onnx_path", default="/home/jovyan/gpal_neural_network/20251201_log/20251201_int16_avm_200w_quat_1000num_quantized_model.bc", type=str)
    parser.add_argument("--img_path", default="/data/ai_group/datasets/bev_park/park_slot_jira/jira3716/all_2025_11_22_11_51_24_avm", type=str)
    parser.add_argument("--save_path", default="/data/ai_group/datasets/bev_park/park_slot_jira/jira3716/all_2025_11_22_11_51_24_avm_1201model_bc_res", type=str)
    parser.add_argument("--save_txt", default="200w_quat_j6avm_det_res", type=str)
    args = parser.parse_args()
    return args

args = parse_args()

class PrintLogger:
    """同时将输出打印到控制台和写入文件"""
    
    def __init__(self, log_file="output.log", mode="w", timestamp=True):
        """
        初始化打印记录器
        
        参数:
            log_file: 日志文件路径
            mode: 文件打开模式 ('w' 覆盖, 'a' 追加)
            timestamp: 是否在每条日志前添加时间戳
        """
        self.terminal = sys.stdout
        self.log_file = log_file
        self.timestamp = timestamp
        
        # 创建日志文件所在目录
        os.makedirs(os.path.dirname(log_file) or '.', exist_ok=True)
        
        # 打开日志文件
        self.log = open(log_file, mode, encoding="utf-8")
        
        # 写入日志头
        if mode == "w":
            self.write_header()
    
    def write_header(self):
        """写入日志文件头"""
         
        header = "Times Date : " + time.strftime("%d/%m/%Y") + " - " + time.strftime("%H:%M:%S")
        self.log.write(header)
    
    def write(self, message):
        """处理写入操作"""
        # 写入控制台
        self.terminal.write(message)
        
        # 写入文件
        self.log.write(message)
        self.log.flush()  # 确保数据立即写入文件
    
    def flush(self):
        """刷新缓冲区"""
        self.terminal.flush()
        self.log.flush()
    
    def close(self):
        """关闭日志文件"""
        footer = f"\n{'='*20} Log Ended at {time.strftime('%Y-%m-%d %H:%M:%S')} {'='*20}\n"
        self.log.write(footer)
        self.log.close()
        # 恢复原始标准输出
        sys.stdout = self.terminal
        print(f"日志已保存到: {os.path.abspath(self.log_file)}")

def log_to_file(log_file="output.log", mode="w", timestamp=True):
    """
    装饰器：将函数的所有打印输出捕获到文件
    
    使用示例：
    @log_to_file("my_script.log")
    def main():
        print("这将同时输出到控制台和文件")
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            original_stdout = sys.stdout
            try:
                # 重定向标准输出
                sys.stdout = PrintLogger(log_file, mode, timestamp)
                return func(*args, **kwargs)
            finally:
                # 恢复标准输出
                if isinstance(sys.stdout, PrintLogger):
                    sys.stdout.close()
                sys.stdout = original_stdout
        return wrapper
    return decorator

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

def detect_avm_nv12(avm_img, img_name, onnx_mode_path):

    avm_y, avm_uv = bgr_to_nv12_split(avm_img)
    
    avm_w = 768
    avm_h = 768
    
    ort_session = HBRuntime(onnx_mode_path)
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
    save_folder = onnx_mode_path.split('/')[-1].split('.')[0] + "_quantize_nv12_aarch64" 
    savePath = os.path.join(args.save_path, save_folder)
    if not os.path.exists(savePath):
        os.makedirs(savePath)
    savename = savePath + '/' + img_name
    # avm_img = np.array(np.clip(outputs[0], 0, 255)).astype(np.uint8)

    # cv2.imwrite(savename.replace('.jpg','_avm.jpg'), avm_img)
    # savePath = "/home/gpal/gpal_work/ParkingSlot/parking_slot/master_v3/code_train_segmentation/connvert_onnx/slot/{}.jpg".format(img_name[0:-4])
    # cv2.imwrite(savename.replace('.jpg','_point_map.jpg'), point_img)
    # cv2.imwrite(savename.replace('.jpg','_line_map.jpg'), line_img)

    SlotDetInstance = HeatMap(w, h)
    vertexElements = SlotDetInstance.doProc(heatmapValue, linemapValue)
    # print("return vertextElement ", vertexElements)
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
    test_pic_paths = glob.glob(os.path.join(test_dir, "*.jpg"))
    # do_save_img = False
    do_save_img = True
    frame_idx = 0
    # StatPackage = initStatPack()
    point_num = 0
    line_num = 0
    print("process img num= ", len(test_pic_paths))
    # valid_txt_path = "/data/ai_group/datasets/bev_park/park_slot_jira/1114_jira/datas/car_and_server_datas/server/pointline.txt"
    # with open(valid_txt_path, "r") as f:
    #     file_paths = f.readlines()

    # img_valid_path = []
    # for img_path in test_pic_paths:
    #     img_name = os.path.basename(img_path)
    #     for file_path in file_paths:
    #         valid_img_path = file_path.split(" ")[1]
    #         if img_name in valid_img_path:
    #             img_valid_path.append(img_path)


    img_valid_path = test_pic_paths
    print("valid path ", len(img_valid_path))
    # if not os.path.exists(args.save_txt):
    #     os.makedirs(args.save_txt)
    
    for test_path in img_valid_path:
        # print("path ", test_path)
        avm_img = cv2.imread(test_path)
        model_h = 768
        model_w = 768
        # vertexElements = detect_fisheye(fisheye_img, img_file, onnx_mode_path)
        img_name = os.path.basename(test_path)
        vertexElements = detect_avm_nv12(avm_img, img_name, onnx_mode_path)
        # save_txt_path = os.path.join(args.save_txt, img_name.replace(".jpg", ".txt"))
        line_pt_info = []
        
        for i in range(len(vertexElements)):
            point = vertexElements[i][0]
            orients = vertexElements[i][1]
            line_num = line_num + len(orients)
            line_pt_info.append({"pt":(point[0], point[1])})
            for j in range(len(orients)):
                ori = orients[j]
                stp = (point[0], point[1])
                edp = (int(stp[0] + ori[0]*ori[3]), int(stp[1] + ori[1]*ori[3]))
                line_pt_info.append({"line":(point[0], point[1], edp[0], edp[1])})
        point_num = point_num + len(vertexElements)


        # with open(save_txt_path, "a") as f:
        #     for det_info in line_pt_info:
        #         f.write(f"{json.dumps(det_info)}\n")


        #     print("cur frame line ", len(orients))
        # print("cur frame point ", len(vertexElements))
        
    print("sum point num ", point_num)
    print("sum line num ", line_num)

if __name__ == '__main__':
    main(args)