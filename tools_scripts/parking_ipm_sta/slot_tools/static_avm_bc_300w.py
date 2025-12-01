
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

def parse_args():
    parser = argparse.ArgumentParser(description="slot_onnx")
    parser.add_argument("--onnx_path", default="/home/jovyan/gpal_neural_network/20251022_log/20251022_int16_avm_quantized_model.bc", type=str)
    parser.add_argument("--label_path", default="/data/ai_group/datasets/bev_park/train_test_dataset/PointLineData_Test/pointline.txt", type=str)
    parser.add_argument("--save_path", default="/data/ai_group/datasets/bev_park/park_slot_jira/1114_jira/res/1114_serveravm_quantized_model_300w_quantized_20251022model_bc_res_static_thr5", type=str)
    parser.add_argument("--save_txt", default="300w_quat_serveravm_det_res_static_thr5", type=str)
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

def detect(orig_img, model, model_h, model_w, device):
    # orig_img = cv2.imread(imgfile)
    rawh, raww, _ = orig_img.shape

    img = preprocess(orig_img, model_h, model_w)
    # if do_save_img:
    #     cv2.imwrite('./img.jpg', img)
    model.eval()
    with torch.no_grad():
        heatmapValueTensor, vecmapValueTensor, = model(img.to(device))

    #get heatmap w h ch 
    _, _, h, w = heatmapValueTensor.shape
    heatmapValue = heatmapValueTensor[0,0,:,:].cpu().detach().numpy()
    vecmapValue = vecmapValueTensor[0,0,:,:].cpu().detach().numpy()
    # point_img, line_img = getFeatureMap(heatmapValue, vecmapValue, w, h)
    # savePath = imgfile.replace(test_dir, out_dir)
    # print(savePath)
    # if do_save_img:
    #     cv2.imwrite(savePath.replace('.jpg','_point_map.jpg'), point_img)
    #     cv2.imwrite(savePath.replace('.jpg','_line_map.jpg'), line_img)
    SlotDetInstance = HeatMap(w, h)
    vertexElements = SlotDetInstance.doProc(heatmapValue, vecmapValue)
    # SlotDetInstance.drawVE(heatmap_img, savePath.replace('.jpg','_draw.jpg'))
    # SlotComposeInstance = Matcher(w, h)
    # slotsList = []
    # SlotComposeInstance.ComposeSlots(vertexElements)
    # if do_save_img:
        # SlotComposeInstance.DrawSlots(matchpair_img, savePath.replace('.jpg','_slot.jpg'))
    # return vertexElements, slotsList


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
    # SlotDetInstance.drawVE(show_img, savename.replace('.jpg','_draw.jpg'))
    return vertexElements

def detect_point(ori_img,model, model_h, model_w, device):
    img = preprocess(ori_img, model_h, model_w)
    # if do_save_img:
    #     cv2.imwrite('./img.jpg', img)
    model.eval()
    with torch.no_grad():
        heatmapValueTensor = model(img.to(device))

    #get heatmap w h ch 
    _, _, h, w = heatmapValueTensor.shape
    heatmapValue = heatmapValueTensor[0,0,:,:].cpu().detach().numpy()
    # vecmapValue = vecmapValueTensor[0,0,:,:].cpu().detach().numpy()
    # point_img, line_img = getFeatureMap(heatmapValue, vecmapValue, w, h)
    # savePath = imgfile.replace(test_dir, out_dir)
    # print(savePath)
    # if do_save_img:
    #     cv2.imwrite(savePath.replace('.jpg','_point_map.jpg'), point_img)
    #     cv2.imwrite(savePath.replace('.jpg','_line_map.jpg'), line_img)
    SlotDetInstance = HeatMap(w, h)
    vertexElements = SlotDetInstance.doProcPoint(heatmapValue)
    # SlotDetInstance.drawVE(heatmap_img, savePath.replace('.jpg','_draw.jpg'))
    # SlotComposeInstance = Matcher(w, h)
    slotsList = []
    # SlotComposeInstance.ComposeSlots(vertexElements)
    # if do_save_img:
        # SlotComposeInstance.DrawSlots(matchpair_img, savePath.replace('.jpg','_slot.jpg'))
    return vertexElements, slotsList
############################################################
# def initStatPack():
#     StatPack =  {}
#     StatPack['point_pixel_error_sum'] = 0.0
#     StatPack['angle_error'] = 0.0
#     StatPack['point_total_num'] = 0
#     StatPack['point_true_num'] = 0
#     StatPack['point_miss_num'] = 0
#     StatPack['point_false_num'] = 0
#     StatPack['line_total_num'] = 0
#     StatPack['line_true_num'] = 0
#     StatPack['line_miss_num'] = 0
#     StatPack['line_false_num'] = 0
#     StatPack['point_det_num'] = 0
#     StatPack['line_det_num'] = 0
#     return StatPack

# def updatePack(StatPack, resultPack):
#     print("StatPack['point_pixel_error_sum'] is: {}; resultPack['point_error'] is: {}".format(StatPack['point_pixel_error_sum'], resultPack['point_error']))
#     print("StatPack['angle_error'] is: {}; resultPack['angle_error'] is: {}".format(StatPack['angle_error'], resultPack['angle_error']))
#     print("StatPack['point_total_num'] is: {}; resultPack['point_total_num'] is: {}".format(StatPack['point_total_num'], resultPack['point_total_num']))
#     print("StatPack['point_true_num'] is: {}; resultPack['point_true_num'] is: {}".format(StatPack['point_true_num'], resultPack['point_true_num']))
#     print("StatPack['point_miss_num'] is: {}; resultPack['point_miss_num'] is: {}".format(StatPack['point_miss_num'], resultPack['point_miss_num']))
#     print("StatPack['point_false_num'] is: {}; resultPack['point_false_num'] is: {}".format(StatPack['point_false_num'], resultPack['point_false_num']))
#     print("StatPack['line_total_num'] is: {}; resultPack['line_total_num'] is: {}".format(StatPack['line_total_num'], resultPack['line_total_num']))
#     print("StatPack['line_true_num'] is: {}; resultPack['line_true_num'] is: {}".format(StatPack['line_true_num'], resultPack['line_true_num']))
#     print("StatPack['line_miss_num'] is: {}; resultPack['line_miss_num'] is: {}".format(StatPack['line_miss_num'], resultPack['line_miss_num']))
#     print("StatPack['line_false_num'] is: {}; resultPack['line_false_num'] is: {}".format(StatPack['line_false_num'], resultPack['line_false_num']))
    
#     print("StatPack['point_det_num'] is: {}; resultPack['point_det_num'] is: {}".format(StatPack['point_det_num'],resultPack['point_det_num'] ))
#     print("StatPack['line_de_num'] is: {}; resultPack['line_det_num'] is: {}".format(StatPack['line_det_num'], resultPack['line_det_num']))
    
#     StatPack['point_pixel_error_sum'] = StatPack['point_pixel_error_sum'] + resultPack['point_error']
#     StatPack['angle_error'] = StatPack['angle_error'] + resultPack['angle_error']
#     StatPack['point_total_num'] = StatPack['point_total_num'] + resultPack['point_total_num']
#     StatPack['point_true_num'] = StatPack['point_true_num'] + resultPack['point_true_num']
#     StatPack['point_miss_num'] = StatPack['point_miss_num'] + resultPack['point_miss_num']
#     StatPack['point_false_num'] = StatPack['point_false_num'] + resultPack['point_false_num']
#     StatPack['line_total_num'] = StatPack['line_total_num'] + resultPack['line_total_num']
#     StatPack['line_true_num'] = StatPack['line_true_num'] + resultPack['line_true_num']
#     StatPack['line_miss_num'] = StatPack['line_miss_num'] + resultPack['line_miss_num']
#     StatPack['line_false_num'] = StatPack['line_false_num'] + resultPack['line_false_num']

#     StatPack['point_det_num'] = StatPack['point_det_num'] + resultPack['point_det_num']
#     StatPack['line_det_num'] = StatPack['line_det_num'] + resultPack['line_det_num']

# @log_to_file(os.path.join(args.save_path,"static_pointline.log"))
# def outputStat(StatPack):
#     if StatPack['point_total_num'] != 0:
#         StatPack['point_pixel_error_sum'] = StatPack['point_pixel_error_sum'] / StatPack['point_total_num']
#     if StatPack['line_total_num'] != 0:
#         StatPack['angle_error'] = StatPack['angle_error'] / StatPack['line_total_num']
#     if StatPack['point_total_num'] != 0:
#         StatPack['point_recall'] = StatPack['point_true_num']*1.0/ StatPack['point_total_num']
#         StatPack['point_miss_rate'] = StatPack['point_miss_num'] * 1.0 / StatPack['point_total_num']
#     if StatPack['point_det_num'] != 0:
#         StatPack['point_precision'] = StatPack['point_true_num'] * 1.0 / StatPack['point_det_num']
#         StatPack['point_FDR'] = StatPack['point_false_num'] * 1.0 / StatPack['point_det_num']
    
#     if StatPack['line_total_num'] != 0:
#         StatPack['line_recall'] = StatPack['line_true_num']*1.0/ StatPack['line_total_num']
#         StatPack['line_miss_rate'] = StatPack['line_miss_num'] * 1.0 / StatPack['line_total_num']
#     if StatPack['line_det_num'] != 0:
#         StatPack['line_precision'] = StatPack['line_true_num'] * 1.0 / StatPack['line_det_num']
#         StatPack['line_FDR'] = StatPack['line_false_num'] * 1.0 / StatPack['line_det_num']

#     printTitleInfo()
#     print ("--->point count result : ")
    
#     print("point recall = ", StatPack['point_recall'])
#     print("point_precision = ", StatPack['point_precision'])
#     print ("point average pixel error : ", StatPack['point_pixel_error_sum'])
#     print("point FDR: ", StatPack['point_FDR'])
#     print("point miss rate = ", StatPack['point_miss_rate'])

#     print ("total ann point numbers = ", StatPack['point_total_num'])
#     print("point_det_num = ", StatPack['point_det_num']) 
#     print("point_true_num = ", StatPack['point_true_num']) 
#     print("point_false_num = ", StatPack['point_false_num']) 
#     print("point miss num = ", StatPack['point_miss_num'])
    

#     print ("--->line count result : ")
#     print ("line recall = ", StatPack['line_recall'])
#     print("line_precision = ", StatPack['line_precision'])
#     print ("point line angle error degree : ", np.degrees(StatPack['angle_error']))
#     print("line FDR = ", StatPack['line_FDR']) 
#     print("line miss rate = ", StatPack['line_miss_rate'])

#     print ("total ann line numbers = ", StatPack['line_total_num'])
#     print("line_det_num = ", StatPack['line_det_num'])
#     print("line_true_num = ", StatPack['line_true_num'])
#     print("line_false_num = ", StatPack['line_false_num']) 
#     print("line miss num = ", StatPack['line_miss_num'])
    
#     return 
    
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
    if not os.path.exists(args.save_txt):
        os.makedirs(args.save_txt)
    ann_dir = "/data/ai_group/datasets/bev_park/train_test_dataset/PointLineData_Test"
    StatPackage = initStatPack()
    model_name = os.path.basename(args.onnx_path)
    for test_path in pic_path_list:
        # print("path ", test_path)
        avm_img = cv2.imread(os.path.join(ann_dir,test_path))
        model_h = 768
        model_w = 768
        # vertexElements = detect_fisheye(fisheye_img, img_file, onnx_mode_path)
        img_name = os.path.basename(test_path)
        vertexElements = detect_avm_nv12(avm_img, img_name, onnx_mode_path)
        save_txt_path = os.path.join(args.save_txt, img_name.replace(".jpg", ".txt"))
        line_pt_info = []
        #         print("img_file ", front_img_path)
        # print("vertexElements ", vertexElements)
        label_img_raww = 768
        label_img_rawh = 768
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


      

        #     print("cur frame line ", len(orients))
        # print("cur frame point ", len(vertexElements))
        
#         print("img_file ", front_img_path)
#         print("vertexElements ", vertexElements)
#         label_img_raww = 768
#         label_img_rawh = 768
#         sw, sh = getImageSizeScale(label_img_raww, label_img_rawh, model_w, model_h)
#         valid_folders_list = front_img_path.split("/")[-4:]
#         valid_pic_folders = os.path.join(valid_folders_list[0], valid_folders_list[1], valid_folders_list[3])
#         #do Heatmap Statistics 
#         txtPath = os.path.join(ann_dir, valid_pic_folders.replace('.jpg', '.txt'))
#         if not os.path.exists(txtPath):
#             print("!!!!{} json path is not exist ".format(txtPath))
#         print("txtPath is: {}".format(txtPath))
#         labelInstance = TXTLabelLoader(sw, sh)
#         # heatmapResultPack = labelInstance.doHeatmapStatistics_json_anno(jsonPath, vertexElements)
#         heatmapResultPack = labelInstance.doHeatmapStatistics(txtPath, vertexElements)
#         updatePack(StatPackage, heatmapResultPack)
#     #do Slot Match Statistics

# outputStat(StatPackage)

if __name__ == '__main__':
    main(args)