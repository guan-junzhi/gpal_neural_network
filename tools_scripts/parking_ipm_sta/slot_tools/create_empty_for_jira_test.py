import os
import numpy as np
import cv2
import json
import glob as glob
def cal_point_line_match(key_pt, line_pt):
    res = False
    dx = key_pt[0] - line_pt[0]
    dy = key_pt[1] - line_pt[1]
    if abs(dx) < 10 and abs(dy) < 10:
        res = True
    return res

def json_txt(json_root, txt_root):
    json_paths = glob.glob(os.path.join(json_root, "*/*/*/*/*.json"))
    print('label ', len(json_paths))
    for json_path in json_paths:
        json_file = json_path.split("/")[-1]
        valid_folders_list = json_path.split("/")[-4:]
        valid_pic_folders = os.path.join(valid_folders_list[0], valid_folders_list[1], valid_folders_list[3])
        #do Heatmap Statistics 
        txtPath = os.path.join(txt_root, valid_pic_folders.replace('.json', '.txt'))
        txt_parent = os.path.dirname(txtPath)
        if not os.path.exists(txt_parent):
            os.makedirs(txt_parent)
        json_list = json.load(open(json_path))
        annos = json_list["annotation"]
        keyline_objs = annos["object"]
        # slots = annos["slot"]
        txt_file = open(txtPath, 'w')
        key_pts = []
        lines = []
        txt_file.write('imgwh {} {}\n'.format(annos['imgsize']['width'], annos['imgsize']['height']))

        for keypt in keyline_objs:
            if keypt["name"] == "keypoint":
                keypt_type = keypt["keypoint_type"]
                key_pt = keypt["pt"][0]
                key_pt_xy = [int(float(key_pt['x'])), int(float(key_pt['y']))]
                key_pts.append(key_pt_xy)
            if keypt["name"] == "line":
                line_type = keypt["line_type"]
                line_pt = keypt["pt"]
                pt0 = [int(float(line_pt[0]['x'])), int(float(line_pt[0]['y']))]
                pt1 = [int(float(line_pt[1]['x'])), int(float(line_pt[1]['y']))]
                lines.append([pt0, pt1])
        for pt in key_pts:
            txt_file.write('{} {} '.format(pt[0], pt[1]))
            for line in lines:
                res = cal_point_line_match(pt, line[0])
                if res:
                    txt_file.write('{} {} '.format(line[1][0], line[1][1]))
                else:
                    res = cal_point_line_match(pt, line[1])
                    if res:
                        txt_file.write('{} {} '.format(line[0][0], line[0][1]))
            txt_file.write('\n')
        txt_file.close()

def create_empty_txt(img_root, txt_root):
    json_paths = glob.glob(os.path.join(img_root, "*/*/*.jpg"))
    print('label ', len(json_paths))
    for json_path in json_paths:
        json_file = json_path.split("/")[-1]
        valid_folders_list = json_path.split("/")[-3:]
        valid_pic_folders = os.path.join(valid_folders_list[0], valid_folders_list[1], valid_folders_list[2])
        #do Heatmap Statistics 
        txtPath = os.path.join(txt_root, valid_pic_folders.replace('.jpg', '.txt'))
        txt_parent = os.path.dirname(txtPath)
        if not os.path.exists(txt_parent):
            os.makedirs(txt_parent)

        txt_file = open(txtPath, 'w')
        key_pts = []
        lines = []
        txt_file.write('imgwh {} {}\n'.format(1920,1635))
        txt_file.close()


def convert_to_txt(root, folder_num=4):
    if folder_num == 4:
        img_paths = glob.glob(os.path.join(root, "*/*/*/*.jpg"))
    if folder_num == 3:
        img_paths = glob.glob(os.path.join(root, "*/*/*.jpg"))
    txt_file = open(os.path.join(root, 'pointline.txt'), "w")
    print("img lenth ", len(img_paths))
    for img_path in img_paths:
        img_path_split = img_path.split('/')
        jpg_img_path_list = img_path_split[-3:]
        point_line_txt_path = os.path.join("pointline_txt", jpg_img_path_list[0], jpg_img_path_list[1], jpg_img_path_list[-1].replace(".jpg", ".txt"))
        jpg_img_path = img_path.replace(root + '/', ' ')
        txt_path = os.path.join(root, point_line_txt_path)
        if not os.path.exists(txt_path):
            print("!!!! txt not exits")
            continue

        line_str = point_line_txt_path + jpg_img_path + '\n'
        txt_file.writelines(line_str)
    
    txt_file.close()

if __name__ == '__main__':
    jira = False
    if jira == False:
        json_root = '/home/gpal/gpal_work/ParkingSlot/parking_slot/datas/train_test_data/PointLineData_Test'
        txtroot = os.path.join(json_root, 'pointline_txt')
        json_txt(json_root, txtroot)
        convert_to_txt(json_root, folder_num=4)
    else:
        imgroot = "/media/gpa/data/parkslot_datas/jira/水平_垂直1114_泊入差/car_and_server_avm_test_static/server"
        txtroot = os.path.join(imgroot, 'pointline_txt')
        create_empty_txt(imgroot, txtroot)
        convert_to_txt(imgroot, folder_num=3)
    print('Done')
    

                    
    