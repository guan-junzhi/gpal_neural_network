import os
import numpy as np
import cv2
import json
import glob as glob

def draw_json_object(json_path, img):
    line_img = img.copy()
    point_img = img.copy()
    
    colors = [
        (0, 0, 255),    # 红色, T
        (0, 255, 0),    # 绿色, L
        (255, 0, 0),    # 蓝色, I
        (128, 128, 128),
        (255, 255, 0),  # 青色, 斜T
        (0, 255, 255)   # 黄色， 斜L
    ]

    colors_line = [
        (0, 0, 255),    # 红色, entrance_line
        (0, 255, 0),    # 绿色, slot_line
    ]

    keypoint_type_dict = ["T", "L", "I", "U", "Oblique_T", "Oblique_L"]
    line_type_dict = ["entrance_line", "slot_line"]
    json_list = json.load(open(json_path))
    annos = json_list["annotation"]
    keyline_objs = annos["object"]
    # slots = annos["slot"]

    for keypt in keyline_objs:
        color = ""
        if keypt["name"] == "keypoint":
            keypt_type = keypt["keypoint_type"]
            key_pt = keypt["pt"][0]
            for idx_key, key_type in enumerate(keypoint_type_dict):
                if keypt_type == key_type: 
                    color = colors[idx_key]
                    break

            cv2.circle(point_img, (int(float(key_pt['x'])), int(float(key_pt['y']))), 5, color, -1)
        
        if keypt["name"] == "line":
            line_type = keypt["line_type"]
            line_pt = keypt["pt"]
            
            for idx_line, key_type in enumerate(line_type_dict):
                if key_type == line_type: 
                    color = colors_line[idx_line]
                    break
                
            cv2.line(line_img, 
                     (int(float(line_pt[0]['x'])), int(float(line_pt[0]['y']))), 
                     (int(float(line_pt[1]['x'])), int(float(line_pt[1]['y']))), 
                     color, 3)


    return point_img, line_img
                
            
def visul_label(json_root, save_visual_root):
    json_paths = glob.glob(os.path.join(json_root, "*/*/*/*.json"))
    print('files num = ', len(json_paths))
    for json_path in json_paths:
        json_file = json_path.split("/")[-1]
        folder0 = json_path.split("/")[-4]
        folder1 = json_path.split("/")[-3]
        jpg_name = json_file.replace(".json", ".jpg")
        img_path = json_path.replace("label", "avm").replace(".json", ".jpg")
        #========show====================
        img = cv2.imread(img_path)
        print("img_path ", img_path)
        if img is None:
            print("{} img is None ", img_path)
      
        show_img, line_img = draw_json_object(json_path, img)
        # show_old_img = 
        show_three_img = np.hstack((img, show_img, line_img))
        save_folder = os.path.join(save_visual_root)
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
        visual_img_path = os.path.join(save_folder, jpg_name)
        cv2.imwrite(visual_img_path, show_three_img)

def visual_old_format(json_root, save_visual_root):
    json_paths = glob.glob(os.path.join(json_root, "*/*/*/*.json"))
    for json_path in json_paths:
        if 'new_json' not in json_path:
            continue
        json_file = json_path.split("/")[-1]
        folder0 = json_path.split("/")[-3]
        folder1 = json_path.split("/")[-2]
        jpg_name = json_file.replace(".json", ".jpg")
        img_path = json_path.replace(".json", ".jpg").replace("new_json", "src")
        #========show====================
        img = cv2.imread(img_path)
        print("img_path ", img_path)
        if img is None:
            print("{} img is None ", img_path)
      
        show_img, line_img = draw_json_object(json_path, img)
        # show_old_img = 
        show_three_img = np.hstack((img, show_img, line_img))
        # save_folder = os.path.join(save_visual_root, folder0, folder1)
        # if not os.path.exists(save_folder):
        #     os.makedirs(save_folder)
        save_folder = save_visual_root
        visual_img_path = os.path.join(save_folder, jpg_name)
        cv2.imwrite(visual_img_path, show_three_img)


if __name__ == "__main__":
    json_root = "/home/gpal/gpal_work/ParkingSlot/parking_slot/datasets_tmp/0817/20250817_commit_valid2"
    save_visual_root = json_root + "_visual"
    if not os.path.exists(save_visual_root):
        os.makedirs(save_visual_root)
    visul_label(json_root, save_visual_root)
    print('done')
    # visual_old_format(json_root, save_visual_root)
