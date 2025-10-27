import os
import numpy as np
import random
import shutil
import glob
data_root = "/home/gpal/gpal_work/ParkingSlot/parking_slot/datasets_tmp/0817/"
json_path_list = glob.glob(os.path.join(data_root, "*/*/*/*/*.json"))

train_data_dir = '/home/gpal/gpal_work/ParkingSlot/parking_slot/datas/train_test_data/PointLineData_Train'
test_data_dir = '/home/gpal/gpal_work/ParkingSlot/parking_slot/datas/train_test_data/PointLineData_Test'
if not os.path.exists(train_data_dir):
    os.makedirs(train_data_dir)

if not os.path.exists(test_data_dir):
    os.makedirs(test_data_dir)

lenth = len(json_path_list)
print("file lenth ", lenth)
random.shuffle(json_path_list)
split_idx = int(0.8 * lenth)
print("split_idx ", split_idx)

train_files = json_path_list[:split_idx]
test_files = json_path_list[split_idx:]
print(f'train= {len(train_files)}, test= {len(test_files)}')
for json_path in train_files:
    train_json_path = json_path.replace(data_root, train_data_dir)
    img_path = json_path.replace('label', 'avm').replace('.json', '.jpg')
    train_pic_path = img_path.replace(data_root, train_data_dir)
    train_json_folder = os.path.dirname(train_json_path)
    train_pic_folder = os.path.dirname(train_pic_path)
    if not os.path.exists(train_json_folder):
        os.makedirs(train_json_folder)    

    if not os.path.exists(train_pic_folder):
        os.makedirs(train_pic_folder) 
    shutil.copyfile(json_path, train_json_path)
    shutil.copyfile(img_path, train_pic_path)

for json_path in test_files:
    test_json_path = json_path.replace(data_root, test_data_dir)
    img_path = json_path.replace('label', 'avm').replace('.json', '.jpg')
    test_pic_path = img_path.replace(data_root, test_data_dir)
    test_json_folder = os.path.dirname(test_json_path)
    test_pic_folder = os.path.dirname(test_pic_path)
    if not os.path.exists(test_json_folder):
        os.makedirs(test_json_folder)    

    if not os.path.exists(test_pic_folder):
        os.makedirs(test_pic_folder) 

    shutil.copyfile(json_path, test_json_path)
    shutil.copyfile(img_path, test_pic_path)
print("end")
