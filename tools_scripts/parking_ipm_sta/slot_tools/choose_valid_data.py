import os
import shutil
import glob

def choose_valid_json_pic():
    data_selected = "/home/gpal/gpal_work/ParkingSlot/parking_slot/datas/selected_pointline_train/src_visual/2"
    flag = 1
    if flag == 1:
        pics_path = glob.glob(os.path.join(data_selected, "*.jpg"))
    else:
        pics_path = glob.glob(os.path.join(data_selected, "*/*.jpg"))

    ori_src_root = "/home/gpal/gpal_work/ParkingSlot/parking_slot/datas/selected_pointline_train/src"
    ori_json_root = "/home/gpal/gpal_work/ParkingSlot/parking_slot/datas/selected_pointline_train/new_json"

    save_root = "/home/gpal/gpal_work/ParkingSlot/parking_slot/datasets/selected_pointline_train_sec_choosed"
    # save_src_parent = os.path.join(save_root, "src")
    # save_json_parent = os.path.join(save_root, "json")
    cnt = 0

    for valid_pic in pics_path:
        if flag == 0:
            valid_folders_list = valid_pic.split("/")[-3:]
            valid_pic_folders = os.path.join(valid_folders_list[0], valid_folders_list[1], valid_folders_list[2])
            valid_json_folders = valid_pic_folders.replace(".jpg", ".json")
        else:
            file_name = valid_pic.split("/")[-1]
            file_paths = glob.glob(os.path.join(ori_src_root, "*/*/*.jpg"))
            for file_path in file_paths:
                file_paths_pic_name = file_path.split('/')[-1]
                if file_name == file_paths_pic_name:
                    valid_folders_list = file_path.split("/")[-3:]
                    valid_pic_folders = os.path.join(valid_folders_list[0], valid_folders_list[1], valid_folders_list[2])
                    valid_json_folders = valid_pic_folders.replace(".jpg", ".json")

        src_img_path = os.path.join(ori_src_root, valid_pic_folders)
        src_json_path = os.path.join(ori_json_root, valid_json_folders)
        
        dst_src_path = os.path.join(save_root, valid_folders_list[0], valid_folders_list[1], 'avm', valid_folders_list[2])
        dst_json_path = os.path.join(save_root, valid_folders_list[0], valid_folders_list[1], 'label', valid_folders_list[2].replace(".jpg", ".json"))
        dst_src_dir = os.path.dirname(dst_src_path)
        if not os.path.exists(dst_src_dir):
            os.makedirs(dst_src_dir)

        dst_json_dir = os.path.dirname(dst_json_path)
        if not os.path.exists(dst_json_dir):
            os.makedirs(dst_json_dir)
        if not os.path.exists(src_img_path) or not os.path.exists(src_json_path):
            print("no exist img ", src_img_path)
            print("no exist json ", src_json_path)
            continue
        cnt = cnt + 1
        print("cnt ", cnt)
        shutil.copy2(src_img_path, dst_src_dir)  
        shutil.copy2(src_json_path, dst_json_dir)


def choose_good_json_use_file_name(file_name, json_ori_path, img_ori_path, save_good_path):
    json_name = file_name.replace('.jpg', '.json')

    json_ori_list = glob.glob(os.path.join(json_ori_path, "*/*/*.json"))
    num = len(json_ori_list)
    print('jsons num ', num)
    for json_path in json_ori_list:
        if json_name in json_path:
            save_json_path = json_path.replace(json_ori_path, save_good_path)
            img_path = json_path.replace(json_ori_path, img_ori_path).replace('.json', '.jpg')
            save_parent = os.path.dirname(save_json_path)
            
            save_json_parent = os.path.join(save_parent, 'label')
            save_img_parent = os.path.join(save_parent, 'avm')
            save_img_path = os.path.join(save_img_parent, file_name)
            save_json_path = os.path.join(save_json_parent, json_name) #update save file format
            if not os.path.exists(save_json_parent):
                os.makedirs(save_json_parent)
            if not os.path.exists(save_img_parent):
                os.makedirs(save_img_parent)

            shutil.copy2(json_path, save_json_path)
            shutil.copy2(img_path, save_img_path)
            break

def choose_good_json_use_file_name_0730format(file_name, json_ori_path, save_good_path):
    json_name = file_name.replace('.jpg', '.json')

    json_ori_list = glob.glob(os.path.join(json_ori_path, "*/*/*/*.json"))
    num = len(json_ori_list)
    # print('jsons num ', num)
    for json_path in json_ori_list:
        if json_name in json_path:
            save_json_path = json_path.replace(json_ori_path, save_good_path)
            img_path = json_path.replace('label', 'avm').replace('.json', '.jpg')
            save_parent = os.path.dirname(save_json_path)
            
            # save_json_parent = os.path.join(save_parent, 'label')
            # save_img_parent = os.path.join(save_parent, 'avm')
            save_img_path = img_path.replace(json_ori_path, save_good_path)
            save_json_path = json_path.replace(json_ori_path, save_good_path)
            save_img_parent = os.path.dirname(save_img_path)
            save_json_parent = os.path.dirname(save_json_path)
            
            if not os.path.exists(save_json_parent):
                os.makedirs(save_json_parent)
            if not os.path.exists(save_img_parent):
                os.makedirs(save_img_parent)

            shutil.copy2(json_path, save_json_path)
            shutil.copy2(img_path, save_img_path)
            break

def choose_good_json_0730format():
    visual_good_path = '/home/gpal/gpal_work/ParkingSlot/parking_slot/datasets_tmp/0817/backpoint_choose/bad_new_json_visual'
    json_ori_path = '/home/gpal/gpal_work/ParkingSlot/parking_slot/datasets_tmp/0817/backpoint_choose/20250817_commit_unvlid_backpoint'
    save_good_path = '/home/gpal/gpal_work/ParkingSlot/parking_slot/datasets_tmp/0817/backpoint_choose/20250817_commit_need_manu_change_json'

    # visual_imgs = os.listdir(visual_good_path)
    visual_imgpath_list = glob.glob(os.path.join(visual_good_path, "*.jpg"))
    print("valid num ", len(visual_imgpath_list))
    for imgpath in visual_imgpath_list:
        img_name = os.path.basename(imgpath)
        # choose_good_json_use_file_name(img_name, json_ori_path, img_ori_path, save_good_path)
        choose_good_json_use_file_name_0730format(img_name, json_ori_path, save_good_path)
    print('end')

if __name__ == '__main__':
    choose_good_json_0730format()
    # choose_valid_json_pic()

