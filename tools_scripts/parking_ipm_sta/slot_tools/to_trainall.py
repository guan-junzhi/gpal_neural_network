import os
import glob

flag = 0
flag_str = ['Train', 'Test']
txt_str = ['train_all_0928.txt', 'test_all.txt']

root_path = "/data/ai_group/datasets/bev_park/train_test_dataset/PointLineData_" + flag_str[flag]
save_path = os.path.join(root_path, txt_str[flag])

no_need_folder = ["20251020", "20251027", "20251104_valid", "20251110"]
txt_file = open(save_path, 'w')

file_list = glob.glob(os.path.join(root_path, "*/*/*/*/*.json"))
print("file len ", len(file_list))
for file_path in file_list:
    flag = True
    for folder in no_need_folder:
        if folder in file_path:
            print("no need ", file_path)
            flag = False
            break
    if flag == True:
        file_path = file_path.replace(root_path + '/', '')
        txt_file.write(file_path + "\n")
txt_file.close()
         




