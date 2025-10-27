import os

flag = 0
flag_str = ['Train', 'Test']
txt_str = ['train_all.txt', 'test_all.txt']

root_path = "/data/ai_group/datasets/bev_park/train_test_dataset/PointLineData_" + flag_str[flag]
save_path = os.path.join(root_path, txt_str[flag])

txt_file = open(save_path, 'w')
for root, folder, file in os.walk(root_path):
    for one_file in file:
        if (one_file.split('.')[-1] == 'json'):
            file_path = os.path.join(root, one_file)
            print("abs path ", file_path)
            file_path = file_path.replace(root_path + '/', '')
            print("part path ", file_path)
            txt_file.write(file_path + "\n")
txt_file.close()




