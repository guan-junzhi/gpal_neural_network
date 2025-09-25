import os
import cv2
import json
import numpy as np
from tqdm import tqdm
import glob

import multiprocessing
multiprocessing.set_start_method('spawn', force = True)
from multiprocessing import Pool

def GenVideoFromImages(input_dir, output_file):

    image_list = os.listdir(input_dir)
    print(image_list)
    # exit(1)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    output_video = None
    image_list.sort()
    for d in tqdm(image_list):
        image = cv2.imread(os.path.join(input_dir, d), cv2.IMREAD_UNCHANGED)
        size = (int(image.shape[1]),int(image.shape[0])) 
        if output_video is None:
            output_video = cv2.VideoWriter(output_file, fourcc, 15.0, size, True)    


        output_video.write(image)

if __name__ == "__main__":
    dir = "online_pred_vis"

    todo_list = glob.glob(dir+'/*')
    

    pool = Pool(processes=16)
    for subdir in todo_list:
        output_file = subdir+".mp4"
        if not os.path.exists(output_file):
            
            # input_dir = os.path.join(dir, subdir)
            input_dir = subdir
            print(output_file, input_dir)
            GenVideoFromImages(input_dir, output_file)
            # pool.apply_async(GenVideoFromImages, (input_dir, output_file))
    #     # exit(1)

        
    # for seq in dirs:
        
    pool.close()
    pool.join()