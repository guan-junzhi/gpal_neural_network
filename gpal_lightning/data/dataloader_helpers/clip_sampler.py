from torch.utils.data import Sampler
import numpy as np
from numpy.random import randint

def DatalistByclip(datalist, key = "scene", ret_idx = False):
    datalist_by_clip = {}
    for ele_i, ele in enumerate(datalist):
        clip_key = ele[key]
        if clip_key not in datalist_by_clip:
            datalist_by_clip[clip_key] = []
        if ret_idx:
            datalist_by_clip[clip_key].append(ele_i)
        else:
            datalist_by_clip[clip_key].append(ele)
    return datalist_by_clip


class ClipSampler(Sampler):
    def __init__(self, data_source, default_resample_len=100, batch_size=8, length_range=[5, 15], rank = 0):
        self.data_source = data_source
        self.indices = list(range(len(self.data_source)))  # 全局索引
        self.current_index = 0  # 当前索引位置
        self.default_resample_len = default_resample_len
        self.batch_size = batch_size
        self.length_range = length_range
        self.rank = rank

    def RandomByClip(self, datalist, length_range=[5, 15], batch_size=8):
        epoch_len = self.default_resample_len
        
        datalist_by_clip = DatalistByclip(datalist, "scene", True)  # 按clip分组
        clip_key_list = list(datalist_by_clip.keys())
        
        flatten_idxs = np.zeros([epoch_len, 2], dtype = np.int32) -1
        
        for i in range(epoch_len):
            if (flatten_idxs[i, 0] < 0) or (flatten_idxs[i, 1] < 0):
                clip_idx = randint(0, len(clip_key_list))  # 哪个clip的位置索引
                clip_idx = [randint(0, len(clip_key_list)) for _ in range(self.rank+1)][-1]
                
                clip_key = clip_key_list[clip_idx]
                frames_in_clip = datalist_by_clip[clip_key]
                # frame_start_idx = randint(0, len(frames_in_clip)-length_range[0])
                
                if len(frames_in_clip) < length_range[0]:
                    continue
                
                frame_start_idx = [
                    randint(0, len(frames_in_clip)-length_range[0]) for _ in range(self.rank+1)][-1]
                
                # frame_end_idx = frame_start_idx + \
                #     randint(length_range[0], length_range[1])
                frame_end_idx = frame_start_idx + \
                    [randint(length_range[0], length_range[1])
                     for _ in range(self.rank+1)][-1]
                frame_end_idx = min(frame_end_idx, len(frames_in_clip) - 1)
                for j in range(frame_end_idx - frame_start_idx):
                    if (i + j * batch_size) >= epoch_len:
                        break
                    flatten_idxs[i + j * batch_size, 0] = clip_idx
                    flatten_idxs[i + j * batch_size, 1] = frame_start_idx + j

        print('flatten_idxs have -1:', np.sum(flatten_idxs == -1))
        dataset = [datalist_by_clip[clip_key_list[ele[0]]][ele[1]]
                for ele in flatten_idxs]
        # print(dataset)
        # exit(1)
        return dataset

    def __iter__(self):
        print(len(self.data_source), self.data_source[0])
        self.indices = self.RandomByClip(
            self.data_source, self.length_range, self.batch_size)
        print(self.indices[:100])

        return iter(self.indices)  # 直接返回顺序索引的迭代器

    def __len__(self):
        print(f"ClipSampler rank: {self.rank} __len__: {len(self.indices)}")
        return len(self.indices)  # 返回总样本数
