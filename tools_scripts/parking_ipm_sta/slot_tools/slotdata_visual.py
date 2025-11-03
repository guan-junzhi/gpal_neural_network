import os
import torch
import numpy as np
import torch.utils.data as data
from PIL import Image
import cv2
import xml.etree.ElementTree as ET
import json

import cv2
import numpy as np
from typing import List, Tuple, Optional
from pathlib import Path
current_path = Path(__file__).resolve().parent
parent_path = current_path.parent
import sys
path_ = os.path.expanduser("/home/jovyan/gpal_neural_network")
sys.path.append(path_)
import transforms as T

def heatmap_to_point(heatmap: np.ndarray) -> Tuple[int, int]:
    """将热力图转换为关键点坐标（最大值点）"""
    idx = np.unravel_index(np.argmax(heatmap), heatmap.shape)
    return (idx[1], idx[0])  # (x, y) 顺序

def overlay_heatmap(
    image: np.ndarray, 
    heatmap: np.ndarray, 
    alpha: float = 0.5, 
    colormap: int = cv2.COLORMAP_JET,
    point_radius: int = 5,
    point_color: Tuple[int, int, int] = (0, 0, 255),
    show_text: bool = True
) -> np.ndarray:
    """
    将热力图叠加到原始图像上
    
    参数:
        image: 原始图像 (H, W, 3)，BGR格式
        heatmap: 热力图 (H, W) 或 (H, W, 1)
        alpha: 叠加透明度
        colormap: OpenCV颜色映射
        point_radius: 关键点半径
        point_color: 关键点颜色 (B, G, R)
        show_text: 是否显示关键点坐标
        
    返回:
        叠加后的图像 (H, W, 3)
    """
    # 确保热力图维度正确
    if heatmap.ndim == 3:
        heatmap = heatmap.squeeze(2)
    
    # 调整热力图大小以匹配原始图像
    # heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
    
    # 将热力图归一化到0-255
    heatmap_normalized = cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    # 应用颜色映射
    heatmap_colored = cv2.applyColorMap(heatmap_normalized, colormap)
    
    # 叠加热力图到原始图像
    overlaid = cv2.addWeighted(image, 1 - alpha, heatmap_colored, alpha, 0)
    
    
    # 显示坐标文本
    # if show_text:
    #     cv2.putText(overlaid, f"({x}, {y})", (x+10, y-10),
    #                 cv2.FONT_HERSHEY_SIMPLEX, 0.5, point_color, 2)
    
    return overlaid

def preprocess_img(avm_img):
    img_tensor = torch.tensor(avm_img) / 255.0
    img_tensor = img_tensor.permute(2, 0, 1)
    # mean = torch.tensor([0.481093804, 0.457524588, 0.407870549]).view(3, 1, 1)
    # std = torch.tensor([1.0, 1.0, 1.0]).view(3, 1, 1)
    # normalized_tensor = (img_tensor - mean) / std  # 应用归一化公式
  
    return img_tensor

class VOCSegmentation(data.Dataset):
    def __init__(self, root, transforms=None, txt_name: str = "train.txt"):
        super(VOCSegmentation, self).__init__()
        # assert year in ["2007", "2012"], "year must be in ['2007', '2012']"
        # root = os.path.join(voc_root, "VOCdevkit", f"VOC{year}")
        # assert os.path.exists(root), "path '{}' does not exist.".format(root)
        image_dir = os.path.join(root, 'img')
        mask_dir = os.path.join(root, 'ground')

        txt_path = os.path.join(root, txt_name)
        assert os.path.exists(txt_path), "file '{}' does not exist.".format(txt_path)
        with open(os.path.join(txt_path), "r") as f:
            file_names = [x.strip() for x in f.readlines() if len(x.strip()) > 0]
        #!!!
        self.images = [os.path.join(image_dir, x + ".jpg") for x in file_names]#[:10000]
        self.masks = [os.path.join(mask_dir, x + ".png") for x in file_names]#[:10000]
        assert (len(self.images) == len(self.masks))
        self.transforms = transforms

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is the image segmentation.
        """
        img = Image.open(self.images[index]).convert("RGB") #!!!
        target = Image.open(self.masks[index])

        if self.transforms is not None:
            img, target = self.transforms(img, target)
        #print(target)
        return img, target

    def __len__(self):
        return len(self.images)

    @staticmethod
    def collate_fn(batch):
        images, targets = list(zip(*batch))
        batched_imgs = cat_list(images, fill_value=0)
        batched_targets = cat_list(targets, fill_value=255)
        return batched_imgs, batched_targets


def cat_list(images, fill_value=0):
    # 计算该batch数据中，channel, h, w的最大值
    max_size = tuple(max(s) for s in zip(*[img.shape for img in images]))
    # print("max_size",max_size)
    batch_shape = (len(images),) + max_size
    # print("batch_shape",batch_shape)
    batched_imgs = images[0].new(*batch_shape).fill_(fill_value)
    # print("batched_imgs",batched_imgs)
    for img, pad_img in zip(images, batched_imgs):
        pad_img[..., :img.shape[-2], :img.shape[-1]].copy_(img)
    return batched_imgs


class PLAssigner:
    def __init__(self, h, w, point_sigma, line_sigma, line_pad):
        super(PLAssigner, self).__init__()
        self.h = h
        self.w = w
        self.point_sigma = point_sigma
        self.line_sigma = line_sigma
        self.line_pad = line_pad

    def assign(self, anno):
        point_list = anno['points']
        line_list = anno['lines']

        gt_maps = np.zeros((2, self.h, self.w), dtype=np.float32) # ch h w
        for ct in point_list :
           gt_maps[0] = self.draw_msra_gaussian(gt_maps[0], ct, self.point_sigma)
       
        for ln in line_list:
            gt_maps[1] = self.draw_guass_line(gt_maps[1], ln, self.line_sigma, self.line_pad)
        # print(gt_maps[0].shape)
        # point_save = gt_maps[0] * 255
        # line_save = gt_maps[1] * 255
        # cv2.imwrite("/tmpnfs/yaoming.zhang/landmark_pytorch/ldmk_data/j3_test/194_out/torch_test_31/gt_point_small.jpg", point_save)
        # cv2.imwrite("/tmpnfs/yaoming.zhang/landmark_pytorch/ldmk_data/j3_test/194_out/torch_test_31/gt_line_small.jpg", line_save)
        return gt_maps

    def draw_msra_gaussian(self, heatmap, center, sigma):
        tmp_size = sigma * 3
        mu_x = int(center[0] + 0.5)
        mu_y = int(center[1] + 0.5)
        w, h = heatmap.shape[0], heatmap.shape[1]
        ul = [int(mu_x - tmp_size), int(mu_y - tmp_size)]
        br = [int(mu_x + tmp_size + 1), int(mu_y + tmp_size + 1)]
        if ul[0] >= h or ul[1] >= w or br[0] < 0 or br[1] < 0:
            return heatmap
        size = 2 * tmp_size + 1
        x = np.arange(0, size, 1, np.float32)
        y = x[:, np.newaxis]
        x0 = y0 = size // 2
        g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))
        g_x = max(0, -ul[0]), min(br[0], h) - ul[0]
        g_y = max(0, -ul[1]), min(br[1], w) - ul[1]
        img_x = max(0, ul[0]), min(br[0], h)
        img_y = max(0, ul[1]), min(br[1], w)
        heatmap[img_y[0]:img_y[1], img_x[0]:img_x[1]] = np.maximum(
            heatmap[img_y[0]:img_y[1], img_x[0]:img_x[1]],
            g[g_y[0]:g_y[1], g_x[0]:g_x[1]])
        return heatmap

    def draw_guass_line(self, linemap, line, line_sigma, line_pad):
        line_pad = int(line_sigma * 3 + 0.5)
        st_x = int(line[0] + 0.5)
        st_y = int(line[1] + 0.5)
        ed_x = int(line[2] + 0.5)
        ed_y = int(line[3] + 0.5)
        vec = [float(ed_x - st_x), float(ed_y - st_y)]
        norm = vec[0] ** 2 + vec[1] ** 2
        norm = norm ** 0.5
        if ( norm < 1.0):
            return linemap
        vec_norm = vec
        vec_norm[0] = vec[0] / norm
        vec_norm[1] = vec[1] / norm
        h, w = linemap.shape[0], linemap.shape[1]
        min_x = max(0, min(st_x, ed_x) - line_pad)
        max_x = min(w, max(st_x, ed_x) + line_pad)
        min_y = max(0, min(st_y, ed_y) - line_pad)
        max_y = min(h, max(st_y, ed_y) + line_pad)
        for y in range(min_y, max_y):
            for x in range(min_x, max_x):
                vec_cur = [x - st_x, y - st_y]
                vert_dist = abs(vec_cur[0] * vec_norm[1] - vec_cur[1] * vec_norm[0])
                para_proj = vec_cur[0] * vec_norm[0] + vec_cur[1] * vec_norm[1]
                if (para_proj >= 0 and vert_dist <= line_pad):
                    gauss = np.exp(-(vert_dist * vert_dist) / (2 * line_sigma * line_sigma)) 
                    linemap[y][x] = max(linemap[y][x], gauss)
        return linemap

class SlotDataset(data.Dataset):
    def __init__(self, subset_list, w, h, point_sigma, line_sigma, line_pad, transforms=None, txt_name: str = "train.txt"):
        super(SlotDataset, self).__init__()
        for subset_name in subset_list:
            assert os.path.exists(subset_name), "%s isn't existed." % subset_name
        self.subset_list = subset_list
        self.w, self.h = w, h
        self.assigner = PLAssigner(h, w, point_sigma, line_sigma, line_pad)
        self.img_paths, self.slot_ann_paths =[], []

        for i in range(len(self.subset_list)):
            train_txt_path = os.path.join(self.subset_list[i], txt_name)
            lines = open(train_txt_path, 'r')
            for line in lines:
                line = str(line.rstrip('\n'))
                self.img_paths.append(self.subset_list[i] + '/img/' + line + '.jpg')
                self.slot_ann_paths.append(self.subset_list[i] + '/Annotations/' + line + '.xml')
        self.transforms = transforms

    def __getitem__(self, idx):
        """
        Args:
            idx (int): Index

        Returns:
            tuple: (image, target) where target is the image segmentation.
        """
        assert os.path.exists(self.img_paths[idx]), "%s isn't existed." % self.img_paths[idx]
        # read image
        image = cv2.imread(self.img_paths[idx])
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # read landmark gt mask 
        # lm_mask = cv2.imread(self.lm_mask_paths[idx], 0)

        # read slot annotation
        slot_anno = self.pull_anno(idx)
        slot_maps = self.assigner.assign(slot_anno)
        anno = slot_maps
        slot_maps = torch.from_numpy(slot_maps.astype(np.float32))

        if self.transforms is not None:
            # image, anno= self.transforms(image, anno)
            image= self.transforms(image)
        
        return image, slot_maps

    def __len__(self):
        return len(self.img_paths)



    def pull_anno(self, idx):
        xml_path = self.slot_ann_paths[idx]
        assert os.path.exists(xml_path), "%s isn't existed." % xml_path
        anno = self.xml_load(xml_path)
        return anno

    def xml_load(self, xml_path):
        anno = {}
        tree = ET.parse(xml_path)
        root = tree.getroot()
        size_node = root.find('size')
        rawh = float(size_node.find('height').text)
        raww = float(size_node.find('width').text)
        coex = self.w * 1.0 / raww
        coey = self.h * 1.0 / rawh
        # coex = 1.0
        # coey = 1.0
        anno['w'] = int(raww)
        anno['h'] = int(rawh)
        point_list = []
        line_list = []
        for obj_item in root.findall('object'):
            name_node = obj_item.find('name')
            obj_name = name_node.text
            bbox_node = obj_item.find('bndbox')
            if obj_name == 'keypoint':
                x = float(bbox_node.find('xmin').text) * coex
                y = float(bbox_node.find('ymin').text) * coey
                point_list.append([x,y])
            if obj_name == 'vecproj':
                x1 = float(bbox_node.find('xmin').text) * coex
                y1 = float(bbox_node.find('ymin').text) * coey
                x2 = float(bbox_node.find('xmax').text) * coex
                y2 = float(bbox_node.find('ymax').text) * coey
                line_list.append([x1,y1,x2,y2])
        anno['points'] = point_list
        anno['lines'] = line_list
        return anno

class SlotAllDataset(data.Dataset):
    def __init__(self, subset_list, train_root, w, h, point_sigma, line_sigma, line_pad, transforms=None, img_transforms=None,  old_data=None):
        super(SlotAllDataset, self).__init__()
        for subset_name in subset_list:
            assert os.path.exists(subset_name), "%s isn't existed." % subset_name
        self.subset_list = subset_list
        # print(self.subset_list)
        self.img_root = train_root
        self.w, self.h = w, h
        self.point_sigma = point_sigma
        self.line_sigma = line_sigma
        self.line_pad = line_pad
        # self.assigner = PLAssigner(h, w, point_sigma, line_sigma, line_pad)
        self.assigner = PLAssigner(self.h, self.w, self.point_sigma, self.line_sigma, self.line_pad)
        self.img_paths, self.slot_ann_paths =[], []

        for i in range(len(self.subset_list)):
            train_txt_path = self.subset_list[i]
            lines = open(train_txt_path, 'r')
            for line in lines:
                line_str = str(line.rstrip('\n')) 
                txt_path = os.path.join(train_root, line_str)
                json_path_split = txt_path.split("/")
                json_folder_part = json_path_split[-3:]
                json_folders = os.path.join(json_folder_part[0], json_folder_part[1], json_folder_part[2])
                file_name = json_folder_part[2].split(".")[0]
                img_path = os.path.join(txt_path.replace(".json", ".jpg").replace("label", "avm"))
                if os.path.exists(img_path) and os.path.exists(txt_path):
                    self.img_paths.append(img_path)
                    self.slot_ann_paths.append(txt_path)

        self.transforms = transforms
        self.img_transforms = img_transforms

    def __getitem__(self, idx):
        """
        Args:
            idx (int): Index

        Returns:
            tuple: (image, target) where target is the image segmentation.
        """
        # print(self.img_paths[idx])
        assert os.path.exists(self.img_paths[idx]), "%s isn't existed." % self.img_paths[idx]   
        # print("cur img path", self.img_paths[idx])
        image = self.pull_img(idx)
        slot_anno = self.pull_anno(idx)
        slot_maps = self.assigner.assign(slot_anno)
        anno = slot_maps
        
        # slot_maps = torch.from_numpy(slot_maps.astype(np.float32))
        slot_gt = np.zeros((2, self.h, self.w), dtype=np.float32) # ch h w
        if self.transforms is not None:
            # image = self.img_transforms(image)

            trans_1 = self.transforms(image=image, mask1=anno[0], mask2=anno[1])
            image_trans = trans_1['image']
            point_gt = trans_1['mask1']
            line_gt = trans_1['mask2']
            
            slot_gt[0] = point_gt
            slot_gt[1] = line_gt
            image_gt = self.img_transforms(image_trans)
        else:
            slot_gt[0] = anno[0]
            slot_gt[1] = anno[1]
            image_gt = self.img_transforms(image)
            
            
        slot_maps = torch.from_numpy(slot_gt.astype(np.float32))
        return image_gt, slot_maps

    def __len__(self):
        return len(self.img_paths)


    def pull_img(self, idx):
        image = cv2.imread(self.img_paths[idx])
        if image is None:
            print("!!!!image is None = ", self.img_paths[idx])
        image = cv2.resize(image, (self.w, self.h))
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image

    def pull_anno(self, idx):
        json_path = self.slot_ann_paths[idx]
        assert os.path.exists(json_path), "%s isn't existed." % json_path
        anno = self.json_load(json_path)
        return anno
    
    def json_load(self, json_path):
        anno = {}
        annos = json.load(open(json_path))
        objs = annos["annotation"]["object"]
        raw_img_w = annos["annotation"]["imgsize"]["width"]
        raw_img_h = annos["annotation"]["imgsize"]["height"]
        anno['w'] = int(float(raw_img_w))
        anno['h'] = int(float(raw_img_h))
        coex = self.w * 1.0 / anno['w'] 
        coey = self.h * 1.0 / anno['h']
        slotpt_list = []
        slotline_list = []
 
        for obj in objs:
            if obj["name"] == "keypoint":
                slot_pt = obj["pt"][0]
                x = float(slot_pt['x']) * coex
                y = float(slot_pt['y']) * coey
                slotpt_list.append([x, y])
            if obj["name"] == "line":
                slot_line = obj["pt"]
                pt0 = slot_line[0]
                pt1 = slot_line[1]
                x0 = float(pt0['x']) * coex
                y0 = float(pt0['y']) * coey

                x1 = float(pt1['x']) * coex
                y1 = float(pt1['y']) * coey
                slotline_list.append([x0, y0, x1, y1])
        anno['points'] = slotpt_list
        anno['lines'] = slotline_list

        return anno

    def xml_load(self, xml_path):
        anno = {}
        tree = ET.parse(xml_path)
        root = tree.getroot()
        size_node = root.find('size')
        rawh = float(size_node.find('height').text)
        raww = float(size_node.find('width').text)
        # coex = self.w * 1.0 / raww
        # coey = self.h * 1.0 / rawh
        coex = 1
        coey = 1
        anno['w'] = int(raww)
        anno['h'] = int(rawh)
        point_list = []
        line_list = []
        for obj_item in root.findall('object'):
            name_node = obj_item.find('name')
            obj_name = name_node.text
            bbox_node = obj_item.find('bndbox')
            if obj_name == 'keypoint':
                x = float(bbox_node.find('xmin').text) * coex
                y = float(bbox_node.find('ymin').text) * coey
                point_list.append([x,y])
            if obj_name == 'vecproj':
                x1 = float(bbox_node.find('xmin').text) * coex
                y1 = float(bbox_node.find('ymin').text) * coey
                x2 = float(bbox_node.find('xmax').text) * coex
                y2 = float(bbox_node.find('ymax').text) * coey
                line_list.append([x1,y1,x2,y2])
        anno['points'] = point_list
        anno['lines'] = line_list
        return anno
    
    # def debug_show(self, idx):
    #     image = self.pull_img(idx)
    #     image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    #     anno = self.pull_anno(idx)
    #     gtmap = self.assigner.assign(anno)
    #     print ('image  shape : ', image.shape)
    #     cv2.imshow('src image', image)

    #     #print ('heatmap shape : ', heatmap.shape)
    #     cv2.imshow('heatmap', gtmap[0])
    #     cv2.imshow('linemap', gtmap[1])
    #     cv2.waitKey()
    

    # def save_gt(self, savePath):
    #     for idx in range(self.__len__()):
    #         if idx > 100:
    #             break
    #         image = self.pull_img(idx)
    #         image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    #         anno = self.pull_anno(idx)
    #         gtmap = self.assigner.assign(anno)
    #         kmap = np.zeros((self.h, self.w, 1), np.uint8)
    #         lmap = np.zeros((self.h, self.w, 1), np.uint8)
    #         for j in range(self.h):
    #             for i in range(self.w):
    #                 kmap[j][i] = int(gtmap[0][j][i] * 255)
    #                 lmap[j][i] = int(gtmap[1][j][i] * 255)
    #                 # print("kmap value ", kmap[j][i])

    #         cv2.imwrite(savePath + '/' + str(idx) + '_img.jpg', image)
    #         cv2.imwrite(savePath + '/' + str(idx) + '_point.jpg', kmap)
    #         cv2.imwrite(savePath + '/' + str(idx) + '_vector.jpg', lmap)
        
    def save_heatmap(self, idx, savePath):
        img_path = self.img_paths[idx]
        img_name = os.path.basename(img_path)
        img_name_str = img_name.split('.')[0]
        image = self.pull_img(idx)
        # image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        anno = self.pull_anno(idx)
        gtmap = self.assigner.assign(anno)
        res_img = overlay_heatmap(image, gtmap[0], point_radius=3)
        line_img = overlay_heatmap(image, gtmap[1], point_radius=3)
        cv2.imwrite(savePath + '/' + img_name_str + '_pt.jpg', res_img)
        cv2.imwrite(savePath + '/' + img_name_str + '_line.jpg', line_img)
    
    def save_all_heatmap(self, savePath):
        if not os.path.exists(savePath):
            os.makedirs(savePath)
        for idx in range(self.__len__()):
            if idx > 100:
                break
            image = self.pull_img(idx)
            # image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            anno = self.pull_anno(idx)
            gtmap = self.assigner.assign(anno)

            # kmap = np.zeros((self.h, self.w, 1), np.uint8)
            # lmap = np.zeros((self.h, self.w, 1), np.uint8)
            # for j in range(self.h):
            #     for i in range(self.w):
            #         kmap[j][i] = int(gtmap[0][j][i] * 255)
            #         lmap[j][i] = int(gtmap[1][j][i] * 255)
                    # print("kmap value ", kmap[j][i])

            res_img = overlay_heatmap(image, gtmap[0], point_radius=3)
            line_img = overlay_heatmap(image, gtmap[1], point_radius=3)
            cv2.imwrite(savePath + '/' + str(idx) + '_pt.jpg', res_img)
            cv2.imwrite(savePath + '/' + str(idx) + '_line.jpg', line_img)
    
    def check_set_valid(self):
        total_num = len(self.img_paths)
        cnt = 0
        for idx in range(total_num):
            cnt = cnt + 1
            if ( cnt%1000 == 1):
                print ('proc :', (cnt//1000)*1000)
            img_path = self.img_paths[idx]
            json_path = self.slot_ann_paths[idx]
            img_valid, xml_valid = False, False
            if (os.path.exists(img_path)):
                img_valid = True
            else:
                print (img_path, ' not exist !')
            if (os.path.exists(json_path)):
                xml_valid = True
            else:
                print (json_path, ' not exist !')
            if (not(img_valid and xml_valid)):
                continue
            image = cv2.imread(img_path)
            anno = self.json_load(json_path)
            if (image.shape[0] != anno['h'] or image.shape[1] != anno['w']):
                print (img_path, ' size not match ! - ', image.shape, anno['h'], anno['w'])
            if (not self.check_ann_valid(anno)):
                print (json_path, ' xml not valid !')
        print ('check image & xml down ')
            
    def check_ann_valid(self, anno):
        point_list = anno['points']
        line_list = anno['lines']
        ret = True
        for line in line_list:
            st_x = int(line[0] + 0.5)
            st_y = int(line[1] + 0.5)
            ed_x = int(line[2] + 0.5)
            ed_y = int(line[3] + 0.5)
            vec = [float(ed_x - st_x), float(ed_y - st_y)]
            norm = vec[0] ** 2 + vec[1] ** 2
            norm = norm ** 0.5
            if ( norm < 1.0):
                return False
        return ret

class SlotAllDataset_val(data.Dataset):
    def __init__(self, subset_list, img_root, w, h, point_sigma, line_sigma, line_pad, transforms=None, img_transforms=None, txt_name: str = "train.txt", old_data=None):
        super(SlotAllDataset_val, self).__init__()
        for subset_name in subset_list:
            assert os.path.exists(subset_name), "%s isn't existed." % subset_name
        self.subset_list = subset_list
        self.img_root = img_root
        self.w, self.h = w, h
        self.point_sigma = point_sigma
        self.line_sigma = line_sigma
        self.line_pad = line_pad
        # self.assigner = PLAssigner(h, w, point_sigma, line_sigma, line_pad)
        self.img_paths, self.slot_ann_paths =[], []


        for i in range(len(self.subset_list)):
            train_txt_path = os.path.join(self.subset_list[i], txt_name)
            lines = open(train_txt_path, 'r')
            for line in lines:
                line = str(line.rstrip('\n'))
                json_path_split = line.split("/")
                json_folder_part = json_path_split[-3:]
                json_folders = os.path.join(json_folder_part[0], json_folder_part[1], json_folder_part[2])
                file_name = json_folder_part[2].split(".")[0]
                img_path = os.path.join(self.img_root, json_folders.replace(".json", ".jpg"))

                self.img_paths.append(img_path)
                self.slot_ann_paths.append(line)
        
                
        self.transforms = transforms
        self.img_transforms = img_transforms

    def __getitem__(self, idx):
        """
        Args:
            idx (int): Index

        Returns:
            tuple: (image, target) where target is the image segmentation.
        """
        assert os.path.exists(self.img_paths[idx]), "%s isn't existed." % self.img_paths[idx]
        # read image
        image = cv2.imread(self.img_paths[idx])
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # image = cv2.resize(image, (self.w, self.h))
        # print("img path", self.img_paths[idx])
        # read landmark gt mask 
        # lm_mask = cv2.imread(self.lm_mask_paths[idx], 0)
        
        # read slot annotation
        slot_anno = self.pull_anno(idx)
        assigner = PLAssigner(slot_anno['h'], slot_anno['w'], self.point_sigma, self.line_sigma, self.line_pad)
        slot_maps = assigner.assign(slot_anno)
        anno = slot_maps
        slot_maps = torch.from_numpy(slot_maps.astype(np.float32))
        image = self.img_transforms(image)
        
        # if self.transforms is not None:
        #     # image = self.img_transforms(image)

        #     trans_1 = self.transforms(image=image, mask1=anno[0], mask2=anno[1])
        #     image_trans = trans_1['image']
        #     point_gt = trans_1['mask1']
        #     line_gt = trans_1['mask2']
        #     # print("point shape", point_gt.shape)
        #     # print("line shape", line_gt.shape)
        #     # print("self h", self.h)
        #     # print("self w", self.w)
        #     # ch h w
        #     slot_gt = np.zeros((2, self.h, self.w), dtype=np.float32) 
        #     slot_gt[0] = point_gt
        #     slot_gt[1] = line_gt
        #     image_gt = self.img_transforms(image_trans)
        # slot_maps = torch.from_numpy(slot_gt.astype(np.float32))
        return image, slot_maps

    def __len__(self):
        return len(self.img_paths)
    
    def pull_anno(self, idx):
        json_path = self.slot_ann_paths[idx]
        assert os.path.exists(json_path), "%s isn't existed." % json_path
        anno = self.json_load(json_path)
        return anno
    
    def json_load(self, json_path):
        anno = {}
        annos = json.load(open(json_path))
        objs = annos["annotation"]["object"]
        raw_img_w = annos["annotation"]["imgsize"]["width"]
        raw_img_h = annos["annotation"]["imgsize"]["height"]
        anno['w'] = int(float(raw_img_w))
        anno['h'] = int(float(raw_img_h))

        slotpt_list = []
        slotline_list = []
 
        for obj in objs:
            if obj["name"] == "keypoint":
                slot_pt = obj["pt"][0]
                x = float(slot_pt['x'])
                y = float(slot_pt['y'])
                slotpt_list.append([x, y])
            if obj["name"] == "line":
                slot_line = obj["pt"]
                pt0 = slot_line[0]
                pt1 = slot_line[1]
                x0 = float(pt0['x'])
                y0 = float(pt0['y'])

                x1 = float(pt1['x'])
                y1 = float(pt1['y'])
                slotline_list.append([x0, y0, x1, y1])
        anno['points'] = slotpt_list
        anno['lines'] = slotline_list

        return anno

    def xml_load(self, xml_path):
        anno = {}
        tree = ET.parse(xml_path)
        root = tree.getroot()
        size_node = root.find('size')
        rawh = float(size_node.find('height').text)
        raww = float(size_node.find('width').text)
        # coex = self.w * 1.0 / raww
        # coey = self.h * 1.0 / rawh
        coex = 1
        coey = 1
        anno['w'] = int(raww)
        anno['h'] = int(rawh)
        point_list = []
        line_list = []
        for obj_item in root.findall('object'):
            name_node = obj_item.find('name')
            obj_name = name_node.text
            bbox_node = obj_item.find('bndbox')
            if obj_name == 'keypoint':
                x = float(bbox_node.find('xmin').text) * coex
                y = float(bbox_node.find('ymin').text) * coey
                point_list.append([x,y])
            if obj_name == 'vecproj':
                x1 = float(bbox_node.find('xmin').text) * coex
                y1 = float(bbox_node.find('ymin').text) * coey
                x2 = float(bbox_node.find('xmax').text) * coex
                y2 = float(bbox_node.find('ymax').text) * coey
                line_list.append([x1,y1,x2,y2])
        anno['points'] = point_list
        anno['lines'] = line_list
        return anno

if __name__ == '__main__':
    # import albumentations as A
    

    train_root = "/data/ai_group/datasets/bev_park/train_test_dataset/PointLineData_Train"
    json_dir = "/data/ai_group/datasets/bev_park/train_test_dataset/PointLineData_Train/train_all.txt"
    # transforms = A.Compose([
    #     # A.PadIfNeeded(
    #     #     min_height=896, min_width=896, border_mode=cv2.BORDER_CONSTANT, value=0, p=1
    #     # ),
    #     # A.Resize(height=896, width=896, p=1.0),
    #     # A.RandomSizedCrop(min_max_height=(300, 512), 
    #     #             height=512, width=512, p=0.5),
    #     A.HorizontalFlip(p=0.5),
    #     A.VerticalFlip(p=0.5),
    #     A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=90, p=0.2)
    # ], additional_targets={'image':'image', 'mask1':'mask','mask2':'mask'})
    
    transforms_slot = T.Compose_img([
           #T.RGBToBGR(),
            T.ToTensor_img(),
            # T.Resize(896, 896),
            # T.Normalize(mean=(122.67892/255, 116.66877/255, 104.00699/255), std=(1, 1, 1)),
        ])
    w = 768
    h = 768
    subset_list = [json_dir]
    slotData = SlotAllDataset(subset_list, train_root, w, h, 3, 1.6, 0, transforms=None, img_transforms=transforms_slot) 
    lenth = slotData.__len__()
    # slotData.check_set_valid()
    savePath = "/data/ai_group/datasets/bev_park/train_test_dataset/train_data_visual"
    if not os.path.exists(savePath):
        os.makedirs(savePath)
    for i in range(lenth):
        label = slotData.__getitem__(i)
        slotData.save_heatmap(i, savePath)
    
    # slotData.save_gt(savePath)
    