import numpy as np
import sys
import os
import cv2
import math
import json


class TXTLabelLoader(object):
    def __init__(self, sw, sh, img_path=None):
        self.param = self.initParam(sw, sh)
        self.img_file = img_path

    def initParam(self, sw, sh):
        param = {}
        # param['point_dis_thr'] = 4
        # param['point_dis_thr'] = 15
        param['point_dis_thr'] = 5
        param['line_angle_thr'] = float(0.96619)  # cos(angle_5)
        param['scale_w'] = sw
        param['scale_h'] = sh
        return param

    def doHeatmapStatistics(self, txt_path, detections):
        annotations = self.decodePointLineLabel(txt_path)
        resultPack = self.errorCaculate(detections, annotations)
        return resultPack

    def dodiffDisHeatmapStatistics(self, txt_path, detections, dis=None):
        annotations = self.decodediffDisPointLineLabel(txt_path)
        resultPack = self.errorCaculate(detections, annotations)
        return resultPack

    def doSlotMatchStatistics(self, txt_path, detections):
        annotations = self.decodeSlotLabel(txt_path)
        resultPack = self.perfCaculate(detections, annotations)
        return resultPack

    def decodeSlotLabel(self, txt_path):
        # [p0x, p0y, p1x, p1y, p2x, p2y, p3x, p3y]
        annotations = []
        sw = self.param['scale_w']
        sh = self.param['scale_h']
        txtFile = open(txt_path, 'r')
        for line in txtFile:
            if 'imgwh' in line:
                continue
            line = str(line.rstrip('\n'))
            splitLine = line.split(' ')
            if (len(splitLine) < 8):
                continue
            gt = splitLine[0:8]
            for i in range(len(gt)/2):
                gt[2*i] = int(int(gt[2*i]) * sw)
                gt[2*i+1] = int(int(gt[2*i+1]) * sh)
            annotations.append(gt)
        return annotations

    def img2baselink(self, master_point):
        car_info = {}
        car_info['start_x'] = 406
        car_info['start_y'] = 344
        car_info['width'] = 82
        car_info['pixel_len'] = 2.23047

        baselink_point = np.array([
            (master_point[0] - (car_info['start_x'] +
             car_info['width'] * 0.5)) * car_info['pixel_len'],
            (car_info['start_y'] - master_point[1]) * car_info['pixel_len'],
        ])
        return baselink_point

    def decodePointLineLabel(self, txt_path):
        # load : [px, py, ed1x, ed1y , ...] ...

        annotations = []
        sw = self.param['scale_w']
        sh = self.param['scale_h']
        txtFile = open(txt_path, 'r')
        for line in txtFile:
            if 'imgwh' in line:
                continue
            line = str(line.rstrip('\n').rstrip())
            splitLine = line.split(' ')
            point = [int(splitLine[0]), int(splitLine[1])]
            linepoints = splitLine[2:]

            orients = []
            for i in range(int(len(linepoints)/2)):
                endP = [int(linepoints[2*i]), int(linepoints[2*i+1])]
                dx = endP[0] - point[0]
                dy = endP[1] - point[1]
                norm = math.sqrt(dx*dx + dy*dy)
                if norm == 0:
                    continue
                sinr = dx / norm
                cosr = dy / norm
                orients.append([sinr, cosr])
            # trans = transforms('image':img, 'keypoint':point)
            point = [int(point[0] * sw), int(point[1] * sh)]
            # baselink_point = self.img2baselink(point)
            annotations.append([point, orients])
        return annotations

    def decodePointLabel_json_anno(self, json_path):
        sw = self.param['scale_w']
        sh = self.param['scale_h']
        anno = {}
        annos = json.load(open(json_path))
        objs = annos["annotation"]["object"]
        raw_img_w = annos["annotation"]["imgsize"]["width"]
        raw_img_h = annos["annotation"]["imgsize"]["height"]
        anno['w'] = int(float(raw_img_w))
        anno['h'] = int(float(raw_img_h))
        coex = sw
        coey = sh
        annotations = []

        for obj in objs:
            if obj["name"] == "keypoint":
                slot_pt = obj["pt"][0]
                x = float(slot_pt['x']) * coex
                y = float(slot_pt['y']) * coey
                point = [int(x), int(y)]

            annotations.append([point])

        return annotations

    def decodediffDisPointLineLabel(self, txt_path):
        # load : [px, py, ed1x, ed1y , ...] ...

        annotations_300 = []
        annotations_500 = []
        annotations_800 = []
        annotations_1000 = []
        sw = self.param['scale_w']
        sh = self.param['scale_h']
        txtFile = open(txt_path, 'r')
        for line in txtFile:
            if 'imgwh' in line:
                continue
            line = str(line.rstrip('\n'))
            splitLine = line.split(' ')
            point = [int(splitLine[0]), int(splitLine[1])]
            linepoints = splitLine[2:]
            # print("sssssssssssssss", point,linepoints)
            orients = []
            for i in range(int(len(linepoints)/2)):
                endP = [int(linepoints[2*i]), int(linepoints[2*i+1])]
                # print("wwsssssssssssssss", point,endP)
                dx = endP[0] - point[0]
                dy = endP[1] - point[1]
                norm = math.sqrt(dx*dx + dy*dy)
                if norm == 0:
                    continue
                sinr = dx / norm
                cosr = dy / norm
                orients.append([sinr, cosr])
            # trans = transforms('image':img, 'keypoint':point)
            point = [int(point[0] * sw), int(point[1] * sh)]
            baselink_point = self.img2baselink(point)
            if baselink_point[0] <= 300 and baselink_point[0] >= -300 and baselink_point[1] <= 300 and baselink_point[1] >= -300:
                annotations_300.append([point, orients])
            elif baselink_point[0] <= 500 and baselink_point[0] >= -500 and baselink_point[1] <= 500 and baselink_point[1] >= -500:
                annotations_500.append([point, orients])
            elif baselink_point[0] <= 800 and baselink_point[0] >= -800 and baselink_point[1] <= 800 and baselink_point[1] >= -800:
                annotations_800.append([point, orients])
            else:
                annotations_1000.append([point, orients])
        return annotations_300, annotations_500, annotations_800, annotations_1000

    def perfCaculate(self, detections, annotations):
        # dections : list of slot
        #    slot : ['up_point'] = [x,y,s]
        #    slot : ['dn_point'] = [x,y,s]
        #    slot : ['up_line'] = [sinr,cosr,hist,len]
        #    slot : ['dn_line'] = [sinr,cosr,hist,len]
        # annotations: list of [p0x, p0y, p1x, p1y, p2x, p2y, p3x, p3y] ...
        # set match flag to every object:
        for det in detections:
            det['flag'] = -1
        for ann in annotations:
            ann.extend([-1])

        slot_total_num = 0
        slot_true_num = 0
        slot_miss_num = 0
        slot_false_num = 0
        point_thr = self.param['point_dis_thr']

        for det in detections:
            for ann in annotations:
                if (self.isSlotMatch() == True):
                    det['flag'] = 1
                    ann[8] = 1

    def isSlotMatch(self, det, ann, thr):
        pup = det['up_point']
        pdn = det['dn_point']
        p0 = [ann[0], ann[1]]
        p3 = [ann[6], ann[7]]
        dx = max(abs(pup[0]-p0[0]), abs(pdn[0]-pdn[0]))
        dy = max(abs(pup[1]-p0[1]), abs(pdn[1]-pdn[1]))
        max_pixel_dis = max(abs(dx), abs(dy))
        if (max_pixel_dis < thr):
            return True

    def errorCaculate(self, detections, annotations):
        # detections : list [point[x,y,s] orients_list[[sin,cos,hist_score,act_len],...], ...]
        # annotations : list [point[x,y], orients_list[[sin,cos], ...]]
        # set match flag to every object:
        det_pt = 0
        det_line = 0
        for det in detections:
            det_pt = det_pt + 1
            det[0].extend([-1])
            for lin in det[1]:
                lin.extend([-1])
                det_line = det_line + 1
        ann_pt = 0
        ann_line = 0
        for ann in annotations:
            ann[0].extend([-1])
            ann_pt = ann_pt + 1
            for lin in ann[1]:
                lin.extend([-1])
                ann_line = ann_line + 1
        
        # detections : list [point[x,y,s,f] orients_list[[sin,cos,hist_score,act_len,f],...], ...]
        # annotations : list [point[x,y,f], orients_list[[sin,cos,f], ...]]
        ann_size = len(annotations)
        det_size = len(detections)

        point_error = 0.0
        angle_error = 0.0
        point_total_num = 0
        point_true_num = 0
        point_miss_num = 0
        point_false_num = 0
        line_total_num = 0
        line_true_num = 0
        line_miss_num = 0
        line_false_num = 0

        point_thr = self.param['point_dis_thr']
        angle_thr = self.param['line_angle_thr']
        # img = cv2.imread(self.img_file)
        # img = cv2.resize(img, (896, 896))
        for det in detections:
            # proc on point & lines
            # cv2.circle(img, (det[0][0], det[0][1]), 2, (0,0,250), -1)
            for ann in annotations:
                # cv2.circle(img, (ann[0][0], ann[0][1]), 2, (255,0,0), -1)
                point_error, isMatched = self.pointMatch(
                    det, ann, point_thr, point_error)
                if (isMatched):
                    angle_error = self.lineListMatch(
                        det, ann, angle_thr, angle_error)
            # cv2.imwrite("/tmpnfs/yaoming.zhang/landmark_pytorch/ldmk_data/j3_test/test_ann.jpg", img)
        point_total_num, point_true_num, point_miss_num, point_false_num, point_det_num = self.countPointPerf(
            detections, annotations)
        line_total_num, line_true_num, line_miss_num, line_false_num, line_det_num = self.countLinePerf(
            detections, annotations)

        resultPack = {}
        resultPack['point_error'] = point_error
        resultPack['angle_error'] = angle_error
        resultPack['point_total_num'] = point_total_num
        resultPack['point_true_num'] = point_true_num
        resultPack['point_miss_num'] = point_miss_num
        resultPack['point_false_num'] = point_false_num
        resultPack['line_total_num'] = line_total_num
        resultPack['line_true_num'] = line_true_num
        resultPack['line_miss_num'] = line_miss_num
        resultPack['line_false_num'] = line_false_num
        resultPack['point_det_num'] = point_det_num
        resultPack['line_det_num'] = line_det_num

        return resultPack

    def errorCaculate_point(self, detections, annotations):
        # detections : list [point[x,y,s] orients_list[[sin,cos,hist_score,act_len],...], ...]
        # annotations : list [point[x,y], orients_list[[sin,cos], ...]]
        # set match flag to every object:
        for det in detections:
            det.extend([-1])

        for ann in annotations:
            ann[0].extend([-1])

        # detections : list [point[x,y,s,f] orients_list[[sin,cos,hist_score,act_len,f],...], ...]
        # annotations : list [point[x,y,f], orients_list[[sin,cos,f], ...]]
        ann_size = len(annotations)
        det_size = len(detections)

        point_error = 0.0
        angle_error = 0.0
        point_total_num = 0
        point_true_num = 0
        point_miss_num = 0
        point_false_num = 0
        line_total_num = 0
        line_true_num = 0
        line_miss_num = 0
        line_false_num = 0

        point_thr = self.param['point_dis_thr']
        angle_thr = self.param['line_angle_thr']
        # img = cv2.imread(self.img_file)
        # img = cv2.resize(img, (896, 896))
        for det in detections:
            # proc on point & lines
            # cv2.circle(img, (det[0][0], det[0][1]), 2, (0,0,250), -1)
            for ann in annotations:
                # cv2.circle(img, (ann[0][0], ann[0][1]), 2, (255,0,0), -1)
                point_error, isMatched = self.pointMatch(
                    det, ann, point_thr, point_error)

            # cv2.imwrite("/tmpnfs/yaoming.zhang/landmark_pytorch/ldmk_data/j3_test/test_ann.jpg", img)
        point_total_num, point_true_num, point_miss_num, point_false_num = self.countPointPerf(
            detections, annotations)
        # line_total_num, line_true_num, line_miss_num, line_false_num = self.countLinePerf(detections, annotations)

        resultPack = {}
        resultPack['point_error'] = point_error
        resultPack['angle_error'] = angle_error
        resultPack['point_total_num'] = point_total_num
        resultPack['point_true_num'] = point_true_num
        resultPack['point_miss_num'] = point_miss_num
        resultPack['point_false_num'] = point_false_num
        resultPack['line_total_num'] = line_total_num
        resultPack['line_true_num'] = line_true_num
        resultPack['line_miss_num'] = line_miss_num
        resultPack['line_false_num'] = line_false_num
        return resultPack

    def pointMatch(self, det, ann, thr, err):
        # compare det[0] vs ann[0]
        # det_point[x,y,s,f] -- ann_point[x,y,f]
        flag = 0
        dx = det[0][0]-ann[0][0]
        dy = det[0][1]-ann[0][1]
        max_pixel_dis = max(abs(dx), abs(dy))
        if (max_pixel_dis < thr):
            det[0][3] = 1
            if ann[0][2] < 0:
                ann[0][2] = 1
            else:
                ann[0][2] += 1
            dd = math.sqrt(dx*dx + dy*dy)
            err = err + dd
            flag = 1
        return err, flag

    def lineListMatch(self, det, ann, thr, err):
        # line list : det[1] vs ann[1]
        for det_line in det[1]:
            for ann_line in ann[1]:
                err = self.lineMatch(det_line, ann_line, thr, err)
        return err

    def lineMatch(self, det_line, ann_line, thr, err):
        # compare det_line vs ann_line
        # det_line[sin,cos,hist_score,act_len,f] -- ann_line[sin,cos,f]
        match_cos = det_line[0] * ann_line[0] + det_line[1] * ann_line[1]
        if (match_cos > thr):
            det_line[4] = 1
            if ann_line[2] < 0:
                ann_line[2] = 1
            else:
                ann_line[2] += 1

            err = err + abs(np.arccos(match_cos))
            # err proc todo!
        return err

    def countPointPerf(self, detections, annotations):
        point_total_num = len(annotations)
        det_total_num = len(detections)
        point_true_num = 0
        point_miss_num = 0
        point_false_num = 0
        multi_det_point_num  = 0
        for ann in annotations:
            if (ann[0][2] >= 1):
                point_true_num = point_true_num + 1
                if (ann[0][2] > 1):
                    multi_det_point_num = multi_det_point_num + ann[0][2] - 1 

            if (ann[0][2] < 0):
                point_miss_num = point_miss_num + 1

        for det in detections:
            if (det[0][3] < 0):
                point_false_num = point_false_num + 1
        point_false_num += multi_det_point_num
        return point_total_num, point_true_num, point_miss_num, point_false_num, det_total_num

    def countLinePerf(self, detections, annotations):
        line_total_num = 0
        line_true_num = 0
        line_miss_num = 0
        line_false_num = 0
        multi_det_line_num = 0
        static_det_line_num = 0
        for ann in annotations:
            for ann_line in ann[1]:
                line_total_num = line_total_num + 1
                if (ann_line[2] >= 1):
                    line_true_num = line_true_num + 1
                    if (ann_line[2] > 1):
                        multi_det_line_num = multi_det_line_num + ann_line[2] - 1
                if (ann_line[2] < 0):
                    line_miss_num = line_miss_num + 1
               
        for det in detections:
            for det_line in det[1]:
                static_det_line_num = static_det_line_num + 1
                if (det_line[4] < 0):
                    line_false_num = line_false_num + 1
        line_false_num += multi_det_line_num
        line_det_num = static_det_line_num
        return line_total_num, line_true_num, line_miss_num, line_false_num, line_det_num
