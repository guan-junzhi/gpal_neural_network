import numpy as np  
import sys,os  
import cv2
import math
import logging
from scipy.ndimage import maximum_filter

class HeatMap(object):
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.param = self.initHeatMapParam()
        self.keyPoints = []
        self.vertexElements = []

    def initHeatMapParam(self):
        HeatMapParam = {}
        HeatMapParam['CheckParam'] = self.initCheckLogicParam()
        HeatMapParam['PointThrParam'] = self.initPointThrParam()
        HeatMapParam['LineThrParam'] = self.initLineThrParam()
        return HeatMapParam

    def initCheckLogicParam(self):
        CheckParam = {}
        CheckParam['solid_score'] = int(1)
        CheckParam['check_no_sep_ori_point'] = int(1)
        CheckParam['solid_half_score'] = int(6)
        CheckParam['cover_sin_thr'] = float(0.258)
        return CheckParam

    def initPointThrParam(self):
        PointThrParam = {}
        PointThrParam['point_conf_thr'] = float(0.3)
        # PointThrParam['point_conf_thr'] = float(0.4)
        PointThrParam['point_nms_min_dis'] = int(4)
        return PointThrParam

    def initLineThrParam(self):
        LineThrParam = {}
        LineThrParam['bins'] = 360
        # LineThrParam['len'] = 40
        LineThrParam['len'] = 80
        LineThrParam['cover_sin_thr'] = float(0.258)
        LineThrParam['cover_cos_thr'] = float(0.966)
        LineThrParam['active_value'] = float(0.274)
        LineThrParam['detect_value'] = float(0.1)
        LineThrParam['hist_min_score'] = float(7.5)
        LineThrParam['confirm_score_thr'] = float(6.0)
        LineThrParam['min_active_len'] = int(9)
        LineThrParam['search_radius'] = int(30)
        LineThrParam['vertex_radius'] = int(12)
        LineThrParam['match_cos_thr'] = float(0.96619)
        return LineThrParam

    def doProc(self, heatmap, vecmap):
        self.heatmap = heatmap
        self.vecmap = vecmap
        keyPoints = self.GetKeyPoint2()
        # print("keyPoints ", keyPoints)
        self.keyPoints = self.NMSKeyPoints(keyPoints)
        # print("nmskeyPoints ", self.keyPoints)
        self.vertexElements = self.GetAllLines(self.keyPoints)
        return self.vertexElements

    def drawVE(self, mask, savePath):
        h, w, c = mask.shape
        for i in range(len(self.vertexElements)):
            point = self.vertexElements[i][0]
            orients = self.vertexElements[i][1]
            for j in range(len(orients)):
                ori = orients[j]
                stp = (int(point[0]), int(point[1]))
                edp = (int(stp[0] + ori[0]*ori[3]), int(stp[1] + ori[1]*ori[3]))
                cv2.line(mask, stp, edp, (0, 250, 0), 2)
                # mdp = ((edp[0]+stp[0])/2 , (edp[1]+stp[1])/2)
                # line_label = 'p' + str(i) + '_l' + str(j)
                # cv2.putText(mask, line_label, mdp, 1, 0.8, (0,0,0), 1)
        for i in range(len(self.vertexElements)):
            point = self.vertexElements[i][0]
            cv2.circle(mask, (int(point[0]), int(point[1])), 2, (0,0,250), -1)
            dx = 5
            if point[0] > w/2:
                dx = -15
            mdp = (point[0] + dx, point[1] + 5)
            # point_label = 'p' + str(i)
            # cv2.putText(mask, point_label, mdp, 1, 0.8, (0,0,200), 1)
        cv2.imwrite(savePath, mask)


    def drawVEPoint(self, mask, savePath):
        h, w, c = mask.shape
        for i in range(len(self.keyPoints)):
            point = self.keyPoints[i]
   
            cv2.circle(mask, (point[0], point[1]), 2, (0,0,250), -1)
        cv2.imwrite(savePath, mask)

    def GetKeyPoint(self): #points are visited by top-down(by y=0 -> y=h)
        keypoints = []
        w = self.width
        h = self.height
        thr = self.param['PointThrParam']['point_conf_thr']
        for j in range (h):
            for i in range (w):
                curValue = self.heatmap[j,i]
                if curValue < thr :
                    continue
                peakPointFlag = 1
                for jj in range(3):
                    for ii in range(3):
                        idxj = max(0,min(j + 1 - jj, h - 1))
                        idxi = max(0,min(i + 1 - ii, w - 1))
                        if curValue < self.heatmap[idxj,idxi]:
                            peakPointFlag = 0
                if (peakPointFlag):
                    keypoints.append([i, j, curValue])
        return keypoints
    
    def GetKeyPoint2(self):
        keypoints = []
        w = self.width
        h = self.height
        thr = self.param['PointThrParam']['point_conf_thr']
        neighborhood_max = maximum_filter(self.heatmap, size=3, mode='constant', cval=0)
        
        # 步骤2：峰值判定（向量化，替代所有循环）
        # 条件1：值大于阈值；条件2：当前值等于邻域最大值（即局部峰值）
        peak_mask = (self.heatmap > thr) & (self.heatmap == neighborhood_max)
        
        # 步骤3：提取峰值点的坐标和值（i=x=列，j=y=行，和原代码一致）
        # np.where返回 (行索引, 列索引)，对应原代码的j, i
        j_coords, i_coords = np.where(peak_mask)
        values = self.heatmap[j_coords, i_coords]
        
        # 组合成 [i, j, value] 格式，和原代码keypoints一致
        keypoints = np.column_stack([i_coords, j_coords, values]).tolist()
    
        return keypoints
    
    def NMSKeyPoints(self, keypoints):
        if len(keypoints) < 2:
            return keypoints
        size = len(keypoints)
        sortIdx = []
        for i in range(len(keypoints)):
            sortIdx.append([i, keypoints[i][2]])
        #keypoints.sort(key=lambda x : x[2], reverse=True)
        sortIdx.sort(key=lambda x : x[1], reverse=True)
        delIdx = []
        thr = self.param['PointThrParam']['point_nms_min_dis']
        for i in range(size - 1):
            cur = keypoints[sortIdx[i][0]]
            for j in range(i+1, size):
                nxt = keypoints[sortIdx[j][0]]
                dis = max(abs(cur[0] - nxt[1]), abs(cur[1] - nxt[1]))
                if (dis <= thr):
                    delIdx.append(sortIdx[j][0])
        delIdx = list(set(delIdx))
        delIdx.sort(reverse=True)
        for idx in delIdx:
            del keypoints[idx]
        return keypoints

    def GetAllLines(self, keyPoints):
        keyPoints = sorted(keyPoints, key=lambda item: item[2], reverse=True)
        vertexElements = []

        for keypoint in keyPoints:
            candidateOrients = self.GenCandidateLines(keypoint)
            #list of [orientx, orienty, hist_score, act_len]
            candidateOrients.sort(key=lambda x : x[2], reverse=True) #sort by hist_score
            candidateOrients = self.NMSLines(candidateOrients)
            #vertex : point[x,y,s] orients_list[[sin,cos,hist_score,act_len],...]
            vertexElements.append([keypoint, candidateOrients])
        return vertexElements

    def GenCandidateLines(self, keypoint):
        vecmap = self.vecmap
        w = self.width
        h = self.height
        lineThrParam = self.param['LineThrParam']
        r = lineThrParam['len']
        bins = lineThrParam['bins']
        stride = int(360 / bins)
        active_thr = lineThrParam['active_value']
        detect_thr = lineThrParam['detect_value']
        min_active_thr = lineThrParam['min_active_len']
        hist_min_thr = lineThrParam['hist_min_score']
        hist = np.zeros(bins, np.float32)
        actlen = np.zeros(bins, np.float32)
        cx = keypoint[0]
        cy = keypoint[1]
        # print("cx cy ", cx, " ", cy)
        for bin in range(bins):
            rad = float(bin*stride) / 180 * math.pi
            cosr = math.cos(rad)
            sinr = math.sin(rad)
            for len in range(r + 1):
                dx = len*cosr
                dy = - len*sinr
                cntx = cx + dx
                cnty = cy + dy
                if (cntx < 0 or cntx > w - 1 or cnty < 0 or cnty > h - 1):
                    continue
                data = self.interPolaData(cntx, cnty)
                # print(data, " ")
                hist[bin] = hist[bin] + data
                if (data >= active_thr):
                    actlen[bin] = actlen[bin] + 1
                else :
                    if (actlen[bin] > 0 and data < detect_thr):
                       break
        candidateOrients = []
        for bin in range(bins):
            if (actlen[bin] < min_active_thr):
                continue
            if (hist[bin] < hist_min_thr):
                continue
            if (self.isPeakLine(hist, bins, bin)):
                thi = int(stride * bin + stride / 2.0)
                rad = 0.0
                if thi < 180:
                    rad = float(thi) * math.pi / 180.0
                else:
                    rad = float(thi - 180) * math.pi / 180.0 - math.pi
                # print("angle ", rad)
                orientx = math.cos(rad)
                orienty = - math.sin(rad)
                can_ori = [orientx, orienty, hist[bin], actlen[bin]]
                # print(can_ori)
                candidateOrients.append(can_ori)
        # print('GenCandidateLines')
        return candidateOrients

    def isPeakLine(self, hist, bins, bin):
        ret = 0
        preb = bin - 1
        nxtb = bin + 1
        if preb < 0:
            preb = bins - 1
        if nxtb == bins:
           nxtb = 0
        if (hist[bin] >= hist[preb] and hist[bin] >= hist[nxtb]):
            ret = 1
        return ret

    def interPolaData(self, fx, fy):
        w = self.width
        h = self.height
        vecmap = self.vecmap
        iu = int(fx)
        iv = int(fy)
        coex = fx - iu
        coey = fy - iv
        coex1 = 1 - coex
        coey1 = 1 - coey
        v1 = vecmap[iv][iu]
        v2 = v3 = v4 = 0.0
        if (iu < w - 1):
            v2 = vecmap[iv][iu + 1]
        if (iv < h - 1):
            v3 = vecmap[iv + 1][iu]
        if (iu < w - 1 and iv < h - 1):
            v4 = vecmap[iv + 1][iu + 1]
        data = coex1*coey1*v1 + coex*coey1*v2 + coex1*coey*v3 + coex*coey*v4
        return data

    def NMSLines(self, candidateOrients):
        #[orientx, orienty, hist_score, act_len]
        delIdx = []
        thr = self.param['LineThrParam']['match_cos_thr'] 
        size = len(candidateOrients)
        if (size < 2):
            return candidateOrients
        for i in range(size - 1):
            cur = candidateOrients[i]
            for j in range(i+1, size):
                nxt = candidateOrients[j]
                detla = cur[0] * nxt[0] + cur[1] * nxt[1]
                if (detla > thr):
                    delIdx.append(j)
        delIdx = list(set(delIdx))
        delIdx.sort(reverse=True)
        for idx in delIdx:
            del candidateOrients[idx]
        return candidateOrients

                
