import os
import numpy as np
import cv2
import json
import glob as glob

def get_slot_box(lines, linedict):
    min_x = 999999
    min_y = 999999
    max_x = 0
    max_y = 0
    for line_id in lines:
        line_info = linedict[line_id]
        pts = line_info['pt']
        if len(pts) != 2:
            print("line points num is error")
            continue
      
        x0 = int(float(pts[0]['x']))
        y0 = int(float(pts[0]['y']))
        x1 = int(float(pts[1]['x']))
        y1 = int(float(pts[1]['y']))
        if x0 < min_x:
            min_x = x0
        if x1 < min_x:
            min_x = x1
        
        if y0 < min_y:
            min_y = y0
        
        if y1 < min_y:
            min_y = y1
        
        if x0 > max_x:
            max_x = x0
        if x1 > max_x:
            max_x = x1
        
        if y0 > max_y:
            max_y = y0
        if y1 > max_y:
            max_y = y1
    width = max_x - min_x
    height = max_y - min_y


def draw_transparent_mask(image, mask_points, color, alpha=0.5):
    """
    在原图上绘制半透明的mask
    
    参数:
    image: 原始图像 (numpy array, BGR格式)
    mask_points: mask的顶点坐标列表 [(x1, y1), (x2, y2), ...]
    color: 掩码颜色 (BGR元组), 默认为绿色
    alpha: 透明度 (0-1之间的浮点数), 数值越小越透明
    
    返回:
    result: 添加了半透明mask的图像
    """
    # 创建与原图大小相同的空白掩码
    mask = np.zeros_like(image)
    
    # 将mask_points转换为numpy数组并绘制填充多边形
    pts = np.array(mask_points, np.int32)
    pts = pts.reshape((-1, 1, 2))
    cv2.fillPoly(mask, [pts], color)
    
    # 将掩码与原图进行alpha混合
    result = cv2.addWeighted(image, 1, mask, alpha, 0)
    
    return result

def sort_points_convex(points):
    """对凸多边形的顶点进行排序（顺时针或逆时针）"""
    # 计算质心
    points = np.array(points)
    centroid = np.mean(points, axis=0)
    
    # 计算每个点相对于质心的角度
    angles = np.arctan2(points[:,1] - centroid[1], points[:,0] - centroid[0])
    
    # 根据角度排序
    sorted_indices = np.argsort(angles)
    sorted_points = points[sorted_indices]
    
    return sorted_points, centroid

def generate_colors(n, seed=42):
    """生成n个均匀分布的颜色"""
    np.random.seed(seed)
    colors = []
    
    # 基础颜色
    base_colors = np.array([
        [0, 0, 255],    # 红
        [0, 255, 0],    # 绿
        [255, 0, 0],    # 蓝
        [0, 255, 255],  # 黄
        [255, 255, 0],  # 青
        [255, 0, 255],  # 洋红
    ])
    
    # 如果需要的颜色数量少于基础颜色，直接返回部分基础颜色
    if n <= len(base_colors):
        return [tuple(c) for c in base_colors[:n]]
    
    # 生成额外的颜色
    colors.extend([tuple(c) for c in base_colors])
    
    # 在HSV空间生成剩余颜色以确保均匀分布
    for i in range(n - len(base_colors)):
        # 在HSV空间均匀采样
        hue = i / (n - len(base_colors))
        saturation = 0.8 + np.random.rand() * 0.2  # 饱和度在0.8-1.0之间
        value = 0.8 + np.random.rand() * 0.2      # 亮度在0.8-1.0之间
        
        # 转换为BGR
        hsv = np.uint8([[[hue * 180, saturation * 255, value * 255]]])
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0][0]
        colors.append(tuple(map(int, bgr)))
    
    return colors

def draw_json_object(json_path, img):
    img_show_ptline = img.copy()
    imgh, imgw, _ = img.shape
    print(json_path)
    colors = [
        (0, 0, 255),    # 红色, 垂直
        (0, 255, 0),    # 绿色, 水平
        (255, 255, 0),  # 青色, 斜列
    ]
    limiter_color = [
        (255, 0, 0),
        (0, 255, 0)
    ]

    # keypoint_type_dict = ["T", "L", "I", "U", "Oblique_T", "Oblique_L"]
    slot_type_dict = ["vertical_parking", "Parallel_parking", "Inclined_parking"]
    limiter_dict = ["block_limiter", "rod_limiter"]
    json_list = json.load(open(json_path))
    annos = json_list["annotation"]
    keyline_objs = annos["object"]
    slot_objs = annos["slot"]
    keypointline = annos["keypointline"]

    keydict = {}
    linedict = {}
    limiterdict = {}
    for keyline in keyline_objs:
        if keyline["name"] == "keypoint":
            key_idx = int(keyline["point_id"])
            keydict[key_idx] = keyline
        if keyline["name"] == "line":
            line_idx = int(keyline['line_id'])
            linedict[line_idx] = keyline

        if keyline["name"] == "limiter":
            limiter_id = int(keyline["limiter_id"])
            limiterdict[limiter_id] = keyline

    for slot in slot_objs:
        keyPoints = slot["keyPoints"]
        lines = slot["lines"]
        parking_type = slot["parking_type"]
        slotid = slot["parking_id"]
        is_occupied = slot["is_occupied"]
        limiter = slot["limiter"]

        draw_slot_color = (0, 0, 0)
        slot_points = []
        for idx, slot_type in enumerate(slot_type_dict):
            if parking_type == slot_type:
                draw_slot_color = colors[idx]
                break
        
        for key_id in keyPoints:
            key_info = keydict[key_id]
            if len(key_info['pt']) != 1 and len(key_info['pt']) > 0:
                raise ValueError("key point num must is 1")
            pt = key_info['pt'][0]
            x = int(float(pt['x']))
            y = int(float(pt['y']))
            cv2.circle(img, (x, y), 10, draw_slot_color, -1)
            
  
        for line_id in lines:
            print(line_id)
            line_info = linedict[line_id]
            pts = line_info['pt']
            if len(pts) != 2 and len(pts) > 0:
                print(pts)
                raise ValueError("line points num must is 2")
               
            x0 = int(float(pts[0]['x']))
            y0 = int(float(pts[0]['y']))
            x1 = int(float(pts[1]['x']))
            y1 = int(float(pts[1]['y']))
            cv2.line(img, (x0, y0), (x1, y1), draw_slot_color, 3)
            slot_points.append([x0, y0])
            slot_points.append([x1, y1])
        #===============limiter=====================
        if len(limiter) >= 1:
            limiter_pts_int = []
            limiter_id = limiter[0]
            limiter_info = limiterdict[int(limiter_id)]
            if len(limiter_info['pt']) != 4 and len(limiter_info['pt']) > 0:
                raise ValueError("limiter points num must is 4")
            
            for pt in limiter_info['pt']:
                x0 = int(float(pt['x']))
                y0 = int(float(pt['y']))
                limiter_pts_int.append((x0, y0))
            limiter_pts = np.array(limiter_pts_int)
            limiter_pts = limiter_pts.reshape((-1, 1, 2))
            cv2.fillPoly(img, [limiter_pts], (255, 0, 0))

        if (len(slot_points) > 0):
            slot_points, center_pt = sort_points_convex(slot_points)
            font = cv2.FONT_HERSHEY_SIMPLEX

            cv2.putText(img, str(slotid), (int(center_pt[0]), int(center_pt[1])), font, 2.0, (255, 0, 0))
            if is_occupied == True:
                img = draw_transparent_mask(img, slot_points, draw_slot_color, alpha=0.8)

    n = len(keydict)
    color_n = generate_colors(n)
    idx = 0
    
    mask_img_list = []
    for keypl in keypointline:
        mask_img = np.zeros((imgh, imgw, 3), dtype=np.uint8)

        if len(keypl["keyid"]) != 1:
            raise ValueError("keyid must only 1")
        keyID = int(keypl["keyid"][0])
        lineID_list = keypl["lineid"]
        key_pt = keydict[keyID]["pt"][0]
        color = color_n[idx]
        color = tuple(int(n) for n in color)
        print(type(color))
        x = int(float(key_pt['x']))
        y = int(float(key_pt['y']))

        cv2.circle(mask_img, (x, y), 10, color, -1)
        

        for lineID in lineID_list:
            pts = linedict[int(lineID)]["pt"]
            if len(pts) != 2:
                print("line points num is error")
                continue
            x0 = int(float(pts[0]['x']))
            y0 = int(float(pts[0]['y']))
            x1 = int(float(pts[1]['x']))
            y1 = int(float(pts[1]['y']))
            cv2.line(mask_img, (x0, y0), (x1, y1), color, 3)
        idx += 1
        mask_img_list.append(mask_img)

    img_show_ptline = img_show_ptline * 0.5
    for mask_img in mask_img_list:
        img_show_ptline += mask_img * 0.5

    return img, img_show_ptline
                
            
def visul_label(json_root, save_visual_root):
    json_paths = glob.glob(os.path.join(json_root, "*/*/*/*.json"))
    print('json_lenth ', len(json_paths))
    for json_path in json_paths:
        json_file = json_path.split("/")[-1]
        jpg_name = json_file.replace(".json", ".jpg")
        img_path = json_path.replace("label", "avm").replace(".json", ".jpg")
        #========show====================
        img = cv2.imread(img_path)
        if img is None:
            print("{} img is None ", img_path)
        show_img = img.copy()
        show_img, img_show_ptline = draw_json_object(json_path, show_img)
        # show_old_img = 
        show_three_img = np.hstack((img, show_img, img_show_ptline))
        visual_img_path = os.path.join(save_visual_root, jpg_name)
        cv2.imwrite(visual_img_path, show_three_img)

if __name__ == "__main__":
    json_root = "/home/gpal/gpal_work/ParkingSlot/parking_slot/datasets_tmp/0817/20250817_commit_valid2"
    save_visual_root = "/home/gpal/gpal_work/ParkingSlot/parking_slot/datasets_tmp/0817/20250817_commit_valid2_slot_visual"
    if not os.path.exists(save_visual_root):
        os.makedirs(save_visual_root)
    visul_label(json_root, save_visual_root)
    print('end')

