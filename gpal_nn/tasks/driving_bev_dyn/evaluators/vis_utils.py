import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

# def plot_pr_curves(pr_curves, class_names, name, save_path=None):
#     """
#     绘制PR曲线
#     # Arguments
#         pr_curves: 从ap_per_class_with_curves返回的PR曲线数据
#         class_names: 类别名称列表
#         save_path: 保存图片的路径，如果为None则显示图片
#     """
#     plt.figure(figsize=(10, 8))
    
#     for class_idx, curve_data in pr_curves.items():
#         if len(curve_data['recall']) > 0 and len(curve_data['precision']) > 0:
#             # 只绘制有效的曲线
#             if not (len(curve_data['recall']) == 1 and curve_data['recall'][0] == 0):
#                 plt.plot(curve_data['recall'], curve_data['precision'], 
#                         label=f'{class_names[class_idx-1]} (Class {class_idx})', 
#                         linewidth=2)
    
#     plt.xlabel('Recall')
#     plt.ylabel('Precision')
#     plt.title(f'Precision-Recall Curves {name}')
#     plt.legend()
#     plt.grid(True, alpha=0.3)
#     plt.xlim([0, 1])
#     plt.ylim([0, 1])
    
#     if save_path:
#         plt.savefig(save_path, dpi=300, bbox_inches='tight')
#     else:
#         plt.show()

def plot_pr_curves(pr_curves, extra_info, class_names, unique_classes, recall_at_p50, recall_at_precision=None, save_path=None):
    """
    Plot PR curves for all classes, adapted for boundary cases
    
    # Arguments
        pr_curves: Dictionary containing PR curves for each class
        class_names: List of class names
        unique_classes: Unique class indices
        recall_at_p50: Recall values at Precision=0.5 for each class
        recall_at_precision: Dictionary containing R@P values for each class
        save_path: Path to save the plot (optional)
    """
    plt.figure(figsize=(12, 9))
    
    # Generate colors for each class
    colors = plt.cm.Set3(np.linspace(0, 1, len(unique_classes)))
    
    valid_curves_count = 0  # 统计有效曲线数量
    
    for i, c in enumerate(unique_classes):
        if c in pr_curves:
            recall = pr_curves[c]['recall']
            precision = pr_curves[c]['precision']
            num_dt = pr_curves[c]['num_Dt']
            num_gt = pr_curves[c]['num_Gt']
            
            # 检查是否有有效的PR曲线数据
            has_valid_curve = False
            
            if len(recall) > 0 and len(precision) > 0:
                # 检查是否为有意义的曲线（不只是单点(0,0)或空数组）
                if not (len(recall) == 1 and recall[0] == 0.0 and precision[0] == 0.0):
                    # 进一步检查是否有变化的数据点
                    if len(recall) > 1 or (len(recall) == 1 and (recall[0] > 0 or precision[0] > 0)):
                        has_valid_curve = True
            
            if has_valid_curve:
                class_name = class_names[c-1] if c-1 < len(class_names) else f'Class {c}'
                
                # 构建标签，包含基本信息和R@P信息
                label = f'{class_name:<25} (Dt:{num_dt}, Gt:{num_gt}'
                
                if recall_at_precision and c in recall_at_precision:
                    r_at_p_values = []
                    for key, val in recall_at_precision[c].items():
                        if val >= 0:  # 排除无效值(-1)
                            r_at_p_values.append(f'{key}={val:.3f}')
                    if r_at_p_values:
                        label += f', {", ".join([f"{val:<10}" for val in r_at_p_values])}'
                label += ')'
                
                # 绘制PR曲线
                plt.plot(recall, precision, 
                        color=colors[i], 
                        linewidth=2, 
                        label=label,
                        marker='o' if len(recall) <= 10 else None,  # 少于10个点时显示标记
                        markersize=4)
                
                # 标记R@P特定点
                if recall_at_precision and c in recall_at_precision:
                    for key, r_val in recall_at_precision[c].items():
                        if r_val > 0:  # 只标记有效的点
                            try:
                                p_val = float(key.split('P')[1])  # 从'R@P0.5'提取0.5
                                plt.plot(r_val, p_val, 'o', color=colors[i], 
                                       markersize=8, alpha=0.8, markeredgecolor=colors[i], markeredgewidth=1)
                            except (IndexError, ValueError):
                                continue  # 跳过格式错误的key
                
                valid_curves_count += 1
            
            else:
                # 对于无有效曲线的类别，在图例中显示但不绘制曲线
                class_name = class_names[c-1] if c-1 < len(class_names) else f'Class {c}'
                
                # 确定状态描述
                if num_dt == 0 and num_gt == 0:
                    status = "No data"
                elif num_dt == 0:
                    status = "No predictions"
                elif num_gt == 0:
                    status = "No ground truth"
                else:
                    status = "Invalid curve"
                
                # 添加到图例但不绘制线条
                plt.plot([], [], color=colors[i], linewidth=0, 
                        label=f'{class_name:<25} (Dt:{num_dt}, Gt:{num_gt}, {status})')
    
    # 如果没有任何有效曲线，添加提示
    if valid_curves_count == 0:
        plt.text(0.5, 0.5, 'No valid PR curves to display', 
                ha='center', va='center', transform=plt.gca().transAxes,
                fontsize=16, bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    
    # Add precision reference lines
    if recall_at_precision:
        # 获取所有精度点用于绘制参考线
        all_precision_points = set()
        for class_data in recall_at_precision.values():
            for key in class_data.keys():
                try:
                    p_val = float(key.split('P')[1])
                    all_precision_points.add(p_val)
                except (IndexError, ValueError):
                    continue
        
        # 绘制精度参考线
        for p_val in sorted(all_precision_points):
            plt.axhline(y=p_val, color='gray', linestyle='--', alpha=0.5, linewidth=1,
                       label=f'P={p_val}' if p_val == min(all_precision_points) else None)
    else:
        # 如果没有R@P数据，至少画P=0.5参考线
        plt.axhline(y=0.5, color='red', linestyle='--', alpha=0.7, linewidth=1, label='P=0.5')
    
    plt.xlabel(f'Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.title(f'Precision-Recall Curves {extra_info}', fontsize=14)
    
    plt.legend(
        loc='lower center',             # 图例放置在底部中间
        bbox_to_anchor=(0.5, -0.25),    # 锚点位置（x,y）控制精确位置
        fontsize='small',               # 减小字体尺寸
        frameon=True,                   # 显示边框
        fancybox=True,                  # 圆角边框
        ncol=1 if len(unique_classes) <= 5 else 2  # 动态调整列数
    )
    
    plt.grid(True, alpha=0.3)
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    
    ax = plt.gca()
    ax.xaxis.set_major_locator(ticker.MultipleLocator(0.1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(0.1))
    ax.grid(True, which='major', alpha=0.5, linestyle='-')

    # 可选的次要网格线
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.05))
    ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.05))
    ax.grid(True, which='minor', alpha=0.1, linestyle='--')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()  # 保存后关闭图形以释放内存
    else:
        plt.show()
        
        
def visualize_pred_scores_and_gt_3dbox_use_bev_and_side_box(pts_range,
                                                            class_colors,
                                                            points,
                                                            pred_boxes,
                                                            pred_label_ids,
                                                            pred_scores,
                                                            gt_boxes,
                                                            gt_boxes_label_ids,
                                                            save_imgfile='',
                                                            frame_id='',
                                                            save_dir='',
                                                            fnfp_info = None,
                                                            ):
    """可视化BEV和侧视图场景"""
    if class_colors is None:
        class_colors = ['k', 
                    'blue', 
                    'green', 
                    'yellow', 
                    'purple', 
                    'orange', 
                    'pink', 
                    'brown', 
                    'cyan'
                    ]
    os.makedirs(save_dir, exist_ok=True)
    save_path = save_imgfile
    
    # 创建图形布局
    x_min, y_min, z_min, x_max, y_max, z_max = pts_range
    CLASS_COLORS = class_colors
    
    if x_max > 100:
        fontsize = 8
    if x_max > 140:
        fontsize = 10
    if x_max > 180:
        fontsize = 12
    
    atios = 15
    W, H = int((x_max-x_min)/atios), (int(y_max-y_min)/atios)
    fig = plt.figure(figsize=(W, H))
    plt.rcParams['xtick.labelsize'] = fontsize - 1
    plt.rcParams['ytick.labelsize'] = fontsize - 1
    
    # 修改gridspec设置
    x_range = (x_max - x_min)
    y_range = (y_max - y_min)
    z_range = (z_max - z_min)

    # 根据实际数据范围计算高度比例
    height_ratio_bev = y_range
    height_ratio_side = z_range

    gs = fig.add_gridspec(2, hspace=0, wspace=0, 
                          height_ratios=[height_ratio_side, height_ratio_bev])
    ax_side, ax_bev = gs.subplots()
    
    # 原点方向
    ax_bev.plot([0, 3], [0, 0], color='red', linewidth=1.0)
    ax_bev.plot([0, 0], [0, 3], color='green', linewidth=1.0)

    if points is not None:
        points_vis = points[:, :3]  # 假设前3列是xyz坐标
        # 应用点云范围过滤
        mask = ((points_vis[:, 0] >= x_min) & (points_vis[:, 0] <= x_max) &
                (points_vis[:, 1] >= y_min) & (points_vis[:, 1] <= y_max) &
                (points_vis[:, 2] >= z_min) & (points_vis[:, 2] <= z_max))
        points_vis = points_vis[mask]
        
        # BEV: 绘制 X-Y 平面
        vmin_val = np.percentile(points_vis[:, 2], 10)  # 取第10百分位数
        vmax_val = np.percentile(points_vis[:, 2], 90)  # 取第90百分位数
        scatter_bev = ax_bev.scatter(points_vis[:, 0], points_vis[:, 1], 
                                        c=points_vis[:, 2], 
                                        marker='.',
                                    #  cmap='viridis', 
                                    # vmin=vmin_val,
                                    # vmax=vmax_val,
                                    cmap='gray',
                                    s=0.01, 
                                        )
        # 侧视图: 绘制 X-Z 平面
        scatter_side = ax_side.scatter(points_vis[:, 0], points_vis[:, 2], 
                                        c=points_vis[:, 1], 
                                        marker=',',
                                        cmap='plasma', 
                                        s=0.05, alpha=0.6)
        # plt.colorbar(scatter, ax=ax_bev, label='Z (m)', shrink=0.8)  # 添加颜色条
    
    for i, bbox in enumerate(pred_boxes):
        pred_score = pred_scores[i]
        center  = bbox[:3]
        
        exist_in = ((center[0] >= x_min) & (center[0] <= x_max) &
                    (center[1] >= y_min) & (center[1] <= y_max) &
                    (center[2] >= z_min) & (center[2] <= z_max))
        if not exist_in:
            continue
        
        pred_label_id = pred_label_ids[i]
        
        # 计算矩形的四个角点
        corners = get_8_corners(bbox)
        color = CLASS_COLORS[pred_label_id]
        
        # BEV: 绘制 X-Y 平面
        bev_corners      = corners[[6,7,4,5]]
        bev_corners_head = corners[[5,6]]
        ax_bev.plot(bev_corners[:, 0], bev_corners[:, 1], color=color, linewidth=1.0)
        ax_bev.plot(bev_corners_head[:, 0], bev_corners_head[:, 1], color='k', linewidth=1.0)

        # 中心朝线
        front_center = (corners[5] + corners[6]) / 2
        f1_center = (front_center + center) / 2
        f2_center = front_center * 2 - f1_center
        ax_bev.plot([f2_center[0], front_center[0]], [f2_center[1], front_center[1]], color='k', linewidth=1.0)
        # ax_bev.fill(corners[:, 0], corners[:, 1], color=color, alpha=0.2)  # 填充颜色
        
        # track_id
        ax_bev.text(center[0], center[1], str(''), 
                    ha='center', va='center', fontsize=fontsize-2,
                    #bbox=dict(boxstyle="round,pad=0.1", facecolor='white', alpha=0.8)
                    )
                    
        # 侧视图: 绘制 X-Z 平面
        xz_corners      = corners[[5,4,0,1]]
        xz_corners_head = corners[[1,5]]
        ax_side.plot(xz_corners[:, 0], xz_corners[:, 2], color=color, linewidth=0.5)
        ax_side.plot(xz_corners_head[:, 0], xz_corners_head[:, 2], color='k', linewidth=0.5)


    for i, bbox in enumerate(gt_boxes):
        # pred_score = pred_scores[i]
        center  = bbox[:3]
        
        exist_in = ((center[0] >= x_min) & (center[0] <= x_max) &
                    (center[1] >= y_min) & (center[1] <= y_max) &
                    (center[2] >= z_min) & (center[2] <= z_max))
        if not exist_in:
            continue
        
        # pred_label_id = pred_label_ids[i]
        gt_label_id = gt_boxes_label_ids[i]
        
        # 计算矩形的四个角点
        corners = get_8_corners(bbox)
        color = CLASS_COLORS[int(gt_label_id)]
        
        # BEV: 绘制 X-Y 平面
        bev_corners      = corners[[6,7,4,5]]
        bev_corners_head = corners[[5,6]]
        ax_bev.plot(bev_corners[:, 0], bev_corners[:, 1], color='k', linewidth=1.0)
        ax_bev.plot(bev_corners_head[:, 0], bev_corners_head[:, 1], color='k', linewidth=1.0)

        # 中心朝线
        front_center = (corners[5] + corners[6]) / 2
        f1_center = (front_center + center) / 2
        f2_center = front_center * 2 - f1_center
        ax_bev.plot([f2_center[0], front_center[0]], [f2_center[1], front_center[1]], color='k', linewidth=1.0)
        # ax_bev.fill(corners[:, 0], corners[:, 1], color=color, alpha=0.2)  # 填充颜色
        
        # track_id
        txt = f'gt_{i}'
        text_obj = ax_bev.text(center[0], center[1], txt, 
                    ha='center', va='center', fontsize=fontsize-2, 
                    #bbox=dict(boxstyle="round,pad=0.1", facecolor='white', alpha=0.8)
                    )
                    
        # 侧视图: 绘制 X-Z 平面
        xz_corners      = corners[[5,4,0,1]]
        xz_corners_head = corners[[1,5]]
        ax_side.plot(xz_corners[:, 0], xz_corners[:, 2], color=color, linewidth=0.5)
        ax_side.plot(xz_corners_head[:, 0], xz_corners_head[:, 2], color='k', linewidth=0.5)
    
    
    for curr_class_str, curr_info in fnfp_info.items():
        
        curr_class_id = int(curr_class_str)
        curr_class_color = CLASS_COLORS[curr_class_id]
        
        fn_case_infos = curr_info['fn_cases']
        fp_case_infos = curr_info['fp_cases']
        
        for fn_case_info in fn_case_infos:
            pred_box = fn_case_info['gt_box']
            pred_center = pred_box[:3]
            # fn_corners = get_8_corners(fn_case_info)
            # ax_bev.plot(fn_corners[:, 0], fn_corners[:, 1], color=curr_class_color, linewidth=1.0)
            ax_bev.plot(pred_center[0], pred_center[1], color=curr_class_color, linewidth=1.0, marker='x')
        
        
        for fp_case_info in fp_case_infos:
            
            pred_box = fp_case_info['pred_box']
            pred_center = pred_box[:3]
            # pred_corners = get_8_corners(pred_box)
            # ax_bev.plot(pred_corners[:, 0], pred_corners[:, 1], color=curr_class_color, linewidth=1.0, m)
            ax_bev.plot(pred_center[0], pred_center[1], color=curr_class_color, linewidth=1.0, marker='+')
        
    # === 设置BEV图属性 ===
    ax_bev.set_xlim(x_min, x_max)
    ax_bev.set_ylim(y_min, y_max)
    ax_bev.set_xlabel('X (m)', fontsize=fontsize)
    ax_bev.set_ylabel('Y (m)', fontsize=fontsize)
    ax_bev.set_aspect('equal')
    # ax_bev.set_title(f"Bird's Eye View (BEV)(Y-axis) {timestamp}", fontsize=14, fontweight='bold')
    ax_bev.grid(True, alpha=0.3)
    # ax_bev.legend()
    
    symbol_legend = [Line2D([], [], color='black', marker='x', linestyle='None', label='FN (x)'),
                     Line2D([], [], color='black', marker='+', linestyle='None', label='FP (+)')]
    legend_elements = [Patch(facecolor=CLASS_COLORS[i], edgecolor='k', label=f'Class {i}') 
                       for i in range(len(CLASS_COLORS))] + symbol_legend
    # ax_bev.legend(handles=legend_elements, loc='upper right')
    # ax_bev.legend(handles=symbol_legend, loc='lower right')
    ax_bev.legend(handles=legend_elements, 
                loc='upper right',
                fontsize=6,                    # 字体大小：可以是数字或 'small', 'medium', 'large' 等
                framealpha=0.25,                # 背景透明度
                # fancybox=True,                 # 圆角边框
                # shadow=True,                   # 阴影
                frameon=True,                  # 显示边框
                facecolor='white',             # 背景颜色
                # edgecolor='gray',              # 边框颜色
                borderpad=0.1,                 # 图例内边距
                columnspacing=1.0,             # 列间距
                handlelength=2.0,              # 图例标记长度
                handletextpad=0.5,             # 标记和文字间距
                labelspacing=0.1)              # 标签间距
    
    # === 设置侧视图属性 ===
    ax_side.set_xlim(x_min, x_max)
    ax_side.set_ylim(z_min, z_max)
    # ax_side.set_xlabel('X (m)', fontsize=12)
    ax_side.set_ylabel('Z (m)', fontsize=fontsize)
    # ax_side.set_title('Side View (Y-axis perspective)', fontsize=14, fontweight='bold')
    ax_side.set_title(f"Bird's Eye View (BEV)(Y-axis) {frame_id}", fontsize=fontsize+1, fontweight='bold')
    ax_side.set_aspect('equal')
    ax_side.grid(True, alpha=0.3)
    
    # 遍历所有文本对象进行检查, 超出边界则剔除，保证图像尺寸一致
    fig.canvas.draw()  # 强制渲染
    for text_obj in ax_bev.texts:
        if text_obj.get_text().startswith('gt_'):
            bbox = text_obj.get_window_extent().transformed(ax_bev.transData.inverted())
            text_xmin, text_ymin = bbox.x0, bbox.y0
            text_xmax, text_ymax = bbox.x1, bbox.y1
            
            curr_xlim = ax_bev.get_xlim()
            curr_ylim = ax_bev.get_ylim()
            
            if (text_xmin < curr_xlim[0] or text_xmax > curr_xlim[1] or 
                text_ymin < curr_ylim[0] or text_ymax > curr_ylim[1]):
                text_obj.remove()
    
    plt.tight_layout()
    plt.savefig(f'{save_path}', dpi=300, bbox_inches='tight')
    plt.close()
    
def get_8_corners(box) -> np.ndarray:
    """获取3D框的8个角点坐标"""
    l, w, h = box[3:6]
    
    # 定义8个角点（相对于中心点）
    corners = np.array([
        [-l/2, -w/2, -h/2],  # 0: 后右下
        [l/2, -w/2, -h/2],   # 1: 前右下
        [l/2, w/2, -h/2],    # 2: 前左下
        [-l/2, w/2, -h/2],   # 3: 后左下
        [-l/2, -w/2, h/2],   # 4: 后右上
        [l/2, -w/2, h/2],    # 5: 前右上
        [l/2, w/2, h/2],     # 6: 前左上
        [-l/2, w/2, h/2]     # 7: 后左上
    ])
    
    # 应用旋转（只考虑yaw角）
    yaw = box[6]
    rotation_matrix = np.array([
        [np.cos(yaw), -np.sin(yaw), 0],
        [np.sin(yaw), np.cos(yaw), 0],
        [0, 0, 1]
    ])
    
    rotated_corners = corners @ rotation_matrix.T
    
    # 平移到世界坐标
    world_corners = rotated_corners + box[:3]
    
    return world_corners