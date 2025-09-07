import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

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
                                       markersize=8, alpha=0.8, markeredgecolor='black', markeredgewidth=1)
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