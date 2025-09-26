import os
import re
import argparse
from typing import Iterable, List
from tensorboard.backend.event_processing import event_accumulator
from PIL import Image
import io


def extract_images_from_tensorboard(log_dir: str, output_dir: str):
    # 初始化一个事件累积器来读取日志数据
    print(f"Loading TensorBoard events from: {log_dir}")
    
    # 查找log目录下的所有事件文件
    event_files = []
    for root, _, files in os.walk(log_dir):
        for file in files:
            if file.startswith('events.out.tfevents'):
                event_files.append(os.path.join(root, file))
    
    if not event_files:
        print(f"No TensorBoard event files found in {log_dir}")
        return
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    print(f"Images will be saved to: {output_dir}")
    
    # 定义匹配模式: {name}/vis_{idx}
    pattern = re.compile(r'^(.+)/vis_\d+$')
    
    # 处理每个事件文件
    total_saved = 0
    
    for event_file in event_files:
        try:
            # 初始化事件累积器
            ea = event_accumulator.EventAccumulator(
                event_file,
                size_guidance={
                    event_accumulator.IMAGES: 0,  # 0 表示不限制大小，全部加载
                }
            )
            ea.Reload()  # 加载所有数据
            
            # 获取事件文件中所有图像的标签
            all_image_tags = list(ea.Tags().get('images', []))
            print(f"Found {len(all_image_tags)} image tags in {os.path.basename(event_file)}")
            
            # 筛选符合模式的标签
            matching_tags = [tag for tag in all_image_tags if pattern.match(tag)]
            print(f"Found {len(matching_tags)} tags matching pattern '{pattern.pattern}'")
            
            # 提取并保存匹配标签的图像
            for tag in matching_tags:
                # 从标签中提取name和idx信息
                match = pattern.match(tag)
                if match:
                    name = match.group(1)
                    
                    # 获取该标签下的所有图像条目
                    image_entries = ea.Images(tag)
                    
                    for index, image_entry in enumerate(image_entries):
                        # 图像数据存储在 image_entry.encoded_image_string
                        image_bytes = image_entry.encoded_image_string
                        # 使用 Pillow 打开图像字节流
                        image = Image.open(io.BytesIO(image_bytes))
                        
                        # 构造输出文件路径，包含name、idx、step和索引信息
                        step = image_entry.step
                        # 从标签中提取idx
                        idx = tag.split('/')[-1].replace('vis_', '')
                        
                        # 构建完整的文件名
                        filename = f'{name}_vis{idx}_step{step}_{index:03d}.png'
                        # 检查文件名中是否包含目录分隔符，需要创建相应目录
                        if '/' in filename:
                            file_dir = os.path.join(output_dir, os.path.dirname(filename))
                            os.makedirs(file_dir, exist_ok=True)
                        
                        output_path = os.path.join(output_dir, filename)
                        
                        # 保存图像
                        image.save(output_path)
                        total_saved += 1
                        if total_saved % 100 == 0:
                            print(f"Saved {total_saved} images so far...")
        except Exception as e:
            print(f"Error processing {event_file}: {str(e)}")
    
    print(f"Image extraction completed. Total saved: {total_saved} images.")


if __name__ == "__main__":
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='Extract images from TensorBoard logs with format {name}/vis_{idx}')
    parser.add_argument('--log_dir', type=str, required=True, help='Path to TensorBoard log directory')
    parser.add_argument('--output_dir', type=str, default='extracted_images', help='Path to output directory')
    
    args = parser.parse_args()
    
    # 调用提取函数
    extract_images_from_tensorboard(args.log_dir, args.output_dir)