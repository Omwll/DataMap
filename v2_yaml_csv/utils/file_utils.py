"""
文件写入模块 - 处理CSV和YAML文件的读写
"""
import os
import shutil
import yaml
from collections import defaultdict
from datetime import datetime
from utils.logger import get_logger

# 获取全局日志对象
logger = get_logger()

def write_csv_file(csv_file_path, data_list, has_header=True):
    """
    将数据写入CSV文件
    
    参数:
    - csv_file_path: CSV文件路径
    - data_list: 数据列表，每个元素为(相对路径, 标签)元组
    - has_header: 是否写入标题行
    """
    # 确保目录存在
    os.makedirs(os.path.dirname(os.path.abspath(csv_file_path)), exist_ok=True)
    
    with open(csv_file_path, 'w', encoding='utf-8') as f:
        # 写入标题行
        if has_header:
            f.write("rel_path,label\n")
        
        # 按标签排序
        sorted_data = sorted(data_list, key=lambda x: x[1])
        
        # 写入数据行
        for rel_img_path, label in sorted_data:
            f.write(f"{rel_img_path},{label}\n")
    
    logger.info(f"CSV文件已生成: {csv_file_path}")

def write_yaml_file(yaml_file_path, data_dict):
    """
    将数据写入YAML文件，根据标签信息排序
    
    参数:
    - yaml_file_path: YAML文件路径
    - data_dict: 数据字典
    """
    # 确保目录存在
    os.makedirs(os.path.dirname(os.path.abspath(yaml_file_path)), exist_ok=True)
    
    # 如果包含class_info，对其进行排序
    if 'class_info' in data_dict:
        # 按标签排序类别信息
        sorted_class_info = {}
        # 创建(类别名称, 标签)的列表
        classes_with_labels = []
        for class_name, info in data_dict['class_info'].items():
            if 'label' in info:
                classes_with_labels.append((class_name, info['label']))
        
        # 按标签排序
        sorted_classes = sorted(classes_with_labels, key=lambda x: x[1])
        
        # 重建排序后的class_info字典
        for class_name, _ in sorted_classes:
            sorted_class_info[class_name] = data_dict['class_info'][class_name]
        
        # 替换原来的class_info
        data_dict['class_info'] = sorted_class_info
    
    # 如果存在label_mapping，也对其进行排序
    if 'label_mapping' in data_dict:
        sorted_mapping = {}
        # 创建(类别名称, 新标签)的列表
        classes_with_new_labels = []
        for class_name, mapping in data_dict['label_mapping'].items():
            if 'new_label' in mapping:
                classes_with_new_labels.append((class_name, mapping['new_label']))
        
        # 按新标签排序
        sorted_mapping_classes = sorted(classes_with_new_labels, key=lambda x: x[1])
        
        # 重建排序后的label_mapping字典
        for class_name, _ in sorted_mapping_classes:
            sorted_mapping[class_name] = data_dict['label_mapping'][class_name]
        
        # 替换原来的label_mapping
        data_dict['label_mapping'] = sorted_mapping
    
    # 写入排序后的字典
    with open(yaml_file_path, 'w', encoding='utf-8') as f:
        yaml.dump(data_dict, f, default_flow_style=False, sort_keys=False)
    
    logger.info(f"YAML文件已生成: {yaml_file_path}")

def read_csv_file(csv_file_path, has_header=True):
    """
    从CSV文件读取数据
    
    参数:
    - csv_file_path: CSV文件路径
    - has_header: 是否有标题行
    
    返回:
    - 数据列表，每个元素为(相对路径, 标签)元组
    """
    data_list = []
    
    with open(csv_file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        
        # 如果有标题行，跳过第一行
        start_idx = 1 if has_header else 0
        
        for line in lines[start_idx:]:
            line = line.strip()
            if line:
                parts = line.split(',')
                if len(parts) >= 2:
                    rel_path = parts[0]
                    label = int(parts[1])
                    data_list.append((rel_path, label))
    
    logger.info(f"从CSV文件读取了 {len(data_list)} 条数据: {csv_file_path}")
    return data_list

def read_yaml_file(yaml_file_path):
    """
    从YAML文件读取数据
    
    参数:
    - yaml_file_path: YAML文件路径
    
    返回:
    - 数据字典
    """
    with open(yaml_file_path, 'r', encoding='utf-8') as f:
        data_dict = yaml.safe_load(f)
    
    logger.info(f"从YAML文件读取了数据: {yaml_file_path}")
    return data_dict

# save split files ------------------------------------------------------------------------------



# copy split images ------------------------------------------------------------------------------
def copy_image_files(split_data, root_dir, output_dir, split_name):
    """
    将划分后的图像文件复制到指定的输出目录
    
    参数:
    - split_data: 划分数据列表，格式为[(相对路径, 标签), ...]
    - root_dir: 原始数据集根目录
    - output_dir: 输出根目录
    - split_name: 划分名称 (train/val/test)
    
    返回:
    - 复制的文件数量
    """
    # 创建目标目录
    target_dir = os.path.join(output_dir, split_name)
    os.makedirs(target_dir, exist_ok=True)
    
    copied_count = 0
    
    # 遍历所有图像文件
    for rel_path, label in split_data:
        # 构建源文件和目标文件路径
        src_file = os.path.join(root_dir, rel_path)
        
        # 创建目标子目录 (按标签组织)
        label_dir = os.path.join(target_dir, f"class_{label}")
        os.makedirs(label_dir, exist_ok=True)
        
        # 获取文件名并构建目标文件路径
        filename = os.path.basename(rel_path)
        dst_file = os.path.join(label_dir, filename)
        
        # 复制文件
        try:
            shutil.copy2(src_file, dst_file)
            copied_count += 1
            
            # 每复制100个文件记录一次日志
            # if copied_count % 100 == 0:
            #     logger.debug(f"已复制 {copied_count} 个文件到 {split_name} 目录")
        except Exception as e:
            logger.error(f"复制文件 {src_file} 失败: {str(e)}")
    
    logger.info(f"成功复制 {copied_count} 个文件到 {split_name} 目录")
    return copied_count

def copy_split_files(splits, root_dir, output_dir):
    """
    将所有划分的数据复制到按划分名称组织的目录结构中
    
    参数:
    - splits: 划分数据字典，格式为{'train': [...], 'val': [...], 'test': [...]}
    - root_dir: 原始数据集根目录
    - output_dir: 输出根目录
    
    返回:
    - 包含各划分复制文件数量的字典
    """
    # 创建数据集根目录
    dataset_dir = os.path.join(output_dir, "dataset")
    os.makedirs(dataset_dir, exist_ok=True)
    
    # 记录各划分复制的文件数量
    copy_stats = {}
    
    # 复制各划分的文件
    for split_name, split_data in splits.items():
        if split_data:
            copy_count = copy_image_files(split_data, root_dir, dataset_dir, split_name)
            copy_stats[split_name] = copy_count
    return copy_stats