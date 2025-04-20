import os
from collections import defaultdict
from datetime import datetime
from utils.logger import get_logger
import pandas as pd
import yaml

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.file_utils import write_csv_file, write_yaml_file

class DatasetProcessor:
    """数据集处理基类，提供基本的数据集读取功能"""
    def __init__(self, args):
        """
        初始化数据集处理器
        
        参数:
        - root_dir: 数据集根目录
        - output_dir: 输出目录
        """
        self.root_dir = args.root_dir
        self.output_dir = args.output_dir
        self.logger = get_logger()
        
        # 创建输出目录
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 设置文件路径
        self.dataset_name = args.dataset_name
        self.full_data_csv = args.full_data_path + ".csv"
        self.full_data_yaml = args.full_data_path + ".yaml"
        
        # 可能的子集名称
        self.subset_name = None
    
    def read_dataset(self, class_depth=1, class_pattern=None):
        """
        读取数据集，扫描全部图像并提取类别信息
        
        参数:
        - class_depth: 类别所在的目录层级（从0开始）
        - class_pattern: 用于从路径中提取类别的正则表达式
        
        返回:
        - 数据集信息字典和类别-图像映射
        """

        # 按类别组织图像
        class_to_images = defaultdict(list)
        total_images = 0
        
        self.logger.info(f"开始扫描数据集: {self.root_dir}")
        
        # 使用os.walk遍历数据集
        for dirpath, _, filenames in os.walk(self.root_dir):
            # 获取相对路径
            rel_path = os.path.relpath(dirpath, self.root_dir)
            
            # 处理图像文件
            for filename in filenames:
                if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff')):
                    img_path = os.path.join(dirpath, filename)                                            # 获取图像完整路径
                    rel_img_path = os.path.relpath(img_path, self.root_dir)                               # 获取相对于根目录的路径
                    class_name = self._extract_class_from_path(rel_img_path, class_depth, class_pattern)  # 提取类别
                    
                    if class_name:
                        class_to_images[class_name].append(rel_img_path)
                        total_images += 1
        
        # 为类别分配标签（从0开始）
        class_to_idx = {class_name: idx for idx, class_name in enumerate(sorted(class_to_images.keys()))}

        # 准备数据集信息
        dataset_info = {
            'created_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'class_depth': class_depth,
            'class_pattern': class_pattern,
            'data': self.dataset_name,
            'path': self.root_dir,
            'total_images': total_images,
            'num_classes': len(class_to_images),
            'names': {value: key for key, value in class_to_idx.items()},
            'counts': {key: len(images) for key, images in class_to_images.items()},
        }

        self.logger.info(f"数据集扫描完成: 共有 {len(class_to_images)} 个类别, {total_images} 张图像")
        return dataset_info, class_to_images, class_to_idx
    
    def _extract_class_from_path(self, rel_path, class_depth=1, class_pattern=None):
        """
        从路径中提取类别
        
        参数:
        - rel_path: 相对路径
        - class_depth: 类别所在的目录层级（从0开始）
        - class_pattern: 用于从路径中提取类别的正则表达式
        
        返回:
        - 类别名称
        """
        if class_pattern:
            # 使用正则表达式提取类别
            import re
            match = re.search(class_pattern, rel_path)
            if match:
                return match.group(1)
        
        # 基于目录深度提取类别
        parts = rel_path.split(os.sep)
        if len(parts) > class_depth:
            return parts[class_depth]
        
        return None
    
    def generate_full_dataset(self, class_depth=1, class_pattern=None):
        """
        生成完整数据集文件和信息文件
        
        参数:
        - class_depth: 类别所在的目录层级（从0开始）
        - class_pattern: 用于从路径中提取类别的正则表达式
        
        返回:
        - CSV文件路径和YAML文件路径的元组
        """
        # 读取数据集
        dataset_info, class_to_images, class_to_idx = self.read_dataset(class_depth, class_pattern)
        
        # 准备数据列表
        data_list = []
        for class_name, images in class_to_images.items():
            class_idx = class_to_idx[class_name]
            for rel_img_path in images:
                data_list.append((rel_img_path, class_idx))
        
        # 写入CSV文件
        write_csv_file(self.full_data_csv, data_list)
        
        # 写入YAML文件
        write_yaml_file(self.full_data_yaml, dataset_info)
        
        return self.full_data_csv, self.full_data_yaml, dataset_info, class_to_images, class_to_idx
    
    def fileload(self, class_depth, class_pattern):
        """
        检查数据集是否需要重新加载
        """
        # 1. 读取 CSV 和 YAML 文件
        data_list = pd.read_csv(self.full_data_csv).values.tolist()
        with open(self.full_data_yaml, 'r', encoding='utf-8') as f:
            dataset_info = yaml.safe_load(f)

        # 2. 检查数据集是否需要重新加载
        if dataset_info['dataset_info']['root_dir'] != self.root_dir or \
           dataset_info['dataset_info']['name'] != self.dataset_name or \
           dataset_info['dataset_info']['class_depth'] != class_depth or \
           dataset_info['dataset_info']['class_pattern'] != class_pattern:
                self.logger.warning("数据集根目录或名称不匹配，可能需要重新加载数据集")
                raise FileNotFoundError

        # 3. 基于 dataset_info 获取 class_to_idx
        class_to_idx = {}
        for class_name, info in dataset_info['class_info'].items():
            class_to_idx[class_name] = info['label']

        # 4. 基于 data_list 获取 class_to_images
        class_to_images = defaultdict(list)
        for rel_img_path, label in data_list:
            class_name = list(class_to_idx.keys())[list(class_to_idx.values()).index(label)]
            class_to_images[class_name].append(rel_img_path)
        
        return dataset_info, class_to_images, class_to_idx

    def load(self, class_depth=1, class_pattern=None):
        self.logger.info("开始生成完整数据集")
        try:
            if os.path.exists(self.full_data_csv) and os.path.exists(self.full_data_yaml):
                self.logger.info("无需重新加载, 直接加载完整数据集文件")
                return self.full_data_csv, self.full_data_yaml, *self.fileload(class_depth, class_pattern)

            else:
                self.logger.info("再次读取完整数据集文件")
                return self.generate_full_dataset(class_depth, class_pattern)
            
        except Exception as e:
            self.logger.error(f"加载数据集时发生错误: {e}")
            self.logger.info("再次读取完整数据集文件")
            return self.generate_full_dataset(class_depth, class_pattern)
        finally:
            pass
        
        
