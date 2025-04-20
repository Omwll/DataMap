import os
import random
from utils.logger import get_logger
from utils.file_utils import write_csv_file, write_yaml_file
from datetime import datetime

logger = get_logger()

class DatasetSelector:
    """数据集子集选择器，负责从完整数据集中选择特定子集"""
    
    def __init__(self, args):
        """
        初始化数据集选择器
        
        参数:
        - root_dir: 数据集根目录
        - output_dir: 输出目录
        - dataset_name: 数据集名称，如果为None则使用根目录的basename
        """
        self.root_dir = args.root_dir
        self.output_dir = args.output_dir
        self.dataset_name = args.dataset_name
        self.select_base_path = args.select_base_path
        
        # 可能的子集名称
        self.subset_name = None
    
    def select_classes(self, dataset_info, class_to_images, class_to_idx, num_classes=None, images_per_class=None):
        """
        基于完整数据集选择类别和图像
        
        参数:
        - dataset_info: 完整数据集信息
        - class_to_images: 完整数据集的类别到图像映射
        - class_to_idx: 完整数据集的类别到标签映射
        - num_classes: 要选择的类别数量，如果为None则选择所有类别
        - images_per_class: 每个类别要选择的图像数量，如果为None则选择所有图像
        
        返回:
        - 选择的数据、子集信息字典、新的类别到标签映射字典
        """

        # 1. 选择类别 - 确保类内图像数量足够
        available_classes = []
        for class_name, class_num in dataset_info['counts'].items():
            # 对类内文件数量不做要求，或只选择类内文件数量足够的类别
            if images_per_class is None or class_num >= images_per_class:
                available_classes.append(class_name)
            else:
                logger.warning(f"类别 '{class_name}' 只有 {class_num} 张图像，少于要求的 {images_per_class} 张，将被排除")
        
        if not available_classes:
            raise ValueError("没有类别包含足够数量的图像，请减少每类图像数量或使用更大的数据集")
        
        if num_classes is not None and num_classes < len(available_classes):
            selected_classes = random.sample(available_classes, num_classes)
        else:
            selected_classes = available_classes
            num_classes = len(available_classes)
        
        # 为选定的类别分配新标签（从0开始）
        new_class_to_idx = {class_name: idx for idx, class_name in enumerate(sorted(selected_classes))}

        # 2. 选择图像并准备写入
        selected_data = []
        total_selected_images = 0
        
        # 为每个类别选择图像
        for class_name in sorted(selected_classes):
            images = class_to_images[class_name]
            new_label = new_class_to_idx[class_name]
            
            # 选择图像
            if images_per_class is not None and images_per_class < len(images):
                selected = random.sample(images, images_per_class)
            else:
                selected = images
            
            # 添加到选择的数据中
            for img_path in selected:
                selected_data.append((img_path, new_label))
            
            total_selected_images += len(selected)

        # 准备子集数据集信息
        subset_info = {
            'created_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'data': os.path.basename(self.select_base_path),
            'path': self.root_dir,
            'total_images': total_selected_images,
            'num_classes': len(selected_classes),
            'names': {value: key for key, value in new_class_to_idx.items()},
            'counts': None,
            'mapping': None,
        }
        
        # 添加类别信息和映射关系
        counts_dict = {}
        label_mapping = {}
        for class_name in selected_classes:
            old_label = class_to_idx[class_name]
            new_label = new_class_to_idx[class_name]
            
            # 计算该类被选中的图像数量
            selected_count = sum(1 for _, label in selected_data if label == new_label)
            counts_dict[class_name] = selected_count
            # subset_info['class_info'][class_name] = {
            #     'label': new_label,
            #     'count': selected_count,
            #     'original_label': old_label
            # }
            
            label_mapping[class_name] = f'{old_label} -> {new_label}'
        
        # 添加标签映射信息
        subset_info['mapping'] = label_mapping
        logger.info(f"子集选择完成: 选择了 {len(selected_classes)} 个类别, 共 {total_selected_images} 张图像")
        
        return selected_data, subset_info, new_class_to_idx
    
    def write_subset_files(self, selected_data, subset_info):
        """
        将子集数据写入文件
        
        参数:
        - selected_data: 选择的数据列表
        - subset_info: 子集信息字典
        
        返回:
        - 子集CSV和YAML文件路径
        """
        # 设置文件路径
        subset_csv = self.select_base_path + ".csv"
        subset_yaml = self.select_base_path + ".yaml"
        
        write_csv_file(subset_csv, selected_data)  # 写入CSV文件
        write_yaml_file(subset_yaml, subset_info)  # 写入YAML文件
        
        return subset_csv, subset_yaml