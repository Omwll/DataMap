
import os
import random
import numpy as np
from enum import Enum
from collections import defaultdict
from datetime import datetime
from sklearn.model_selection import train_test_split
from utils.logger import get_logger
from utils.file_utils import write_csv_file, write_yaml_file

logger = get_logger()

class SplitStrategy(Enum):
    """数据集划分策略枚举类"""
    RANDOM = "random"          # 随机划分
    STRATIFIED = "stratified"  # 分层划分 (保持每个类别的比例)

class DatasetSplitter:
    """数据集划分器，负责将数据集划分为训练集、验证集和测试集"""
    
    def __init__(self, args):
        """
        初始化数据集划分器
        
        参数:
        - output_dir: 输出目录
        """
        self.root_dir = args.root_dir
        self.output_dir = args.output_dir
        self.split_base_path = args.split_base_path

    def split_dataset(self, data_list, class_to_idx=None, strategy=SplitStrategy.STRATIFIED, 
                      train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=42):
        """
        划分数据集为训练集、验证集和测试集
        
        参数:
        - data_list: 数据列表，格式为[(相对路径, 标签), ...]
        - class_to_idx: 类别到标签的映射字典
        - strategy: 划分策略，可选RANDOM或STRATIFIED
        - train_ratio: 训练集比例
        - val_ratio: 验证集比例
        - test_ratio: 测试集比例
        - seed: 随机种子
        
        返回:
        - 划分后的数据集字典和划分比例
        """
        # 设置随机种子
        random.seed(seed)
        np.random.seed(seed)
        
        # 检查比例和是否为1
        if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-10:
            logger.warning(f"划分比例之和不为1: {train_ratio + val_ratio + test_ratio}")
            # 归一化比例
            total = train_ratio + val_ratio + test_ratio
            train_ratio /= total
            val_ratio /= total
            test_ratio /= total
        
        split_ratio = {'train': train_ratio, 'val': val_ratio, 'test': test_ratio}
        
        # 初始化分割结果
        splits = {
            'train': [],
            'val': [],
            'test': []
        }
        
        # 执行划分
        if strategy == SplitStrategy.RANDOM:
            # 随机划分 - 使用train_val_test_split函数
            if not data_list:
                logger.warning("数据列表为空，无法进行划分")
                return splits, split_ratio
            
            # 使用train_val_test_split函数进行随机划分
            train_data, val_data, test_data = self._train_val_test_split(
                data_list, 
                split_ratio=split_ratio, 
                random_state=seed, 
                shuffle=True
            )
            
            splits['train'] = train_data
            splits['val'] = val_data
            splits['test'] = test_data
            
        elif strategy == SplitStrategy.STRATIFIED:
            # 分层划分 - 按类别保持比例
            if not class_to_idx:
                logger.warning("未提供类别到标签的映射，无法进行分层划分，将使用随机划分")
                return self.split_dataset(data_list, None, SplitStrategy.RANDOM, 
                                        train_ratio, val_ratio, test_ratio, seed)
            
            # 按类别组织数据
            class_data = defaultdict(list)
            for img_path, label in data_list:
                class_data[label].append((img_path, label))
            
            # 对每个类别进行划分
            for label, class_images in class_data.items():
                # 使用train_val_test_split函数对每个类别进行划分
                train_data, val_data, test_data = self._train_val_test_split(
                    class_images, 
                    split_ratio=split_ratio, 
                    random_state=seed, 
                    shuffle=True
                )
                
                # 合并结果
                splits['train'].extend(train_data)
                splits['val'].extend(val_data)
                splits['test'].extend(test_data)
        
        else:
            raise ValueError(f"不支持的划分策略: {strategy}")
        
        # 记录每个集合的图像数量
        split_counts = {split: len(images) for split, images in splits.items()}
        logger.info(f"数据集划分完成: 训练集 {split_counts['train']}张, 验证集 {split_counts['val']}张, 测试集 {split_counts['test']}张")
        
        return splits, split_ratio
    
    def _train_val_test_split(self,
                              data,
                              split_ratio={'train': 0.7, 'val': 0.15, 'test': 0.15}, 
                              random_state=None, 
                              shuffle=True):
        """
        根据提供的比例字典将数据分割为训练集、验证集和测试集
        
        参数:
        - data: 需要分割的数据，可以是列表、numpy数组等
        - split_ratio: 字典，包含'train'、'val'和'test'的比例，总和应为1
        - random_state: 随机种子，保证结果可重复
        - shuffle: 是否打乱数据顺序
        
        返回:
        - 根据指定比例分割后的数据集，顺序为train, val, test
        - 如果某个集合的比例为0，则返回空列表或数组
        """
        # 检查输入比例是否合法
        total_ratio = sum(split_ratio.values())
        if not np.isclose(total_ratio, 1.0):
            raise ValueError(f"分割比例之和应为1.0，当前总和为 {total_ratio}")
        
        # data = np.array(data)

        # 先初始化所有结果集为空列表
        result = {'train': [], 'val': [], 'test': []}
        
        # 计算非零比例的集合
        non_zero_sets = [k for k, v in split_ratio.items() if v > 0]
        num_non_zero = len(non_zero_sets)
        
        # 只有一个集合有非零比例
        if num_non_zero == 1:
            set_name = non_zero_sets[0]
            result[set_name] = data
        
        # 有两个集合有非零比例
        elif num_non_zero == 2:
            set1, set2 = non_zero_sets
            ratio1 = split_ratio[set1] / (split_ratio[set1] + split_ratio[set2])
            
            # 使用train_test_split分割
            result[set1], result[set2] = train_test_split(data, train_size=ratio1, random_state=random_state, shuffle=shuffle)

        # 正常情况：三个集合都有非零比例
        else:
            # 随机打乱数据
            if shuffle:
                random.seed(random_state)
                indices = list(range(len(data)))
                random.shuffle(indices)
                data = [data[i] for i in indices] 
            
            # 计算各集合对应的样本数量
            total_samples = len(data)
            train_size = int(total_samples * split_ratio['train'])
            val_size = int(total_samples * split_ratio['val'])
            # 确保三个集合的总样本数等于总数据量
            test_size = total_samples - train_size - val_size
            
            # 直接按照样本数量进行切分
            start_idx = 0
            end_idx = train_size
            result['train'] = data[start_idx:end_idx]
            
            start_idx = end_idx
            end_idx = start_idx + val_size
            result['val'] = data[start_idx:end_idx]
            
            start_idx = end_idx
            result['test'] = data[start_idx:]

        return result['train'], result['val'], result['test']
    
    def write_split_files(self, splits, split_ratio, class_to_idx):
        """
        将划分后的数据集写入文件
        
        参数:
        - splits: 划分后的数据集字典，格式为{'train': [...], 'val': [...], 'test': [...]}
        - split_ratio: 划分比例字典，格式为{'train': 0.7, 'val': 0.15, 'test': 0.15}
        - class_to_idx: 类别到标签的映射字典
        - root_dir: 数据集根目录，用于YAML信息
        
        返回:
        - 包含各分割文件路径的字典
        """
        # 生成划分后的CSV和YAML文件


        split_files = {}


        # 准备YAML信息
        split_info = {
            'created_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'path': self.root_dir if self.root_dir else "",
            'split_ratio': split_ratio,
            'nc': len(class_to_idx), 
            'train': None,
            'val': None,
            'test': None,
            'name': {value: key for key, value in class_to_idx.items()},
        }

        #class_counts = defaultdict(int)
        for split_name, split_data in splits.items():
            if not split_data:
                continue
            # 创建CSV文件, 并写入
            csv_path = f"{self.split_base_path}_{split_name}_{''.join(str(split_ratio[split_name]).split('.'))}.csv"
            split_info[split_name] = csv_path
            write_csv_file(csv_path, split_data)

            # for _, label in split_data:
            #     class_counts[label] += 1

        # 写入YAML文件
        yaml_path = f"{self.split_base_path}_split.yaml"
        write_yaml_file(yaml_path, split_info)

        # 保存文件路径
        split_files[split_name] = {
            'csv': csv_path,
            'yaml': yaml_path
        }
        
        return split_files