#!/usr/bin/env python3
"""
数据集处理工具主程序入口
"""

import os
import argparse
import random
import numpy as np
from datetime import datetime

from config import DEFAULT_OUTPUT_DIR, DEFAULT_SEED, DEFAULT_LOG_FILE
from utils.logger import setup_logger, get_logger
# from utils.file_utils import write_split_files
from core.processor import DatasetProcessor
from core.selector import DatasetSelector
from core.splitter import DatasetSplitter, SplitStrategy

class MyProcessor():
    def __init__(self, args):
        self.args = args
        self.args.dataset_name = os.path.basename(args.root_dir)
        self.args.full_data_path = os.path.join(args.output_dir, self.args.dataset_name)

        if isinstance(args.num_classes, int) and args.select_subset:
            note1 = f'_{str(args.num_classes)}'
        elif not args.select_subset:
            note1 = '_a'  # all
        else:
            note1 = ''    # None

        if isinstance(args.images_per_class, int) and args.split_dataset:
            note2 = f'_{str(args.images_per_class)}'
        elif not args.split_dataset:
            note2 = '_a'  # all
        else:
            note2 = ''   # None

        self.args.select_base_path = os.path.join(args.output_dir, f'{self.args.dataset_name}_select{note1}{note2}')
        self.args.split_base_path = os.path.join(args.output_dir, f'{self.args.dataset_name}_split{note1}{note2}')

    def do_process(self):
        # 处理数据集
        # 设置日志级别
        log_level = 'DEBUG' if self.args.verbose else 'INFO'
        setup_logger(log_level=log_level, log_dir=self.args.log_file)
        logger = get_logger()
        
        # 设置随机种子
        random.seed(self.args.seed)
        np.random.seed(self.args.seed)
        
        # 打印启动信息
        logger.info("=" * 50)
        logger.info(f"数据集处理工具 启动时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"数据集根目录: {self.args.root_dir}")
        logger.info(f"输出目录: {self.args.output_dir}")
        logger.info(f"随机种子: {self.args.seed}")
        logger.info("=" * 50)
        
        try:
            # 创建数据集处理器
            processor = DatasetProcessor(self.args)
            
            # 生成完整数据集
            logger.info("开始生成完整数据集")
            csv_file, yaml_file, dataset_info, class_to_images, class_to_idx = processor.generate_full_dataset(
                self.args.class_depth,
                self.args.class_pattern
            )
            logger.info(f"完整数据集生成完成: CSV={csv_file}, YAML={yaml_file}")
            
            # 处理数据集 - 选择子集
            data_to_process = None
            class_idx_mapping = None
            base_name = None
            
            if self.args.select_subset:
                logger.info("开始选择子集")
                if self.args.num_classes is None and self.args.images_per_class is None:
                    logger.warning("未指定类别数量或每类图像数量，将使用全部类别和图像")
                
                # 创建数据集选择器
                selector = DatasetSelector(self.args)
                
                # 选择子集
                subset_data, subset_info, subset_class_to_idx = selector.select_classes(
                    dataset_info,
                    class_to_images,
                    class_to_idx,
                    self.args.num_classes,
                    self.args.images_per_class
                )
                
                # 设置后续使用的数据和类别映射
                data_to_process = subset_data
                class_idx_mapping = subset_class_to_idx
                base_name = selector.subset_name
                
                # 如果不需要划分，则写入子集文件
                if not self.args.split_dataset:
                    subset_csv, subset_yaml = selector.write_subset_files(subset_data, subset_info)
                    logger.info(f"子集数据集文件生成完成: CSV={subset_csv}, YAML={subset_yaml}")
                    
                    # 如果需要复制文件
                    if self.args.copy_files:
                        # 创建数据集划分器用于复制文件
                        splitter = DatasetSplitter(self.args.output_dir, processor.dataset_name, base_name)
                        # 将子集数据转换为划分格式
                        single_split = {'subset': data_to_process}
                        copy_stats = splitter.copy_datasets(single_split, self.args.root_dir)
                        logger.info(f"子集文件复制完成: {copy_stats}")
            else:
                # 使用完整数据集
                data_to_process = []
                for class_name, images in class_to_images.items():
                    class_idx = class_to_idx[class_name]
                    for rel_img_path in images:
                        data_to_process.append((rel_img_path, class_idx))
                class_idx_mapping = class_to_idx
                base_name = processor.dataset_name
            
            # 处理数据集 - 划分数据集
            if self.args.split_dataset:
                logger.info("开始划分数据集")
                split_strategy = SplitStrategy.STRATIFIED if self.args.split_strategy == 'stratified' else SplitStrategy.RANDOM
                
                # 创建数据集划分器
                splitter = DatasetSplitter(self.args)
                
                # 划分数据集
                splits, split_ratio = splitter.split_dataset(
                    data_to_process,
                    class_idx_mapping,
                    split_strategy,
                    self.args.train_ratio,
                    self.args.val_ratio,
                    self.args.test_ratio,
                    self.args.seed,
                )
                
                # 写入划分文件
                split_files = splitter.write_split_files(splits, split_ratio, class_idx_mapping)
                
                # 打印划分结果
                logger.info(f"数据集划分完成: 训练集 {len(splits['train'])}张, 验证集 {len(splits['val'])}张, 测试集 {len(splits['test'])}张")
                for split_name, files in split_files.items():
                    logger.info(f"{split_name}集CSV文件: {files['csv']}")
                    logger.info(f"{split_name}集YAML文件: {files['yaml']}")
                
                # 如果需要复制文件
                if self.args.copy_files:
                    logger.info("开始复制划分文件")
                    copy_stats = splitter.copy_datasets(splits, self.args.root_dir)
                    logger.info(f"划分文件复制完成: {copy_stats}")
            
            logger.info("所有处理完成")
        
        except Exception as e:
            logger.error(f"处理过程中发生错误: {str(e)}", exc_info=True)
            raise
        
        finally:
            logger.info("=" * 50)
            logger.info(f"数据集处理工具 结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            logger.info("=" * 50)


def main():
    """主函数，处理命令行参数并执行相应操作"""
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='数据集处理工具')
    # 基本参数
    parser.add_argument('--root_dir', type=str, default='/data/data_wll/AMU-Tuning-main/data/vggface2_224', help='数据集根目录')
    parser.add_argument('--output_dir', type=str, default="./output", help='输出目录')
    parser.add_argument('--class_depth', type=int, default=0, help='类别所在的目录层级（从0开始）')
    parser.add_argument('--class_pattern', type=str, default=None, help='用于从路径中提取类别的正则表达式')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    parser.add_argument('--log_file', type=str, default="/logs", help='日志文件路径')
    parser.add_argument('--verbose', action='store_true', help='显示详细日志')
    
    # 子集选择参数
    parser.add_argument('--select_subset', type=bool, default=True, help='是否选择子集')
    parser.add_argument('--num_classes', type=int, default=1000, help='要选择的类别数量')
    parser.add_argument('--images_per_class', type=int, default=100, help='每个类别要选择的图像数量')
    
    # 数据集划分参数
    parser.add_argument('--split_dataset', type=bool, default=True, help='是否划分数据集')
    parser.add_argument('--split_strategy', type=str, default='stratified', 
                        choices=['random', 'stratified'], help='划分策略')
    parser.add_argument('--train_ratio', type=float, default=0.8, help='训练集比例')
    parser.add_argument('--val_ratio', type=float, default=0.2, help='验证集比例')
    parser.add_argument('--test_ratio', type=float, default=0, help='测试集比例')
    
    # 文件复制参数
    parser.add_argument('--copy_files', action='store_true', help='是否复制文件到划分目录')
    
    args = parser.parse_args()

    p = MyProcessor(args)
    result = p.do_process()

if __name__ == "__main__":
    result = main()
    # 如果需要在脚本中使用返回的结果，可以在这里处理