"""
日志配置模块 - 提供全局日志配置
"""
import logging
import os
from datetime import datetime

# 全局日志对象
logger = None

def setup_logger(log_level=logging.INFO, log_dir='logs', filename=None):
    """
    配置全局日志系统
    
    参数:
    - log_level: 日志级别，默认为INFO
    - log_dir: 日志文件目录，默认为'logs'
    - filename: 日志文件名，默认为None（使用当前日期时间）
    
    返回:
    - 配置好的logger对象
    """
    global logger
    
    # 如果logger已经配置过，直接返回
    if logger is not None:
        return logger
    
    # 创建日志目录
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    # 设置日志文件名
    if filename is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"dataset_processor_{timestamp}.log"
    
    log_file = os.path.join(log_dir, filename)
    
    # 创建logger
    logger = logging.getLogger('dataset_processor')
    logger.setLevel(log_level)
    
    # 文件处理器
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(log_level)
    
    # 控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    
    # 设置日志格式
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # 添加处理器
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    logger.info("日志系统初始化完成")
    
    return logger

def get_logger():
    """
    获取全局日志对象
    
    返回:
    - 全局logger对象
    """
    global logger
    
    # 如果logger尚未配置，使用默认配置
    if logger is None:
        logger = setup_logger()
    
    return logger