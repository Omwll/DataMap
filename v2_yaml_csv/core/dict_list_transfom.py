
from collections import defaultdict

def l2d_idx2path(data_list):
    """
    将列表转换为字典
    transform list to dict, with the form like: 
        - dict(class_idx: [rel_img_path1, rel_img_path2, ...])

    参数:
    - data_list: 数据列表，格式为[(rel_img_path, class_idx), ...]
    
    返回:
    - idx2path: 类别到文件路径的映射
    """
    # 准备数据字典
    idx2path = defaultdict(list)
    
    for rel_img_path, class_idx in data_list:
        idx2path[class_idx].append[rel_img_path]  # 添加文件路径

    return idx2path

def d2l_path_idx(class2idx, class2path):
    """
    将字典转换为列表
    transform dict to list, with the form like: 
        - list[[rel_img_path, class_idx], ...]
    
    参数:
    - data_dict: 数据字典
    
    返回:
    - 转换后的列表
    """
    # 准备数据列表
    data_list = []
    for class_name, paths in class2path.items():
        class_idx = class2idx[class_name]
        for rel_img_path in paths:
            data_list.append((rel_img_path, class_idx))
    return data_list