import os
import json
import numpy as np
from PIL import Image
import shutil
from tqdm import tqdm
import cv2 as cv

# 加载JSON文件，假设JSON文件结构类似于：
# [
#     {"path": ["/path/to/image1.jpg", "/path/to/image2.jpg"], "class":0, "className":dirName0},
#     {"path": ["/path/to/image3.jpg", "/path/to/image4.jpg"], "class":1, "className":dirName1}
#       ...
# ]

def load_and_save_images(json_path, output_dir):
    # 读取 JSON 文件
    with open(json_path, 'r') as f:
        data = json.load(f)

    assert isinstance(data, list), 'Data type wrong in transform Npy'
    i = 0
    # 遍历每一类
    for item in tqdm(data):
        classname = item['classname']
        assert len(item['path'])>0, f'maybe cant read {classname}'

        # 为每个类别创建对应的目录
        perClassDir = os.path.join(output_dir, classname)
        if not os.path.exists(perClassDir):
            # 如果没有的话，为每个类别创建对应的目录
            os.makedirs(perClassDir)
        else:
            # 如果存在这个目录，那么检查npy文件和照片文件数量是否一致
            if len(os.listdir(perClassDir)) == len(item['path']):
                continue
            else:
                # 清空文件夹内容
                shutil.rmtree(f"{perClassDir}")
                os.makedirs(perClassDir)
                # for filename in os.listdir(perClassDir):
                #     file_path = os.path.join(perClassDir, filename)
                #     if os.path.isfile(file_path) or os.path.islink(file_path):
                #         os.unlink(file_path)
                #     elif os.path.isdir(file_path):
                #         shutil.rmtree(file_path)

        # 遍历每个图像路径
        for imgPath in item['path']:
            try:
                # 加载图像
                # PIL加载
                # img = Image.open(imgPath)
                # img = np.array(img)  # 将图像转为 numpy 数组
                # CV 加载
                img = cv.imread(imgPath)  # 使用 OpenCV 读取图像
                img = cv.cvtColor(img, cv.COLOR_BGR2RGB)  # 将 BGR 转为 RGB 格式
                img = np.array(img)

                # 图像的名称可以从路径中提取，确保与原结构一致
                imgName = os.path.basename(imgPath).split('.')[0] + '.npy'
                img_output_path = os.path.join(perClassDir, imgName)

                # 保存为 npy 文件
                np.save(img_output_path, img)
                i = i+1
                #print(f"Saved: {img_output_path}")

            except Exception as e:
                print(f"Error processing {imgPath}: {e}")

    print(f'total {i} npyFile')
# 设置 JSON 文件路径和输出目录
json_path = '/data/data_wll/AMU-Tuning-main/dataJson/vggface2_224.json'
output_dir = '/data/data_wll/AMU-Tuning-main/dataNpy/vggface2_224'

# 加载并保存图像
load_and_save_images(json_path, output_dir)
