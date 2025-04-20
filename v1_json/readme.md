# `dataPreload`

1. 作用

   用于处理数据集的递归加载、数据分割和缓存保存

   - 处理数据集的递归加载，并管理图像数据集
   - 数据划分，将数据分为 `train/val/test`
   - 数据读写，保存`json`文件并读取

2. `dataPreload`函数结构

   ```
   dataPreload
   ├── __init__(dir, dataName)  
   ├── checkDir(path)			# 目录检查
   ├── findFile(path)			# 递归文件查找
   ├── SplitEveryClass()	     # 数据划分
   ├── saveData2Cache(savePath) # json数据保存
   └── readCache(savedPath)     # 缓存读取
   ```

## 数据集的递归加载

1. `checkDir()`

   - 用于判断当前文件夹是否是叶节点文件夹，并且返回当前文件夹中的所有文件或文件夹
   - 配合递归函数 `findFile()` 判断是否进行递归进行深度搜索

2. `findFile()`

   判断当前文件夹是否是叶节点文件夹

   - 若当前文件夹不是叶结点，将继续调用自己进行递归搜索
   - 若当前文件夹是叶结点，那么将一个类内的数据，包括名称、标签和路径用一个字典表示

## 数据划分

`SplitEveryClass()`用于进行数据划分，划分方式为**分层抽样**

# `load_and_save_images()`

1. 读取`json`缓存文件，将其中所有文件都转换为`.npy`的数据形式

2. 按照类别依次读取和数据转换

   ```
   # 加载JSON文件，假设JSON文件结构类似于：
   [
       {"path": ["/path/to/image1.jpg", "/path/to/image2.jpg"], "class":0, "className":dirName0},
       {"path": ["/path/to/image3.jpg", "/path/to/image4.jpg"], "class":1, "className":dirName1}
         ...
   ]
   ```

   