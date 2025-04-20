import os
import json
import numpy
# dir = os.path.join(datadir, dirname)

class dataPreload():
    def __init__(self, dir, dataName):
        super().__init__()
        self.data = []
        self.dataSplit = {}  # if split data into 3 parts, it's not none
        self.classnum = 0

        self.dataName = dataName
        self.dirpath = os.path.join(dir, dataName)
        assert os.path.isdir(dir), f'There is no {dir}'

    def checkDir(self, path):
        isAllFile = True
        subFilepath = []
        for i in os.listdir(path):
            subdir = os.path.join(path, i)

            # 1. whether dir or file, so just add it 
            subFilepath.append(subdir)

            # 2. if there has a dir in the subdir, just false the isAllFile
            if os.path.isdir(os.path.join(path, i)):
                # print(f'{subdir} is a dir')
                isAllFile = False

        # print('check end')
        return isAllFile, sorted(subFilepath)

    def findFile(self, path):
        # 1. 查看子文件夹是文件夹还是文件
        isAllFile, subFilepath = self.checkDir(path)
        subFilepath = subFilepath[:100]  # maybe make pathnums smalller to Debug

        # 2. 否则遍历当前子文件夹并查看所有子文件是否都是文件，不是的话就报错返回
        if isAllFile:
            # 1. check all items in subFilepath is file
            assert all([os.path.isfile(i) for i in subFilepath]), 'Maybe there are files and dirs in the subdir'

            # 2. must check the classname for the specifical datset
            # if your dataset like the omnight, must change the classname
            classname = subFilepath[0].split('/')[-2]

            # 3. save the filepaths with their name and class
            oneFiledata = {'path':subFilepath, 'class':f'{self.classnum}', 'classname':classname}
            self.classnum += 1
            self.data.append(oneFiledata)
            # for i in oneFiledata.values():  # if you want, you can open it to check
            #     print(i)

        # 3. 如果子文件夹还是文件夹那么就递归进入
        else:
            while(len(subFilepath)):
                self.findFile(subFilepath.pop(0))

    def SplitEveryClass(self):
        # split the data into trainset, valset, testset
        trainset = []
        valset = []
        testset = []
        for singleClass in self.data:
            imagePath = singleClass['path']
            imageClass = singleClass['class']
            imageClassName = singleClass['classname']

            imageLen = len(imagePath)

            # make data in ever class
            for i in range(len(imagePath)):
                temp = [imagePath[i], imageClass, imageClassName]

                if i < 0.7*imageLen:
                    trainset.append(temp)
                elif i < 0.9*imageLen:
                    valset.append(temp)
                else:
                    testset.append(temp)

        self.dataSplit = {'train':trainset, 'val':valset, 'test':testset}

    def saveData2Cache(self, savePath):
        # self.findFile(self.dirpath)
        with open(savePath, 'w') as file:
            if self.dataSplit:
                json.dump(self.dataSplit, file)
            else:
                assert len(self.data), 'There is no data to save'
                json.dump(self.data, file)

    # know about the data
    @staticmethod
    def readCache(savedPath):
        with open(savedPath, 'r') as file:
            data = json.load(file)

        if isinstance(data, list):
            for i in data:
                for k, v in i.items():
                    print(f'{k}: ', len(v) if isinstance(v, list) else v)
        elif isinstance(data, dict):
            for k, v in data.items():
                print(f'{k}: ', len(v) if isinstance(v, list) else v)

dir = '/data/data_wll/AMU-Tuning-main/dataJson'
dataName = 'vggface2_224'
savePath = os.path.join(dir, f'{dataName}.json')

dPreloader = dataPreload(dir, dataName)
# dPreloader.findFile(dPreloader.dirpath)
# dPreloader.SplitEveryClass()
# dPreloader.saveData2Cache(savePath)
dataPreload.readCache(savePath)