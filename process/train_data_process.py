import torch
from torch.utils.data import Dataset, DataLoader
import Parament

import scipy.io as scio
import numpy as np
import re


class trainDataProcess(Dataset):

    def __getitem__(self, index: int):
        path, label = self.datas[index]
        if self.pathDic[path] == 2:
            pathGyro = self.getGyroFromAudio(path)
            self.pathDic[path] = 1
        else:
            randomIndex = np.random.randint(len(self.pathGroup[label]))
            pathGyro = self.getGyroFromAudio(self.pathGroup[label]
                                             [randomIndex])
            del self.pathGroup[label][randomIndex]
        dataGyro = scio.loadmat(pathGyro)
        dataGyro = np.array(dataGyro['dataGyro'])
        dataGyro = torch.FloatTensor(dataGyro)

        data = scio.loadmat(path)

        data = np.array(data['x'])
        data = torch.FloatTensor(data)

        return data, dataGyro, label

    def __len__(self) -> int:
        return len(self.datas)

    def __init__(self, list) -> None:
        super().__init__()
        # path = Parament.dataPath
        # f = open(path, 'r')
        list.extend(list)
        datas = []
        for line in list:
            toChange = line.split('-')
            label = toChange[1]
            if label[0:-1] == 'push':
                label = 0
            elif label[0:-1] == 'double click':
                label = 1
            elif label[0:-1] == 'pull':
                label = 2
            elif label[0:-1] == 'rotation':
                label = 3
            elif label[0:-1] == 'grab':
                label = 4
            elif label[0:-1] == 'release':
                label = 5
            elif label[0:-1] == 'upDown':
                label = 6
            datas.append((toChange[0], label))
        self.pathDic = {}
        self.pathGroup = [[] for i in range(Parament.nclass)]
        for (path, label) in datas:
            self.pathDic[path] = 2
            self.pathGroup[label].append(path)
        self.datas = datas

    def getGyroFromAudio(self, path):
        fatherPath = path[0:path.rfind('/') + 1]
        time = re.findall(r'\d+', path)[0]
        gyroPath = fatherPath + time + "gyro" + ".mat"
        return gyroPath
