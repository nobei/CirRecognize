import torch
from torch.utils.data import Dataset
import scipy.io as scio
import numpy as np
import re


class testData(Dataset):
    def __len__(self) -> int:
        return len(self.datas)

    def __init__(self, list) -> None:
        super().__init__()
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
        self.datas = datas

    def getGyroFromAudio(self, path):
        fatherPath = path[0:path.rfind('/') + 1]
        time = re.findall(r'\d+', path)[0]
        gyroPath = fatherPath + time + "gyro" + ".mat"
        return gyroPath

    def __getitem__(self, index):
        path, label = self.datas[index]
        path_gyro = self.getGyroFromAudio(path)
        dataGyro = scio.loadmat(path_gyro)
        dataGyro = np.array(dataGyro['dataGyro'])
        dataGyro = torch.FloatTensor(dataGyro)

        data = scio.loadmat(path)

        data = np.array(data['x'])
        data = torch.FloatTensor(data)

        return data, dataGyro, label
