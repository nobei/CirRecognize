from torch.utils.data import Dataset
import re
import scipy.io as scio
import numpy as np
import torch


class cirDataLoad(Dataset):
    def __len__(self) -> int:
        return len(self.datas)

    def __getitem__(self, index):
        path, label = self.datas[index]
        dataCir = scio.loadmat(path)
        dataCir = np.array(dataCir['x'])
        dataCir = torch.FloatTensor(dataCir)

        return dataCir, label

    def __init__(self, list):
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

