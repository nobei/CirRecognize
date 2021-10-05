from typing import Any

from torch import nn

from Model.ModelCir import CirModel
from Model.attention import attention
from Model.gyroModel import lstmModel
from Parament import nclass
import torch.nn.functional as F
import torch


class mergeModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.audio = CirModel()
        self.gyro = lstmModel()
        self.attention = attention()
        linear = nn.Sequential()
        linear.add_module('linear0', nn.Linear(52224, 512))
        linear.add_module('linear1', nn.Linear(512, nclass))
        self.linear = linear

    def forward(self, inputAudio, inputGyro):
        audioOut = self.audio(inputAudio)
        gyroOut = self.gyro(inputGyro)
        # gyroOutAttention = self.attention(gyroOut)
        mergeData = torch.cat([audioOut, gyroOut], 1)
        out = self.linear(mergeData)
        return F.log_softmax(out, 1)
        # sumOut = audioOut+gyroOut
        # return F.log_softmax(sumOut,1)


class softVoteMerge(nn.Module):
    def __init__(self) -> None:
        super(softVoteMerge, self).__init__()
        self.audio = CirModel()
        self.gyro = lstmModel()
        self.GyroLinear = nn.Linear(3264, nclass)
        self.attention = attention()
        line = nn.Sequential()
        #        line.add_module('line1', nn.Linear(128, 128))
        line.add_module('line1', nn.Linear(48960, 512))
        line.add_module('line2', nn.Linear(512, 64))
        line.add_module('line3', nn.Linear(64, nclass))
        self.line = line

    def forward(self, inputAudio, inputGyro):
        audioOut = self.audio(inputAudio)
        gyroOut = self.gyro(inputGyro)
        # gyroOutAttention = self.attention(gyroOut)
        audioClass = self.line(audioOut)
        gyroClass = self.GyroLinear(gyroOut)
        softVote = audioClass + gyroClass
        return F.log_softmax(softVote, dim=1)


class onlyAudio(nn.Module):

    def __init__(self) -> None:
        super(onlyAudio, self).__init__()
        linear = nn.Sequential()
        linear.add_module('linear0', nn.Linear(48960, 512))
        linear.add_module('linear1', nn.Linear(512, nclass))
        self.audio = CirModel()
        self.linear = linear

    def forward(self, input):
        out = self.audio(input)
        outClass = self.linear(out)
        return F.log_softmax(outClass, dim=1)


class onlyGyro(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.gyro = lstmModel()
        self.linear = nn.Linear(3264, nclass)
        self.attention = attention()

    def forward(self, input):
        out = self.gyro(input)
        b, c, h, w = out.size()
        # out = self.attention(out)
        out = out.reshape(b, -1)
        outClass = self.linear(out)
        return F.log_softmax(outClass, dim=1)
