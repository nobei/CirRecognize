from typing import Any

from torch import nn
import torch.nn.functional as F
from Parament import nclass


class CirModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        cnn = nn.Sequential()
        # cnn.add_module('myAttention0',myAttention())
        cnn.add_module('conv0',
                       nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(5, 5), stride=(2, 1), padding=(2, 2)))
        cnn.add_module('batch0', nn.BatchNorm2d(32))
        cnn.add_module('relu0', nn.ReLU(True))
        cnn.add_module('pool0',
                       nn.MaxPool2d(kernel_size=(2, 2), stride=2))
        cnn.add_module('conv1',
                       nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), padding=(1, 1)))
        cnn.add_module('batch1', nn.BatchNorm2d(64))
        cnn.add_module('relu1', nn.ReLU(True))
        cnn.add_module('pool1',
                       nn.MaxPool2d(kernel_size=(2, 2), stride=2))

        self.cnn = cnn

    def forward(self, input):
        b, h, w = input.size()
        input = input.unsqueeze(1)
        output = self.cnn(input)
        output = output.reshape(b, -1)
        return output
        # outputClass = self.line(output)
        # return F.log_softmax(outputClass, 1)
