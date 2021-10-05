from torch import nn
import torch
import torch.nn.functional as F
from Parament import nclass


class lstmModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        cnn = nn.Sequential()
        cnn.add_module("conv0",
                       nn.Conv2d(in_channels=3, out_channels=32, kernel_size=(1, 5), stride=(1, 2), padding=(0, 2)))
        cnn.add_module('batch0', nn.BatchNorm2d(32))
        cnn.add_module('relu0', nn.ReLU(True))
        cnn.add_module("conv1",
                       nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(1, 3), stride=(1, 2), padding=(0, 2)))
        cnn.add_module('batch1', nn.BatchNorm2d(64))
        cnn.add_module('relu1', nn.ReLU(True))
        self.cnn = cnn
        self.lstm = nn.LSTM(input_size=64, hidden_size=128, bidirectional=True, num_layers=2, batch_first=True)
        self.linear = nn.Linear(256, nclass)

    def forward(self, input):
        b, h, w = input.size()
        input = input.permute(0, 2, 1)
        input = input.unsqueeze(2)
        output = self.cnn(input)
        output = output.reshape(b, -1)
        return output

        # out = self.linear(out)
        # return out
        # return F.log_softmax(out, 1)
