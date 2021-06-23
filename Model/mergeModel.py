from torch import nn
from Parament import nclass
import torch.nn.functional as F
import torch


class mergeModel(nn.Module):
    def __init__(self,audio,gyro) -> None:
        super().__init__()
        self.audio = audio
        self.gyro = gyro
        linear = nn.Sequential()
        linear.add_module('linear0',nn.Linear(49088,512))
        linear.add_module('linear1',nn.Linear(512,nclass))
        self.linear = linear





    def forward(self, inputAudio,inputGyro):
        audioOut = self.audio(inputAudio)
        gyroOut = self.gyro(inputGyro)
        mergeData = torch.cat([audioOut,gyroOut],1)
        out = self.linear(mergeData)
        return F.log_softmax(out,1)
        # sumOut = audioOut+gyroOut
        # return F.log_softmax(sumOut,1)

