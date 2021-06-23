

from torch import nn
import torch
import torch.nn.functional as F
from Parament import nclass


class lstmModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.lstm = nn.LSTM(input_size=3, hidden_size=64, bidirectional= True,num_layers=2,batch_first=True)
        self.atten = torch.nn.init.normal_(nn.Parameter(torch.FloatTensor(64 * 2),requires_grad=True))
        self.tanh1 = nn.Tanh()
        self.linear = nn.Linear(128,nclass)




    def forward(self, input):
        b,c,w = input.size()
        output,_ = self.lstm(input)

        M = self.tanh1(output)
        alpha = F.softmax(torch.matmul(M, self.atten), dim=1).unsqueeze(-1)
        output = output*alpha
        out = torch.sum(output, 1)
        out = F.relu(out)
        return out
        # out = self.linear(out)
        # return out
        # return F.log_softmax(out, 1)

