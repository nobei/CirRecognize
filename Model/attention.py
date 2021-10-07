from typing import Any

import torch
import torch.nn.functional as F
from torch import nn


class attention(nn.Module):

    def __init__(self) -> None:
        super().__init__()
        self.atten = torch.nn.init.normal_(nn.Parameter(torch.FloatTensor(128 * 2), requires_grad=True))
        self.tanh1 = nn.Tanh()

    def forward(self, input):
        M = self.tanh1(input)
        alpha = F.softmax(torch.matmul(M, self.atten), dim=1).unsqueeze(-1)
        output = input * alpha
        out = torch.sum(output, 1)
        out = F.relu(out)
        return out
