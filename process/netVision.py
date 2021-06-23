import torch
from torchvision import models
from torchviz import make_dot

from Model.ModelCir import CirModel
from Model.gyroModel import lstmModel
from Model.mergeModel import mergeModel

modelAudio = CirModel()
modelGyro = lstmModel()
model = mergeModel(modelAudio,modelGyro)
audio = torch.randn(64,361,71)
gyro = torch.rand(64,200,3)
net_plot = make_dot(modelGyro(gyro),params = dict(modelGyro.named_parameters()))
net_plot.render("net_struct1", view=False)