import math

import torch
from tensorboardX import SummaryWriter

from torch import optim
from torch.utils.data.dataloader import DataLoader
import torch.nn as nn

import numpy as np

from Model.gyroModel import lstmModel
from Model.mergeModel import mergeModel
from process.dataProcess import dataProcess
from Model.ModelCir import  CirModel


import torch.nn.utils.rnn as rnn_utils

from Parament import confusionShowStep, epoch, batchSize, nclass, splitRatio, lrAttention, lrMain



def weight_init(m):
    if isinstance(m, nn.Conv2d):
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()

def confusionDraw(confusionArray):
    for i in range(0, nclass):
        #       for j in range(0,3):
        print(str(confusionArray[i][0]) + ',' + str(confusionArray[i][1]) + ',' + str(confusionArray[i][2])+ ','
              + str(confusionArray[i][3]) + ',' + str(confusionArray[i][4]) + ',' +str(confusionArray[i][5]))


def train(model, criteria, epoch, train_loader, device, optimizer, test_load):
    model = model.train()
    model = model.to(device)
    drawFlag = [False]
    # f = open("../Performance/train_loss6-20.txt",'w')
    # f1 = open("../Performance/acc_6-20.txt",'w')
    for i in range(epoch):
        for i_batch, (data, dataGyro, label) in enumerate(train_loader):
            data = data.to(device)
            dataGyro = dataGyro.to(device)
            label = label.to(device)
            pred = model(data,dataGyro)
            cost = criteria(pred, label)
            model.zero_grad()
            # print(model.weight.is_leaf)
            # model.weight.retain_grad()
            cost.backward()
 #           if (i_batch % 10 == 0):
 #              print(model.test)
            optimizer.step()
            if (i_batch % 20 == 0):
                print('Train Epoch: {}\tLoss: {:.6f}'.format(i, cost.item()))
                # lossError = cost.item()
                # f.write(str(lossError))
                # f.write("\n")


        # if i == 10:
        #     torch.save(model,'model3.pkl')
        if (i+1)%5 == 0:
            valLoss, valAcc = test(test_load, device, model, criteria, drawFlag, i,cost.item())
            print('val error:{:.6f}\tval accuray:{:.6f}'.format(valLoss, valAcc))
            print('\n')
            # f1.write(str(valAcc))
            # f1.write("\n")

            for p in model.parameters():
                p.requires_grad = True
            model.train()

        # if (i+1)%50 == 0:
        #     f.close()
        #     f1.close()

def test(dataSet, device, model, loss, drawFlag, epoch, trainError):
    for p in model.parameters():
        p.requires_grad = False
    model.eval()
    confustionArray = np.zeros([nclass, nclass])

    lossAge = 0.0
    correct = 0.0

    for i_batch, (img, gyro,name) in enumerate(dataSet):
        img = img.to(device)
        name = name.to(device)
        gyro = gyro.to(device)
        pred = model(img,gyro)
        error = loss(pred, name)
        lossAge += error.item()
        # zero = torch.zeros_like(name)
        # one = torch.ones_like(name)
        _, predEnd = torch.max(pred.data, 1)
        confustionArray[name, predEnd] += 1
        correct += (predEnd == name).sum()
        # correct += torch.where(abs(pred-name)<0.5,one,zero).sum()

    lossAge /= len(dataSet.dataset)
    lens = len(dataSet.dataset)
    correct = correct / lens
    # writer = SummaryWriter('runs/scalar_example/6-20')
    # writer.add_scalar('Train', trainError, epoch)
    # writer.add_scalar('test', correct , epoch)
    if (epoch+1)%confusionShowStep == 0:
        confusionDraw(confustionArray)
        drawFlag[0] = True
    return lossAge, correct





if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # device = torch.device('cpu')
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    dataSet = dataProcess()
    train_size = int(splitRatio * len(dataSet))
    test_size = len(dataSet) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataSet, [train_size, test_size])
    dataLoad = DataLoader(train_dataset, batch_size=batchSize, num_workers=16, shuffle=True,pin_memory=True)
    testLoad = DataLoader(test_dataset, batch_size=1,num_workers=1)
    #model = cnn()
    #model = MyModel(2,128,1)
    modelAudio = CirModel()
    modelGyro = lstmModel()
    model = mergeModel(modelAudio,modelGyro)


    #model.apply(weight_init)
    for name, parameters in model.named_parameters():
        print(name, ':', parameters.size())

    # attentionParam = list(map(id,model.attention.parameters()))
    # mainParams = filter(lambda p: id(p) not in attentionParam,
    #                         model.parameters())

    criteria = torch.nn.CrossEntropyLoss(reduction='mean')
    #optimizer = optim.Adam([{'params':model.attention.parameters(), 'lr':lrAttention},{'params':mainParams,'lr':lrMain}])
    optimizer = optim.Adam(model.parameters(), lr=lrMain)
    train(model, criteria, epoch, dataLoad, device, optimizer, testLoad)
