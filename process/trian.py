import math
import random

import torch

from torch import optim
from torch.utils.data.dataloader import DataLoader
import torch.nn as nn

import numpy as np

from Model.mergeModel import mergeModel, softVoteMerge, onlyAudio, onlyGyro
from process.test_data_process import testData
from process.train_data_process import trainDataProcess
from Parament import dataPath

from Parament import confusionShowStep, epoch, batchSize, nclass, splitRatio, lrAttention, lrMain


def random_splite(radio):
    path = dataPath
    f = open(path, 'r')
    paths = []
    for line in f.readlines():
        paths.append(line)
    train_sample = random.sample(paths, int(radio * len(paths)))
    test_sample = list(set(paths) - set(train_sample))
    label1 = 0
    label2 = 0
    label3 = 0
    label4 = 0
    label5 = 0
    label6 = 0
    label7 = 0
    for line in test_sample:
        to_change = line.split('-')
        label = to_change[1]
        if label[0:-1] == 'push':
            label1 = label1 + 1
        elif label[0:-1] == 'double click':
            label2 = label2 + 1
        elif label[0:-1] == 'pull':
            label3 = label3 + 1
        elif label[0:-1] == 'rotation':
            label4 = label4 + 1
        elif label[0:-1] == 'grab':
            label5 = label5 + 1
        elif label[0:-1] == 'release':
            label6 = label6 + 1
        elif label[0:-1] == 'upDown':
            label7 = label7 + 1
    print(str(label1) + " " + str(label2) + " " + str(label3) +
          " " + str(label4) + " " + str(label5) + " " + str(label6) + " " + str(label7))
    return train_sample, test_sample


def cross_vaild(rate):
    path = dataPath
    f = open(path, 'r')
    paths = []
    for line in f.readlines():
        paths.append(line)
    random.shuffle(paths)
    res = []
    dataNum = int(len(paths) / rate);
    for i in range(rate):
        if (i + 1) * dataNum >= len(paths):
            test_sample = paths[i * dataNum:-1]
        else:
            test_sample = paths[i * dataNum:(i + 1) * dataNum]
        train_sample = list(set(paths) - set(test_sample))
        res.append((train_sample, test_sample))
    return res

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
        print(str(confusionArray[i][0]) + ',' + str(confusionArray[i][1]) + ',' + str(confusionArray[i][2]) + ','
              + str(confusionArray[i][3]) + ',' + str(confusionArray[i][4]) + ',' + str(
            confusionArray[i][5]) + ',' + str(confusionArray[i][6]))


def train(model, criteria, epoch, train_loader, device, optimizer, test_load):
    model = model.train()
    model = model.to(device)
    drawFlag = [False]
    # f = open("../Performance/train_loss6-20.txt",'w')
    # f1 = open("../Performance/acc_6-20.txt",'w')
    for i in range(epoch):
        for i_batch, (data, dataGyro, label) in enumerate(train_loader):
            model.zero_grad()
            data = data.to(device)
            dataGyro = dataGyro.to(device)
            label = label.to(device)
            pred = model(data, dataGyro)
            # pred = model(dataGyro)
            cost = criteria(pred, label)
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
        if (i + 1) % 5 == 0:
            valLoss, valAcc = test(test_load, device, model, criteria, drawFlag, i, cost.item())
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

    for i_batch, (img, gyro, name) in enumerate(dataSet):
        img = img.to(device)
        name = name.to(device)
        gyro = gyro.to(device)
        pred = model(img, gyro)
        # pred = model(gyro)
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
    if (epoch + 1) % confusionShowStep == 0:
        confusionDraw(confustionArray)
        drawFlag[0] = True
    return lossAge, correct


if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # device = torch.device('cpu')
    # train_sample, test_sample = random_splite(0.9)
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    # dataSet = dataProcess()
    # train_size = int(splitRatio * len(dataSet))
    # test_size = len(dataSet) - train_size
    # train_dataset, test_dataset = torch.utils.data.random_split(dataSet, [train_size, test_size])
    # train_dataset = trainDataProcess(train_sample)
    # test_dataset = testData(test_sample)
    pathRandom = cross_vaild(9)
    for train_sample,test_sample in pathRandom:
        train_dataset = trainDataProcess(train_sample)
        test_dataset = testData(test_sample)
        dataLoad = DataLoader(train_dataset, batch_size=batchSize, num_workers=8, shuffle=True, pin_memory=True)
        testLoad = DataLoader(test_dataset, batch_size=1, num_workers=1)
        # model = cnn()
        # model = MyModel(2,128,1)

        model = mergeModel()

        # model.apply(weight_init)
        for name, parameters in model.named_parameters():
            print(name, ':', parameters.size())

        # attentionParam = list(map(id,model.attention.parameters()))
        # mainParams = filter(lambda p: id(p) not in attentionParam,
        #                         model.parameters())

        criteria = torch.nn.CrossEntropyLoss(reduction='mean')
        # optimizer = optim.Adam([{'params':model.attention.parameters(), 'lr':lrAttention},{'params':mainParams,'lr':lrMain}])
        optimizer = optim.Adam(model.parameters(), lr=lrMain)
        train(model, criteria, epoch, dataLoad, device, optimizer, testLoad)
