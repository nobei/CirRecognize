from torch import nn


class ultraGloves(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        cnn = nn.Sequential()
        # cnn.add_module('myAttention0',myAttention())
        cnn.add_module('conv0',
                       nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(10, 10), padding=(5, 5)))
        cnn.add_module('batch0', nn.BatchNorm2d(32))
        cnn.add_module('relu0', nn.ReLU(True))
        cnn.add_module('pool0',
                       nn.MaxPool2d(kernel_size=(2, 2), stride=2))

        cnn.add_module('conv1',
                       nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(5, 5), padding=(2, 2)))
        cnn.add_module('batch1', nn.BatchNorm2d(64))
        cnn.add_module('relu1', nn.ReLU(True))
        cnn.add_module('pool1',
                       nn.MaxPool2d(kernel_size=(2, 2), stride=2))

        cnn.add_module('conv2',
                       nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(5, 5), padding=(2, 2)))
        cnn.add_module('batch2', nn.BatchNorm2d(128))
        cnn.add_module('relu2', nn.ReLU(True))
        cnn.add_module('pool2',
                       nn.MaxPool2d(kernel_size=(2, 2), stride=2))

        cnn.add_module('conv3',
                       nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(5, 5), padding=(2, 2)))
        cnn.add_module('batch3', nn.BatchNorm2d(128))
        cnn.add_module('relu3', nn.ReLU(True))
        cnn.add_module('pool3',
                       nn.MaxPool2d(kernel_size=(2, 2), stride=2))

        cnn.add_module('conv4',
                       nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), padding=(1, 1)))
        cnn.add_module('batch4', nn.BatchNorm2d(128))
        cnn.add_module('relu4', nn.ReLU(True))
        cnn.add_module('pool4',
                       nn.MaxPool2d(kernel_size=(2, 2), stride=2))

        self.cnn = cnn

    def forward(self, input):
        b, h, w = input.size()
        input = input.unsqueeze(1)
        output = self.cnn(input)
        output = output.reshape(b, -1)
        return output
