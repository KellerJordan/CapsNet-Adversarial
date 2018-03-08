import torch
from torch import nn
from torch import optim
from torch.autograd import Variable


class CNN(nn.Module):
    
    def __init__(self, in_shape=(1, 28, 28), n_classes=10):
        super(CNN, self).__init__()
        c, w, h = in_shape
        pool_layers = 3
        fc_h = int(h / 2**pool_layers)
        fc_w = int(w / 2**pool_layers)
        self.conv1 = conv_bn_relu(c, 16)
        self.conv2 = conv_bn_relu(16, 32)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = conv_bn_relu(32, 64)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv4 = conv_bn_relu(64, 128)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.linear1 = nn.Linear(128*fc_h*fc_w, 128)
        self.linear2 = nn.Linear(128, n_classes)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.pool1(x)
        x = self.conv3(x)
        x = self.pool2(x)
        x = self.conv4(x)
        x = self.pool3(x)
        x = x.view(x.size(0), -1)
        x = self.linear1(x)
        x = self.linear2(x)
        return x

def conv_bn_relu(in_channels, out_channels, kernel_size=3,
                 stride=1, padding=1):
    return [
        nn.Conv2d(in_channels, out_channels, kernel_size,
                  stride, padding),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    ]

