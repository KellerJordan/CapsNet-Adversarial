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
        self.conv = nn.Sequential(
            *conv_bn_relu(c, 16),
            *conv_bn_relu(16, 32),
            nn.MaxPool2d(kernel_size=2, stride=2),
            *conv_bn_relu(32, 64),
            nn.MaxPool2d(kernel_size=2, stride=2),
            *conv_bn_relu(64, 128),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.flatten = lambda x: x.view(-1, 128*fc_h*fc_w)
        self.linear1 = nn.Linear(128*fc_h*fc_w, 128)
        self.linear2 = nn.Linear(128, n_classes)
    
    def forward(self, x):
        x = self.conv(x)
        x = self.flatten(x)
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

