import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F


class CNN(nn.Module):
    
    def __init__(self, in_shape=(1, 28, 28), n_classes=10):
        super().__init__()
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
        self.fc1 = nn.Linear(128*fc_h*fc_w, 128)
        self.fc2 = nn.Linear(128, n_classes)
    
    def forward(self, x):
        x = self.conv(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.fc2(x)
        return x

def conv_bn_relu(in_channels, out_channels, kernel_size=3,
                 stride=1, padding=1):
    return [
        nn.Conv2d(in_channels, out_channels, kernel_size,
                  stride, padding),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    ]

class BaselineCNN(nn.Module):
    
    def __init__(self, img_colors=1):
        super().__init__()
        self.conv1 = nn.Conv2d(img_colors, 256, 5, padding=2)
        self.conv2 = nn.Conv2d(256, 256, 5, padding=2)
        self.conv3 = nn.Conv2d(256, 128, 5, padding=2)
        flat_dim = 128*28*28
        self.flatten = lambda x: x.view(-1, flat_dim)
        self.fc1 = nn.Linear(flat_dim, 328)
        self.fc2 = nn.Linear(328, 192)
        self.dropout = nn.Dropout()
        self.fc3 = nn.Linear(192, 10)
    
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        logits = self.fc3(x)
        return logits
        
