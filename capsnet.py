import torch
from torch import nn
from torch import optim
from torch.autograd import Variable


class CapsNet(nn.Module):
    
    def __init__(self):
        super(CapsNet, self).__init__()
    
    def forward(self, x):
        return x
