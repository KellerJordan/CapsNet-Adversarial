import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F


# extremely simple network to do basic science with training methods
class BasicNetwork(nn.Module):
    
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 100)
        self.fc2 = nn.Linear(100, 10)
    
    def forward(self, x):
        x = x.view(-1, 784)
        x = F.relu(self.fc1(x))
        out = self.fc2(x)
        return out
