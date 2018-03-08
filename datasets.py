import numpy as np

import torch
from torch.autograd import Variable
from torch.utils import DataLoader

import torchvision
from torchvision import transforms as T
from torchvision import datasets as dset


TRN_TRANSFORM = transforms.Compose([
    transforms.ToTensor()
])
TST_TRANSFORM = transforms.Compose([
    transforms.ToTensor()
])

def get_mnist_dataset(trn_size=60000, tst_size=10000):
    trainset = dset.MNIST(root='./data', train=True,
                          download=True, transform=TRN_TRANSFORM)
    testset = dset.MNIST(root='./data', train=False,
                         download=True, transform=TST_TRANSFORM)

def get_data_loader(trainset, testset, batch_size=128):
    trainloader = DataLoader(trainset, batch_size=batch_size,
                             shuffle=True)
    testloader = DataLoader(testset, batch_size=batch_size,
                            shuffle=False)
    return trainloader, testloader
