import os

import matplotlib.pyplot as plt
import numpy as np

import torch
from torch import nn
from torch.autograd import Variable
import torchvision


def plot_tensor(img, fs=(10, 10), title=''):
    if torch.tensor._TensorBase in type(img).__bases__:
        npimg = img.numpy()
    else:
        npimg = img
    if len(npimg.shape) == 4:
        npimg = np.squeeze(npimg)
    npimg = npimg.transpose(1, 2, 0)
    plt.figure(figsize=fs)
    if npimg.shape[2] > 1:
        plt.imshow(npimg)
    else:
        plt.imshow(npimg, cmap='gray')
    plt.title(title)
    plt.show()

def plot_batch(samples):
    sample_grid = torchvision.utils.make_grid(samples)
    plot_tensor(sample_grid)

def get_argmax(scores):
    val, idx = torch.max(scores, dim=1)
    return idx.data.view(-1).cpu().numpy()

def get_accuracy(pred, target):
    correct = np.sum(pred == target)
    return correct / len(pred)

class Trainer():
    def __init__(self, model, optimizer, criterion, trn_loader, tst_loader, use_cuda=False):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.trn_loader = trn_loader
        self.tst_loader = tst_loader
        self.metrics = {
            'loss': {
                'trn':[],
                'tst':[]
            },
            'accuracy': {
                'trn':[],
                'tst':[]
            },
        }
        self.use_cuda = use_cuda
    
    def run(self, epochs):
        for epoch in range(1, epochs+1):
            trn_loss, trn_acc = self.train()
            tst_loss, tst_acc = self.test()
            print('[*] Epoch %d, TrnLoss: %.3f, TrnAcc: %.3f, TstLoss: %.3f, TstAcc: %.3f'
                % (epoch, trn_loss, trn_acc, tst_loss, tst_acc))
            self.metrics['accuracy']['trn'].append(trn_acc)
            self.metrics['accuracy']['tst'].append(tst_acc)
            self.metrics['loss']['trn'].append(trn_loss)
            self.metrics['loss']['tst'].append(tst_loss)
    
    def train(self):
        self.model.train()

        n_batches = len(self.trn_loader)
        cum_loss = 0
        cum_acc = 0
        for i, (X, y) in enumerate(self.trn_loader):
            if self.use_cuda:
                X = X.cuda()
                y = y.cuda()
            X_var = Variable(X)
            y_var = Variable(y)

            scores = self.model(X_var)
            loss = self.criterion(scores, y_var)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            pred = get_argmax(scores)
            acc = get_accuracy(pred, y_var.data.cpu().numpy())
            cum_acc += acc
            cum_loss += loss.data[0]
        
        return cum_loss / n_batches, cum_acc / n_batches
    
    def test(self):
        self.model.eval()

        n_batches = len(self.tst_loader)
        cum_loss = 0
        cum_acc = 0
        for i, (X, y) in enumerate(self.tst_loader):
            if self.use_cuda:
                X = X.cuda()
                y = y.cuda()
            X_var = Variable(X)
            y_var = Variable(y)

            scores = self.model(X_var)
            loss = self.criterion(scores, y_var)

            pred = get_argmax(scores)
            acc = get_accuracy(pred, y_var.data.cpu().numpy())
            cum_acc += acc
            cum_loss += loss.data[0]
        
        return cum_loss / n_batches, cum_acc / n_batches

    def save_checkpoint(self, filename='checkpoint.pth.tar'):
        state = {
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict()}
        torch.save(state, filename)

    def load_checkpoint(self, filename='checkpoint.pth.tar'):
        if os.path.isfile(filename):
            state = torch.load(filename)
            self.model.load_state_dict(state['model'])
            self.optimizer.load_state_dict(state['optimizer'])
        else:
            print('%s not found.' % filename)
