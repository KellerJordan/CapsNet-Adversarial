import matplotlib.pyplot as plt
import numpy as np

import torch
import torch.nn as nn
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
