import torch
from torch import nn
from torch.autograd import Variable


def perturb_image(model, img, n_iters=100, tgt=5):
    """
    inputs: img (Tensor), model
    outputs: img_fool (Tensor)
    """
    
    # turn requires_grad off for all params in model
    
    img_var = Variable(img, requires_grad=True)
    
    for _ in range(n_iters):
        scores = model(img_var)
        cost = -scores[5]
    
    # turn requires_grad on for all params in model
    
    img_fool = img_var.data
    return img_fool
