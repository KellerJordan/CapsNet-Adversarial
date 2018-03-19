import torch
from torch import nn
from torch.autograd import Variable


def set_grad(model, cond):
    for p in model.parameters():
        p.requires_grad = cond

def generate_fooling_image(model, seed_img, target=5,
                  n_iters=100, alpha=0.005):
    """
    inputs: seed_img (Tensor), model
    outputs: fool_img (Tensor)
    """
    set_grad(model, False)
    
    img_var = Variable(seed_img.clone(), requires_grad=True)
    
    for _ in range(n_iters):
        
        scores = model(img_var)
        objective = scores.squeeze()[target]
        objective.backward()
        
        g = img_var.grad.data.clone()
        img_var.grad.zero_()
        g = g / g.norm()
        
        step = alpha * g
        img_var.data += step.cuda()
        img_var.data = torch.clamp(img_var.data, min=-1, max=1)
    
    fool_img = img_var.data.clone()
    
    set_grad(model, True)
    return fool_img
