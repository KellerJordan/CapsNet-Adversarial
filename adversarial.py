import torch
from torch import nn
from torch.autograd import Variable


def set_grad(model, cond):
    for p in model.parameters():
        p.requires_grad = cond

class Adversary():
    
    def __init__(self, model):
        self.model = model
    
    def generate_example(self, seed_img, target_class, attack, **kwargs):
        set_grad(self.model, False)
        img_var = Variable(seed_img.clone(), requires_grad=True)
        
        if attack == 'GA':
            fool_img = self.GA(img_var, target_class, **kwargs)
        elif attack == 'FGS':
            fool_img = self.FGS(img_var, target_class, **kwargs)
        else:
            raise Exception('[!] Unknown attack method specified')
        
        set_grad(self.model, True)
        return fool_img
    
    
    
    # gradient ascent
    def GA(self, img_var, target, n_iters=100, eta=0.005):
        for _ in range(n_iters):
            scores = self.model(img_var)
            objective = scores.squeeze()[target]
            objective.backward()

            g = img_var.grad.data.clone()
            img_var.grad.zero_()
            g = g / g.norm()

            step = eta * g
            img_var.data += step.cuda()
            img_var.data = torch.clamp(img_var.data, min=-1, max=1)
        
        return img_var.data

    # fast gradient sign
    def FGS(self, img_var, target, n_iters=100, eta=0.005):
        for _ in range(n_iters):
            scores = self.model(img_var)
            objective = scores.squeeze()[target]
            objective.backward()
            
            g = img_var.grad.data.clone()
            g = g.abs() / (g + 1e-4) # sign of gradient
            
            step = eta * g
            img_var.data += step.cuda()
            img_var.data = torch.clamp(img_var.data, min=-1, max=1)
        
        return img_var.data
    