import torch
from torch import nn
from torch.autograd import Variable


def set_grad(model, cond):
    for p in model.parameters():
        p.requires_grad = cond

class Adversary():
    
    def __init__(self, model):
        self.model = model
    
    def generate_example(self, seed_img, attack, target=None, ground=None, **kwargs):
        set_grad(self.model, False)
        img_var = Variable(seed_img.clone(), requires_grad=True)
        
        if attack == 'GA':
            fool_img = self.GA(img_var, target, ground, **kwargs)
        elif attack == 'FGS':
            fool_img = self.FGS(img_var, target, ground, **kwargs)
        else:
            raise Exception('[!] Unknown attack method specified')
        
        set_grad(self.model, True)
        return fool_img.data.cpu()
    
    
    
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
            img_var.data = torch.clamp(img_var.data, min=0, max=1)
        
        return img_var

    # fast gradient sign
    def FGS(self, img_var, target=None, ground=None, n_iters=100, eta=0.005):
        for _ in range(n_iters):
            scores = self.model(img_var).squeeze()
            objective = 0
            if target is not None:
                objective = objective + scores[target]
            if ground is not None:
                objective = objective - scores[ground]
            objective.backward()
            
            g = img_var.grad.data.clone()
            g = g.abs() / (g + 1e-4) # sign of gradient
            
            step = eta * g
            img_var.data += step.cuda()
            img_var.data = torch.clamp(img_var.data, min=0, max=1)
        
        return img_var
    