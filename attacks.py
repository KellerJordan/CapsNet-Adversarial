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
        for i in range(n_iters):
            scores = self.model(img_var).squeeze()
            objective = 0
            if target is not None:
                objective = objective + scores[target]
            if ground is not None:
                objective = objective - scores[ground]
            if target is not None and ground is not None:
                if scores[target].cpu().data[0] > scores[ground].cpu().data[0]:
                    print('success after %d iters' % i)
                    break
            elif ground is not None:
                if scores.max(0)[1].data[0] != ground:
                    print('success after %d iters' % i)
                    break
            objective.backward()
            
            g = img_var.grad.data.clone()
            g = g.abs() / (g + 1e-4) # sign of gradient
            
            step = eta * g
            img_var.data += step.cuda()
            img_var.data = torch.clamp(img_var.data, min=0, max=1)
        
        return img_var
    
def evaluate_example(seed_img, fool_img, full_model):
    seed_var = Variable(seed_img).cuda()
    fool_var = Variable(fool_img).cuda()
    probs_seed, rec_seed = full_model(seed_var)
    probs_fool, rec_fool = full_model(fool_var)
    results = {
        'pred_seed': probs_seed.data.cpu().max(1)[1][0],
        'pred_fool': probs_fool.data.cpu().max(1)[1][0],
        'mse_seed': torch.sum((rec_seed - seed_var)**2).data.cpu()[0],
        'mse_fool': torch.sum((rec_fool - fool_var)**2).data.cpu()[0]}
    reconstructions = {
        'rec_seed': rec_seed.data.cpu(),
        'rec_fool': rec_fool.data.cpu()}
    return results, reconstructions
