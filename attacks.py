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
            res = self.GA(img_var, target, ground, **kwargs)
        elif attack == 'FGS':
            res = self.FGS(img_var, target, ground, **kwargs)
        else:
            raise Exception('[!] Unknown attack method specified')
        
        set_grad(self.model, True)
        if type(res) == tuple:
            fool_image, iters = res
            return fool_image.data.cpu(), iters
        else:
            fool_image = res
            return fool_image.data.cpu()
    
    # gradient ascent
    def GA(self, img_var, target=None, ground=None, n_iters=100, eta=0.005, stop_early=True):
        for i in range(n_iters):
            scores = self.model(img_var).squeeze()
            objective = 0
            if target is not None:
                if stop_early and scores.max(0)[1].data[0] == target:
                    break
                objective = objective + scores[target]
            if ground is not None:
                if stop_early and scores.max(0)[1].data[0] != ground:
                    break
                objective = objective - scores[ground]
            objective.backward()

            g = img_var.grad.data.clone()
            img_var.grad.zero_()
            g = g / g.norm()

            step = eta * g
            img_var.data += step.cuda()
            img_var.data = torch.clamp(img_var.data, min=0, max=1)
        
        return img_var, i

    # fast gradient sign
    def FGS(self, img_var, target=None, ground=None, n_iters=100, eta=0.005, stop_early=True):
        for i in range(n_iters):
            scores = self.model(img_var).squeeze()
            objective = 0
            if target is not None:
                if stop_early and scores.max(0)[1].data[0] == target:
                    break
                objective = objective + scores[target]
            if ground is not None:
                if stop_early and scores.max(0)[1].data[0] != ground:
                    break
                objective = objective - scores[ground]
            objective.backward()
            
            g = img_var.grad.data.clone()
            img_var.grad.zero_()
            g = g.abs() / (g + 1e-15) # sign of gradient
            
            step = eta * g
            img_var.data += step.cuda()
            img_var.data = torch.clamp(img_var.data, min=0, max=1)
        
        return img_var, i
    
def evaluate_example(seed_img, fool_img, model, reconstruction=False):
    seed_var = Variable(seed_img).cuda()
    fool_var = Variable(fool_img).cuda()
    if reconstruction:
        scores_seed, rec_seed = model(seed_var)
        scores_fool, rec_fool = model(fool_var)
    else:
        scores_seed = model(seed_var)
        scores_fool = model(fool_var)
    
    results = {
        'pred_seed': scores_seed.data.cpu().max(1)[1][0],
        'pred_fool': scores_fool.data.cpu().max(1)[1][0]}
    if reconstruction:
        results['mse_seed'] = torch.sum((rec_seed - seed_var)**2).data.cpu()[0]
        results['mse_fool'] = torch.sum((rec_fool - fool_var)**2).data.cpu()[0]
        reconstructions = {
            'rec_seed': rec_seed.data.cpu(),
            'rec_fool': rec_fool.data.cpu()}
        return results, reconstructions
    else:
        return results
