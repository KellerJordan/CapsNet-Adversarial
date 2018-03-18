import torch
from torch import nn
from torch import optim
from torch.autograd import Variable
import torch.nn.functional as F


class CapsNet(nn.Module):
    
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 256, 9)
        self.primary_caps = CapsuleLayer('primary')
        self.digit_caps = CapsuleLayer('digit')
        
    def forward(self, img):
        relu_conv1 = F.relu(self.conv1(img))
        p_caps = self.primary_caps(relu_conv1)
        d_caps = self.digit_caps(p_caps)
        return d_caps

class CapsuleLayer(nn.Module):
    
    def __init__(self, caps_type, routing_iters=3, use_cuda=True):
        super().__init__()
        self.caps_type = caps_type
        if self.caps_type == 'primary':
            self.in_caps = 32
            self.in_dim = 8
            self.capsules = nn.ModuleList(
                [nn.Conv2d(256, self.in_dim, 9, 2) for _ in range(self.in_caps)])
        elif self.caps_type == 'digit':
            self.out_caps = 10
            self.out_dim = 16
            self.in_caps = 32*6*6
            self.in_dim = 8
            self.route_weights = nn.Parameter(torch.randn(self.out_caps, self.in_caps, self.in_dim, self.out_dim))
            self.routing_iters = routing_iters
        else:
            raise Exception('[!] Unknown capsule architecture: %s' % caps_type)
        
        self.use_cuda = use_cuda
    
    # squashing function Eq. (1)
    def squash(self, x, dim=-1):
        sq_norm = (x**2).sum(dim, keepdim=True)
        scale = sq_norm / (1 + sq_norm)
        return scale * x / sq_norm.sqrt()
    
    def forward(self, x):
        if self.caps_type == 'primary':
            x = [capsule(x).view(x.size(0), self.in_dim, -1) for capsule in self.capsules]
            x = torch.cat(x, dim=-1)
            outputs = self.squash(x.transpose(1, 2))
            # batch_size, out_caps, out_dim
        else:
            u_hats = (x[:, None, :, None, :] @ self.route_weights[...]).squeeze()
            u_hats = u_hats.transpose(2, 3)
            # batch_size, out_caps, out_dim, in_caps
            ## batch_size, out_caps, in_caps, out_dim
            
            logits = Variable(torch.zeros((x.size(0), self.in_caps, self.out_caps)))
            if self.use_cuda:
                logits = logits.cuda()
            
            for i in range(self.routing_iters):
                probs = F.softmax(logits, dim=1)
                # batch_size, in_caps, out_caps

                capsule_s = (u_hats @ probs.transpose(1, 2)[..., None]).squeeze()
                outputs = self.squash(capsule_s)
                # batch_size, out_caps, out_dim
                
                updates = (outputs[..., None, :] @ u_hats).squeeze().transpose(1, 2)
                # batch_size, in_caps, out_caps
                
                logits += updates
        
        return outputs

    
class ReconstructionNet(nn.Module):
    
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(160, 512)
        self.fc2 = nn.Linear(512, 1024)
        self.fc3 = nn.Linear(1024, 784)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.sigmoid(self.fc3(x))
        img = x.view(x.size(0), int(x.size(1)**0.5), -1)
        return img[:, None, ...]

class CapsuleDecoder(nn.Module):
    
    def __init__(self):
        super().__init__()
        self.decoder = ReconstructionNet()
    
    def forward(self, d_caps):
        reconstructions = self.decoder(d_caps.view(d_caps.size(0), -1))
        logits = torch.norm(d_caps, dim=-1)
        probs = F.softmax(logits, dim=0) # THIS SHOULD NOT NECESSARILY BE HERE
        return reconstructions, probs

class CapsLoss(nn.Module):
    
    def __init__(self):
        super().__init__()
        self.reconstruction_loss = nn.MSELoss()
    
    def forward(self, images, labels, reconstructions, probs,
                mplus=0.9, mminus=0.1, lmbda=0.5, tradeoff=0.0005):
        present = F.relu(mplus - probs)**2
        absent = F.relu(probs - mminus)**2
        margin_loss = labels * present + lmbda * (1 - labels) * absent
        reconstruction_loss = self.reconstruction_loss(reconstructions, images)
        return margin_loss + tradeoff * reconstruction_loss
        
class CapsPipeline(nn.Module):
    
    def __init__(self):
        super().__init__()
        self.network = CapsNet()
        self.decoder = CapsuleDecoder()
        self.loss = CapsLoss()
    
    def forward(self, images, labels):
        capsules_v = self.network(images)
        reconstructions, probs = self.decoder(capsules_v)
        loss = self.loss(images, labels, reconstructions, probs)
        return probs, reconstructions, loss
