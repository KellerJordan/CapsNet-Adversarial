"""PyTorch implementation of Capsule Network

This code is specialized for MNIST or CIFAR10, but could be easily modified for other
datasets.

This implementation splits the model into a CapsuleNetwork and a CapsuleDecoder. The
CapsuleNetwork outputs the vector representation produced by the digitcaps layer. This
is then fed into the CapsuleDecoder, which transforms these vector srepresentations
into a distribution over class labels and reconstructions of the input images.

Code referenced: https://github.com/gram-ai/capsule-networks for several
implementation details.

Author: Keller Jordan
"""

import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F


class CapsuleNetwork(nn.Module):
    
    def __init__(self, p_caps=32, d_caps_dim=16, img_colors=1):
        super().__init__()
        self.conv1 = nn.Conv2d(img_colors, 256, 9)
        self.primary_caps = CapsuleLayer('primary', p_caps=p_caps)
        self.digit_caps = CapsuleLayer('digit', p_caps=p_caps, d_caps_dim=d_caps_dim)
        
    def forward(self, img):
        relu_conv1 = F.relu(self.conv1(img))
        p_caps = self.primary_caps(relu_conv1)
        d_caps = self.digit_caps(p_caps)
        return d_caps

class CapsuleLayer(nn.Module):
    
    def __init__(self, caps_type, p_caps=32, d_caps_dim=16, routing_iters=3, use_cuda=True):
        super().__init__()
        self.caps_type = caps_type
        if self.caps_type == 'primary':
            self.out_caps = p_caps
            self.out_dim = 8
            self.capsules = nn.ModuleList(
                [nn.Conv2d(256, self.out_dim, 9, 2) for _ in range(self.out_caps)])
        # the digitcaps architecture is also used for the final layer in the model
        # trained on CIFAR10
        elif self.caps_type == 'digit':
            self.out_caps = 10
            self.out_dim = d_caps_dim
            self.in_caps = p_caps*6*6
            self.in_dim = 8
            self.route_weights = nn.Parameter(torch.randn(
                self.out_caps, self.in_caps, self.in_dim, self.out_dim))
            self.routing_iters = routing_iters
        else:
            raise Exception('[!] Unknown capsule layer architecture: %s' % caps_type)
        
        self.use_cuda = use_cuda
    
    # squashing function Eq. (1)
    def squash(self, x, dim=-1):
        sq_norm = (x**2).sum(dim, keepdim=True)
        scale = sq_norm / (1 + sq_norm)
        return scale * x / sq_norm.sqrt()
    
    def forward(self, x):
        if self.caps_type == 'primary':
            x = [capsule(x).view(x.size(0), self.out_dim, -1) for capsule in self.capsules]
            x = torch.cat(x, dim=-1)
            outputs = self.squash(x.transpose(1, 2))
            # batch_size, out_caps, out_dim
        else:
            u_hats = (x[:, None, :, None, :] @ self.route_weights[...]).squeeze(3)
            u_hats = u_hats.transpose(2, 3)
            # batch_size, out_caps, out_dim, in_caps
            
            logits = Variable(torch.zeros((x.size(0), self.in_caps, self.out_caps)))
            if self.use_cuda:
                logits = logits.cuda()
            
            for i in range(1, self.routing_iters+1):
                probs = F.softmax(logits, dim=1)
                # batch_size, in_caps, out_caps

                capsule_s = (u_hats @ probs.transpose(1, 2)[..., None]).squeeze(3)
                outputs = self.squash(capsule_s)
                # batch_size, out_caps, out_dim
                
                if i < self.routing_iters:
                    updates = (outputs[..., None, :] @ u_hats).squeeze(2).transpose(1, 2)
                    # batch_size, in_caps, out_caps

                    logits = logits + updates
        
        return outputs

    
class ReconstructionNetwork(nn.Module):
    
    def __init__(self, img_colors=1):
        super().__init__()
        self.fc1 = nn.Linear(160, 512)
        self.fc2 = nn.Linear(512, 1024)
        self.fc3 = nn.Linear(1024, img_colors*784)
        self.img_colors = img_colors
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.sigmoid(self.fc3(x))
        img = x.view(x.size(0), self.img_colors, int((x.size(1)/self.img_colors)**0.5), -1)
        return img

class CapsuleDecoder(nn.Module):
    
    def __init__(self, img_colors=1, reconstruction=True, mask_incorrect=True):
        super().__init__()
        self.reconstruction = reconstruction
        if reconstruction:
            self.decoder = ReconstructionNetwork(img_colors)
            self.mask_incorrect = mask_incorrect
    
    def forward(self, d_caps, labels=None):
        logits = torch.norm(d_caps, dim=-1)
        
        # every implementation I have seen runs the norms of (already squashed) digitcaps 
        # outputs through softmax as follows, but it may be better to use them as 
        # probabilities of digit presence as in paper
        probs = F.softmax(logits, dim=1)
        
        if self.reconstruction:
            if self.mask_incorrect:
                # mask all but the correct (maximum for unsupervised) digit
                if labels is None:
                    _, class_indices = logits.max(1)
                    identity = Variable(torch.sparse.torch.eye(10)).type(d_caps.data.type())
                    y = identity.index_select(0, class_indices)
                else:
                    y = labels
                masked_caps = d_caps * y[..., None]
                decoder_input = masked_caps.view(masked_caps.size(0), -1)
            else:
                # allow all capsule outputs to be input to reconstruction network
                decoder_input = d_caps.view(d_caps.size(0), -1)
            
            img_hats = self.decoder(decoder_input)
            return probs, img_hats
        else:
            return probs

# module to combine the network and decoder to use with utils.Trainer api
class CapsuleModel(nn.Module):
    
    def __init__(self, net, decoder):
        super().__init__()
        self.net = net
        self.decoder = decoder
    
    def forward(self, x, y=None):
        d_caps = self.net(x)
        probs, reconstructions = self.decoder(d_caps, y)
        return probs, reconstructions

# loss function for capsule network -- margin loss plus reconstruction loss
class CapsuleLoss(nn.Module):
    
    def __init__(self, mplus=0.9, mminus=0.1, lmbda=0.5, rcsn_scale=0.0005):
        super().__init__()
        self.mplus = mplus
        self.mminus = mminus
        self.lmbda = lmbda
        self.rcsn_scale = rcsn_scale
    
    def forward(self, probs, labels, reconstructions=None, images=None):
        present = F.relu(self.mplus - probs)**2
        absent = F.relu(probs - self.mminus)**2
        margin_loss = labels * present + self.lmbda * (1 - labels) * absent
        margin_loss = margin_loss.sum(1).sum(0)
        if reconstructions is not None:
            reconstruction_loss = F.mse_loss(reconstructions, images, size_average=False)
            loss = margin_loss + self.rcsn_scale * reconstruction_loss
        else:
            loss = margin_loss
        return loss.mean()
