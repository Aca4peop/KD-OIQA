import torch
import torch.nn as nn

class policy_Net(nn.Module):

    def __init__(self):
        super(policy_Net, self).__init__()
        
        self.net = nn.Sequential(nn.Linear(1024, 2048),
                                 nn.Linear(2048, 512),
                                 nn.Sigmoid())
        
        self.sign = Bisign.apply
           
    def forward(self, feat, feat_con):
        x = torch.cat((feat, feat_con), dim=1)
        x = self.net(x)

        mask = self.sign(x)

        x = (1-mask) * feat + mask * feat_con
        return mask, x
    
class Bisign(torch.autograd.Function):
    
    @staticmethod
    def forward(ctx, input):
        out = torch.ones(input[0].size()).cuda()
        return out*(input>0.5)
 
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.clamp_(-1, 1)