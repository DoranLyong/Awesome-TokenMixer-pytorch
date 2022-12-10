# (ref) https://github.com/sail-sg/metaformer/blob/main/metaformer_baselines.py

import matplotlib.pyplot as plt 

import torch 
import torch.nn as nn 
import torch.nn.functional as F


class StarReLU(nn.Module):
    """
    StarReLU: s * relu(x) ** 2 + b
    """
    def __init__(self, scale_value=1.0, bias_value=0.0,
        scale_learnable=True, bias_learnable=True, 
        mode=None, inplace=False):
        super().__init__()
        self.inplace = inplace
        self.relu = nn.ReLU(inplace=inplace)
        self.scale = nn.Parameter(scale_value * torch.ones(1),
            requires_grad=scale_learnable)
        self.bias = nn.Parameter(bias_value * torch.ones(1),
            requires_grad=bias_learnable)
    def forward(self, x):
        return self.scale * self.relu(x)**2 + self.bias


if __name__ == "__main__": 
    x = torch.linspace(-5, 5, 1000)
    act = StarReLU()
    out = act(x)
    
    # == Vis. == #
    plt.plot(x.detach().numpy(), out.detach().numpy(), c='r', label='StarReLU')
    plt.xlim(-5, 5)
    plt.ylim(-3, 5)
    plt.legend(loc='best')
    plt.show() 