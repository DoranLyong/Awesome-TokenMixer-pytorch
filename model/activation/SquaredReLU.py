# (ref) https://github.com/sail-sg/metaformer/blob/main/metaformer_baselines.py

import matplotlib.pyplot as plt 

import torch 
import torch.nn as nn 
import torch.nn.functional as F


class SquaredReLU(nn.Module):
    """
        Squared ReLU: https://arxiv.org/abs/2109.08668
    """
    def __init__(self, inplace=False):
        super().__init__()
        self.relu = nn.ReLU(inplace=inplace)
    def forward(self, x):
        return torch.square(self.relu(x))


if __name__ == "__main__": 
    x = torch.linspace(-5, 5, 1000)
    act = SquaredReLU()
    out = act(x)
    
    # == Vis. == #
    plt.plot(x.detach().numpy(), out.detach().numpy(), c='r', label='SquaredReLU')
    plt.xlim(-5, 5)
    plt.ylim(-3, 5)
    plt.legend(loc='best')
    plt.show() 