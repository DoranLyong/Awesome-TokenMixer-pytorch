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
    x = torch.linspace(-5, 5, 1000, requires_grad=True) # requires_grad=True로 설정하여 자동 미분 가능)
    act = StarReLU()
    out = act(x)

    out.backward(torch.ones_like(x)) # out의 각 요소에 대해 역전파 수행
    x_grad = x.grad
    
    # == Vis. == #
    plt.plot(x.detach().numpy(), out.detach().numpy(), c='r', label='StarReLU')
    plt.plot(x.detach().numpy(), x_grad.detach().numpy(), c='b',label='Gradient of StarReLU')
    plt.xlim(-5, 5)
    plt.ylim(-3, 5)
    plt.legend(fontsize="20", loc='best')
    plt.grid(True)
    plt.show() 