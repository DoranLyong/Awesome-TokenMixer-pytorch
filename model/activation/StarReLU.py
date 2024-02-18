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
    x0 = torch.linspace(-5, 5, 1000, requires_grad=True) # requires_grad=True로 설정하여 자동 미분 가능)
    x1 = torch.linspace(-5, 5, 1000, requires_grad=True) 
    x2 = torch.linspace(-5, 5, 1000, requires_grad=True) 
    
        
    act0 = StarReLU(scale_value=1.0, bias_value=0.0)
    out0 = act0(x0)
    out0.backward(torch.ones_like(x0)) # out의 각 요소에 대해 역전파 수행
    x0_grad = x0.grad
    

    act1 = StarReLU(scale_value=1.0, bias_value=1.0)
    out1 = act1(x1)
    out1.backward(torch.ones_like(x1)) # out의 각 요소에 대해 역전파 수행
    x1_grad = x1.grad

    
    act2 = StarReLU(scale_value=0.5, bias_value=-1.0)
    out2 = act2(x2)
    out2.backward(torch.ones_like(x2)) # out의 각 요소에 대해 역전파 수행
    x2_grad = x2.grad


    # == Vis. == #
    plt.plot(x0.detach().numpy(), out0.detach().numpy(), c='r', label=f'StarReLU scale={1.0}, bias={0.0}')
    plt.plot(x0.detach().numpy(), x0_grad.detach().numpy(), c='r', linestyle='--' ,label='Gradient')

    plt.plot(x1.detach().numpy(), out1.detach().numpy(), c='b', label=f'StarReLU scale={1.0}, bias={1.0}')
    plt.plot(x1.detach().numpy(), x1_grad.detach().numpy(), c='b', linestyle='--' ,label='Gradient')    

    plt.plot(x2.detach().numpy(), out2.detach().numpy(), c='g', label=f'StarReLU scale={0.5}, bias={-1.0}')
    plt.plot(x2.detach().numpy(), x2_grad.detach().numpy(), c='g', linestyle='--' ,label='Gradient')    

    plt.title("StarReLU Activation Function")
    plt.xlabel("Input Value (x)")
    plt.ylabel("Output Value")
    plt.legend(fontsize="20", loc='best')
    plt.grid(True)
    plt.show()