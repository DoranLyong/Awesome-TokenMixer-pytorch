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
    x = torch.linspace(-5, 5, 1000, requires_grad=True) # requires_grad=True로 설정하여 자동 미분 가능
    act = SquaredReLU()
    out = act(x)
    out.backward(torch.ones_like(x)) # out의 각 요소에 대해 역전파 수행
    x_grad = x.grad     
    
    # == Vis. == #
    plt.plot(x.detach().numpy(), out.detach().numpy(), c='r', label=f'SquaredReLU')
    plt.plot(x.detach().numpy(), x_grad.detach().numpy(), c='b', label='Gradient of SquaredReLU')    

    plt.title("SquaredReLU Activation Function")
    plt.xlabel("Input Value (x)")
    plt.ylabel("Output Value")
    plt.legend(fontsize="20", loc='best')
    plt.grid(True)
    plt.show()