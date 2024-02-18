""" GELU (Gaussian Error Linear Units)
    It has been shown to improve the performance of deep learning models in a variety of tasks.
    : https://arxiv.org/abs/1606.08415
"""
import matplotlib.pyplot as plt 

import torch 
import torch.nn as nn 


class GELUSctrach(nn.Module):
    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(x * (1 + 0.044715 * x * x)))




if __name__ == "__main__": 
    x = torch.linspace(-5, 5, 1000, requires_grad=True) # requires_grad=True로 설정하여 자동 미분 가능
    act = GELUSctrach()
    out = act(x)
    out.backward(torch.ones_like(x)) # out의 각 요소에 대해 역전파 수행
    x_grad = x.grad 

    # == Vis. == #
    plt.plot(x.detach().numpy(), out.detach().numpy(), c='r', label=f'GELU')
    plt.plot(x.detach().numpy(), x_grad.detach().numpy(), c='r', linestyle='--' ,label='Gradient of GELU')    

    plt.title("GELU Activation Function")
    plt.xlabel("Input Value (x)")
    plt.ylabel("Output Value")
    plt.legend(fontsize="20", loc='best')
    plt.grid(True)
    plt.show()
