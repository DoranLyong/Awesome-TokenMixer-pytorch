import matplotlib.pyplot as plt 

import torch 
import torch.nn as nn 
import torch.nn.functional as F


class GELU(nn.Module):
    def __init__(self):
        super(GELU, self).__init__()
        self.gelu = nn.GELU()

    def forward(self, x):
        return self.gelu(x)



if __name__ == "__main__": 
    x = torch.linspace(-5, 5, 1000, requires_grad=True) # requires_grad=True로 설정하여 자동 미분 가능
    act = GELU()
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