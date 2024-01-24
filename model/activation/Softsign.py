import matplotlib.pyplot as plt 

import torch 
import torch.nn as nn 
import torch.nn.functional as F

# Softsign 함수 정의
def softsign(x):
    return x / (1 + torch.abs(x))


class Softsign(nn.Module):
    def __init__(self):
        super(Softsign, self).__init__()
        self.softsign = nn.Softsign() 

    def forward(self, x): 
        return self.softsign(x) 
    

if __name__ == "__main__": 
    x = torch.linspace(-5, 5, 1000, requires_grad=True) # requires_grad=True로 설정하여 자동 미분 가능
    act = Softsign()
    out = act(x)

    out.backward(torch.ones_like(x))
    x_grad = x.grad

    # == Vis. == #
    plt.plot(x.detach().numpy(), out.detach().numpy(), c='r', label='Softsign')
    plt.plot(x.detach().numpy(), x_grad.detach().numpy(), c='b',label='Gradient of Softsign')
    plt.xlim(-5, 5)
    plt.ylim(-3, 5)
    plt.legend(loc='best')
    plt.grid(True)
    plt.show() 