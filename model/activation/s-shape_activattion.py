# Comparision of Sigmoid, Tanh, and Softsign.

import matplotlib.pyplot as plt 

import torch 
import torch.nn as nn 
import torch.nn.functional as F


class Sigmoid(nn.Module):
    def __init__(self):
        super(Sigmoid, self).__init__()
        """ Make it range in [-1, 1]
        """
        self.sigmoid = nn.Sigmoid() 

    def forward(self, x): 
        return self.sigmoid(x)*2 - 1
    

class Softsign(nn.Module):
    def __init__(self):
        super(Softsign, self).__init__()
        self.softsign = nn.Softsign() 

    def forward(self, x): 
        return self.softsign(x)     
    

class Tanh(nn.Module):
    def __init__(self):
        super(Tanh, self).__init__()
        self.tanh = nn.Tanh() 

    def forward(self, x): 
        return self.tanh(x)     



if __name__ == "__main__": 
    x1 = torch.linspace(-5, 5, 1000, requires_grad=True) # requires_grad=True로 설정하여 자동 미분 가능
    x2 = torch.linspace(-5, 5, 1000, requires_grad=True)
    x3 = torch.linspace(-5, 5, 1000, requires_grad=True)
    
    
    sigmoid_act = Sigmoid()
    out1 = sigmoid_act(x1)
    out1.backward(torch.ones_like(x1)) # out의 각 요소에 대해 역전파 수행
    x1_grad = x1.grad


    softsign_act = Softsign()
    out2 = softsign_act(x2)
    out2.backward(torch.ones_like(x2)) # out의 각 요소에 대해 역전파 수행
    x2_grad = x2.grad


    tanh_act = Tanh()
    out3 = tanh_act(x3)
    out3.backward(torch.ones_like(x3)) # out의 각 요소에 대해 역전파 수행
    x3_grad = x3.grad


    # == Vis. == #
    plt.plot(x1.detach().numpy(), out1.detach().numpy(), c='r', label='Sigmoid')
    plt.plot(x1.detach().numpy(), x1_grad.detach().numpy(), c='r', linestyle='--' ,label='Gradient of Sigmoid')

    plt.plot(x2.detach().numpy(), out2.detach().numpy(), c='b', label='Softsign')
    plt.plot(x2.detach().numpy(), x2_grad.detach().numpy(), c='b', linestyle='--' ,label='Gradient of Softsign')

    plt.plot(x3.detach().numpy(), out3.detach().numpy(), c='g', label='Tanh')
    plt.plot(x3.detach().numpy(), x3_grad.detach().numpy(), c='g', linestyle='--' ,label='Gradient of Tanh')

    plt.xlim(-5, 5)
    plt.ylim(-3, 5)
    plt.legend(fontsize="20", loc='best')
    plt.grid(True)
    plt.show()     