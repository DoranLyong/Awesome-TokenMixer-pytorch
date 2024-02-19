""" Searching for Activation Functions (2017)

    [Ref]
    - https://arxiv.org/abs/1710.05941v2
    - https://paperswithcode.com/method/swish
"""
import matplotlib.pyplot as plt 

import torch 
import torch.nn as nn

class Swish(nn.Module):
    def __init__(self, beta=1.702):        
        super(Swish, self).__init__()
        self.beta = nn.Parameter(torch.tensor([beta]), requires_grad=True)  # 가중치를 학습 가능한 매개변수로 설정

    def forward(self, x):
        return x * torch.sigmoid(self.beta * x)

if __name__ == "__main__": 
    x0 = torch.linspace(-5, 5, 1000, requires_grad=True) # requires_grad=True로 설정하여 자동 미분 가능
    x1 = torch.linspace(-5, 5, 1000, requires_grad=True)
    x2 = torch.linspace(-5, 5, 1000, requires_grad=True)
    x3 = torch.linspace(-5, 5, 1000, requires_grad=True)


    # Activation instance 
    act = Swish(beta=1.702) # approximated GELU
    out0 = act(x0)
    out0.backward(torch.ones_like(x0)) # out의 각 요소에 대해 역전파 수행
    x0_grad = x0.grad 


    act_1 = Swish(beta=10.0) 
    out1 = act_1(x1)
    out1.backward(torch.ones_like(x1)) 
    x1_grad = x1.grad     

    act_2 = Swish(beta=1.0) 
    out2 = act_2(x2)
    out2.backward(torch.ones_like(x2)) 
    x2_grad = x2.grad         

    act_3 = Swish(beta=0.1) 
    out3 = act_3(x3)
    out3.backward(torch.ones_like(x3)) 
    x3_grad = x3.grad             

    # == Vis. == #
    plt.plot(x0.detach().numpy(), out0.detach().numpy(), c='r', label=f'beta={1.702} (approximated GELU)')
    plt.plot(x0.detach().numpy(), x0_grad.detach().numpy(), c='r', linestyle='--' ,label='Gradient')    

    plt.plot(x1.detach().numpy(), out1.detach().numpy(), c='b', label=f'beta={10.0} (almost ReLU)')
    plt.plot(x1.detach().numpy(), x1_grad.detach().numpy(), c='b', linestyle='--' ,label='Gradient')    

    plt.plot(x2.detach().numpy(), out2.detach().numpy(), c='g', label=f'beta={1.0} (Swish-1 a.k.a., SiLU)')
    plt.plot(x2.detach().numpy(), x2_grad.detach().numpy(), c='g', linestyle='--' ,label='Gradient')        

    plt.plot(x3.detach().numpy(), out3.detach().numpy(), c='skyblue', label=f'beta={0.1} (almost line)')
    plt.plot(x3.detach().numpy(), x3_grad.detach().numpy(), c='skyblue', linestyle='--' ,label='Gradient')     

    plt.title("Swish Variants")
    plt.xlabel("Input Value (x)")
    plt.ylabel("Output Value")
    plt.legend(fontsize="20", loc='best')
    plt.grid(True)
    plt.show()