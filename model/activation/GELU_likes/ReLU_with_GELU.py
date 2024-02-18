""" Combination of Squared ReLU and learnable Quick GELU
"""
import matplotlib.pyplot as plt 

import torch 
import torch.nn as nn

class GraphLU(nn.Module):
    """ Inspired from https://arxiv.org/pdf/2308.00574.pdf
    """
    def __init__(self, alpha=1.702, inplace=False):        
        super(GraphLU, self).__init__()
        self.alpha = torch.nn.Parameter(torch.tensor([alpha]), requires_grad=True)  # 가중치를 학습 가능한 매개변수로 설정
        self.relu = nn.ReLU(inplace=inplace)

    def forward(self, x):
        y = x - self.relu(x) # for negative values 
        return self.relu(x) + y * torch.sigmoid(self.alpha * y)


if __name__ == "__main__": 
    x0 = torch.linspace(-5, 5, 1000, requires_grad=True) # requires_grad=True로 설정하여 자동 미분 가능
    x1 = torch.linspace(-5, 5, 1000, requires_grad=True)
    x2 = torch.linspace(-5, 5, 1000, requires_grad=True)


    # Activation instance 
    act = GraphLU(alpha=1.702) # same with the GELU
    out0 = act(x0)
    out0.backward(torch.ones_like(x0)) # out의 각 요소에 대해 역전파 수행
    x0_grad = x0.grad 


    act_1 = GraphLU(alpha=2.5) 
    out1 = act_1(x1)
    out1.backward(torch.ones_like(x1)) 
    x1_grad = x1.grad     

    act_2 = GraphLU(alpha=1.0) 
    out2 = act_2(x2)
    out2.backward(torch.ones_like(x2)) 
    x2_grad = x2.grad         

    # == Vis. == #
    plt.plot(x0.detach().numpy(), out0.detach().numpy(), c='r', label=f'alpha={1.702} (same with GELU)')
    plt.plot(x0.detach().numpy(), x0_grad.detach().numpy(), c='r', linestyle='--' ,label='Gradient')    

    plt.plot(x1.detach().numpy(), out1.detach().numpy(), c='b', label=f'alpha={2.5}')
    plt.plot(x1.detach().numpy(), x1_grad.detach().numpy(), c='b', linestyle='--' ,label='Gradient')    

    plt.plot(x2.detach().numpy(), out2.detach().numpy(), c='g', label=f'alpha={1.0}')
    plt.plot(x2.detach().numpy(), x2_grad.detach().numpy(), c='g', linestyle='--' ,label='Gradient')        

    plt.title("GELU Variants")
    plt.xlabel("Input Value (x)")
    plt.ylabel("Output Value")
    plt.legend(fontsize="20", loc='best')
    plt.grid(True)
    plt.show()

