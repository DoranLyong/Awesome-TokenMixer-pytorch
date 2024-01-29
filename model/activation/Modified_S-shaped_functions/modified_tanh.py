import matplotlib.pyplot as plt 

import torch 
import torch.nn as nn 
import torch.nn.functional as F

class ModifiedTanh(nn.Module):
    def __init__(self, alpha=0.5):
        super(ModifiedTanh, self).__init__()
        self.alpha = torch.nn.Parameter(torch.tensor([alpha]))  # alpha 매개변수

    def forward(self, x):
        return torch.tanh(self.alpha * x)



if __name__ == "__main__":
    x0 = torch.linspace(-5, 5, 1000, requires_grad=True) # requires_grad=True로 설정하여 자동 미분 가능
    x1 = torch.linspace(-5, 5, 1000, requires_grad=True) 
    x2 = torch.linspace(-5, 5, 1000, requires_grad=True)
    x3 = torch.linspace(-5, 5, 1000, requires_grad=True)




    # Activation instance 
    modified_tanh = ModifiedTanh(alpha=0.5)  
    out0 = modified_tanh(x0)
    out0.backward(torch.ones_like(x0)) # out의 각 요소에 대해 역전파 수행
    x0_grad = x0.grad

    modified_tanh_a1 = ModifiedTanh(alpha=1.0)  # baseline like; nn.Softsign() 
    out1 = modified_tanh_a1(x1)
    out1.backward(torch.ones_like(x1)) # out의 각 요소에 대해 역전파 수행
    x1_grad = x1.grad

    modified_tanh_a3 = ModifiedTanh(alpha=2.0)
    out2 = modified_tanh_a3(x2)
    out2.backward(torch.ones_like(x2)) # out의 각 요소에 대해 역전파 수행
    x2_grad = x2.grad

    modified_tanh_b3 = ModifiedTanh(alpha=0.25)
    out3 = modified_tanh_b3(x3)
    out3.backward(torch.ones_like(x3)) # out의 각 요소에 대해 역전파 수행
    x3_grad = x3.grad



    # == Vis. == 
    plt.plot(x0.detach().numpy(), out0.detach().numpy(), c='skyblue', label=f'Tanh alpha={0.5}')
    plt.plot(x0.detach().numpy(), x0_grad.detach().numpy(), c='skyblue', linestyle='--' ,label='Gradient')

    plt.plot(x1.detach().numpy(), out1.detach().numpy(), c='r', label=f'Tanh  alpha={1.0} baseline')
    plt.plot(x1.detach().numpy(), x1_grad.detach().numpy(), c='r', linestyle='--' ,label='Gradient')

    plt.plot(x2.detach().numpy(), out2.detach().numpy(), c='b', label=f'Tanh  alpha={2.0}')
    plt.plot(x2.detach().numpy(), x2_grad.detach().numpy(), c='b', linestyle='--' ,label='Gradient')

    plt.plot(x3.detach().numpy(), out3.detach().numpy(), c='g', label=f'Tanh  alpha={0.25}')
    plt.plot(x3.detach().numpy(), x3_grad.detach().numpy(), c='g', linestyle='--' ,label='Gradient')


    plt.title("Modified Tanh Activation Function")
    plt.xlabel("Input Value (x)")
    plt.ylabel("Output Value")
    plt.legend(fontsize="20", loc='best')
    plt.grid(True)
    plt.show()
