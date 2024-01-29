import matplotlib.pyplot as plt 

import torch 
import torch.nn as nn 
import torch.nn.functional as F

class ModifiedSoftsign(nn.Module):
    def __init__(self, alpha=1.0):
        super(ModifiedSoftsign, self).__init__()
        self.alpha = torch.nn.Parameter(torch.tensor([alpha]))  # 가중치를 학습 가능한 매개변수로 설정

    def forward(self, x):
        x = self.alpha * x
        return x / (1 + torch.abs(x))



if __name__ == "__main__":
    x0 = torch.linspace(-5, 5, 1000, requires_grad=True) # requires_grad=True로 설정하여 자동 미분 가능
    x1 = torch.linspace(-5, 5, 1000, requires_grad=True) 
    x2 = torch.linspace(-5, 5, 1000, requires_grad=True)
    x3 = torch.linspace(-5, 5, 1000, requires_grad=True)




    # Activation instance 
    modified_softsign = ModifiedSoftsign(alpha=0.5)  
    out0 = modified_softsign(x0)
    out0.backward(torch.ones_like(x0)) # out의 각 요소에 대해 역전파 수행
    x0_grad = x0.grad

    modified_softsign_a1 = ModifiedSoftsign(alpha=1.0)  # baseline like; nn.Softsign() 
    out1 = modified_softsign_a1(x1)
    out1.backward(torch.ones_like(x1)) # out의 각 요소에 대해 역전파 수행
    x1_grad = x1.grad

    modified_softsign_a3 = ModifiedSoftsign(alpha=2.0)
    out2 = modified_softsign_a3(x2)
    out2.backward(torch.ones_like(x2)) # out의 각 요소에 대해 역전파 수행
    x2_grad = x2.grad

    modified_softsign_b3 = ModifiedSoftsign(alpha=3.0)
    out3 = modified_softsign_b3(x3)
    out3.backward(torch.ones_like(x3)) # out의 각 요소에 대해 역전파 수행
    x3_grad = x3.grad



    # == Vis. == 
    plt.plot(x0.detach().numpy(), out0.detach().numpy(), c='skyblue', label=f'Softsign alpha={0.5}')
    plt.plot(x0.detach().numpy(), x0_grad.detach().numpy(), c='skyblue', linestyle='--' ,label='Gradient')

    plt.plot(x1.detach().numpy(), out1.detach().numpy(), c='r', label=f'Softsign  alpha={1.0} baseline')
    plt.plot(x1.detach().numpy(), x1_grad.detach().numpy(), c='r', linestyle='--' ,label='Gradient')

    plt.plot(x2.detach().numpy(), out2.detach().numpy(), c='b', label=f'Softsign  alpha={2.0}')
    plt.plot(x2.detach().numpy(), x2_grad.detach().numpy(), c='b', linestyle='--' ,label='Gradient')

    plt.plot(x3.detach().numpy(), out3.detach().numpy(), c='g', label=f'Softsign  alpha={3.0}')
    plt.plot(x3.detach().numpy(), x3_grad.detach().numpy(), c='g', linestyle='--' ,label='Gradient')


    plt.title("Modified Softsign Activation Function")
    plt.xlabel("Input Value (x)")
    plt.ylabel("Output Value")
    plt.legend(fontsize="20", loc='best')
    plt.grid(True)
    plt.show()
