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
    x = torch.linspace(-5, 5, 1000)
    act = GELU()
    out = act(x)

    # == Vis. == #
    plt.plot(x, out, c='r', label='GELU')
    plt.xlim(-5, 5)
    plt.ylim(-3, 5)
    plt.legend(loc='best')
    plt.show() 