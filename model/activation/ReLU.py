# (ref1) https://github.com/freeknortier/Activation-Functions-Visualizations/blob/master/ReLu.py
# (ref2) https://pytorch.org/docs/stable/_modules/torch/nn/modules/activation.html#ReLU
# (ref3) https://alexiej.github.io/deepnn/#activation-functions

import matplotlib.pyplot as plt 

import torch 
import torch.nn as nn 
import torch.nn.functional as F


class ReLU(nn.Module):
    def __init__(self, inplace):
        super(ReLU, self).__init__()
        self.relu = nn.ReLU(inplace=inplace)

    def forward(self, x):
        return self.relu(x)



if __name__ == "__main__": 
    x = torch.linspace(-5, 5, 1000)
    relu = ReLU(inplace=False)
    out = relu(x)

    
    # == Vis. == #
    plt.plot(x, out, c='r', label='ReLU')
    plt.xlim(-5, 5)
    plt.ylim(-3, 5)
    plt.legend(loc='best')
    plt.show() 