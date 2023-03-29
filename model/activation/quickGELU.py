# (ref) https://github.com/hendrycks/GELUs
# (ref) https://github.com/OpenGVLab/UniFormerV2/blob/main/slowfast/models/uniformerv2_model.py
""" Quick GELU
"""
import matplotlib.pyplot as plt 

import torch 
import torch.nn as nn

class QuickGELU(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(1.702 * x)

if __name__ == "__main__": 
    x = torch.linspace(-5, 5, 1000)
    act = QuickGELU()
    out = act(x)

    # == Vis. == #
    plt.plot(x, out, c='r', label='Quick GELU')
    plt.xlim(-5, 5)
    plt.ylim(-3, 5)
    plt.legend(loc='best')
    plt.show() 