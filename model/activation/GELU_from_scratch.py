""" GELU (Gaussian Error Linear Units)
    It has been shown to improve the performance of deep learning models in a variety of tasks.
    : https://arxiv.org/abs/1606.08415
"""
import matplotlib.pyplot as plt 

import torch 


def gelu(x): 
    return 0.5 * x * (1 + torch.tanh(x * (1 + 0.044715 * x * x)))




if __name__ == "__main__": 
    x = torch.linspace(-5, 5, 1000)
    out = gelu(x)

    # == Vis. == #
    plt.plot(x, out, c='r', label='GELU')
    plt.xlim(-5, 5)
    plt.ylim(-3, 5)
    plt.legend(loc='best')
    plt.show()
