""" the layer normalization (also called "layer scale") is a technique used to normalize the activations of the layers. 
    It helps to stabilize the training process and improve the generalization performance of the model.
"""
import matplotlib.pyplot as plt 

import torch
import torch.nn as nn

class LayerScale(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super(LayerScale, self).__init__()
        self.gamma = nn.Parameter(torch.ones(dim))
        self.beta = nn.Parameter(torch.zeros(dim))
        self.eps = eps

    def forward(self, x):
        """ x : (B, N, D)
        """
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta




if __name__ == "__main__": 
    x = torch.randn(64, 512) + 3

    layer_scale = LayerScale(512)
    out = layer_scale(x)

    # == Vis. == #
    plt.hist(x.flatten().detach().numpy(), bins=100, label="Input")
    plt.hist(out.flatten().detach().numpy(), bins=100, label="Layer Scale")
    plt.legend(loc='best')
    plt.show()