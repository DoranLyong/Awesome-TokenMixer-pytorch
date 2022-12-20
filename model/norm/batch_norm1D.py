""" Batch normalization (also called "batch scale") is a technique used to normalize the activations of the layers in a deep learning model, 
    specifically the mean and variance of the activations over a mini-batch of training examples.

    (ref) https://youtu.be/4gal2zIjm3M
    (ref) https://gaussian37.github.io/dl-concept-batchnorm/
"""
import matplotlib.pyplot as plt 

import torch
import torch.nn as nn

class BatchNorm1D(nn.Module):
    """ --num_features: the number of features(= dim)
    """    
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super(BatchNorm1D, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.gamma = nn.Parameter(torch.ones(num_features))
        self.beta = nn.Parameter(torch.zeros(num_features))
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))

    def forward(self, x):
        if self.training:
            # Compute the mean and variance of the mini-batch
            mean = x.mean(dim=0, keepdim=False)
            var = ((x - mean) ** 2).mean(dim=0, keepdim=False)

            # Update the running mean and variance
            self.running_mean.mul_(1 - self.momentum).add_(self.momentum * mean.data)
            self.running_var.mul_(1 - self.momentum).add_(self.momentum * var.data)
        else:
            mean = torch.tensor(self.running_mean).to(x.device)
            var = torch.tensor(self.running_var).to(x.device)

        # Normalize the activations
        return self.gamma * (x - mean) / (var + self.eps).sqrt() + self.beta


if __name__ == "__main__":
    x = torch.randn(64, 512) + 3

    batch_norm = BatchNorm1D(512)
    out = batch_norm(x)

    # == Vis. == #
    plt.hist(x.flatten().detach().numpy(), bins=100, label="Input")
    plt.hist(out.flatten().detach().numpy(), bins=100, label="Layer Scale")
    plt.legend(loc='best')
    plt.show()
