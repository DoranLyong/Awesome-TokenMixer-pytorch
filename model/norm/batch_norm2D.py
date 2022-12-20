""" Batch normalization (also called "batch scale") is a technique used to normalize the activations of the layers in a deep learning model, 
    specifically the mean and variance of the activations over a mini-batch of training examples.

    (ref) https://youtu.be/4gal2zIjm3M
    (ref) https://gaussian37.github.io/dl-concept-batchnorm/
"""
import matplotlib.pyplot as plt 

import torch
import torch.nn as nn

class BatchNorm2d(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.gamma = nn.Parameter(torch.ones(num_features))
        self.beta = nn.Parameter(torch.zeros(num_features))
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))

    def forward(self, x):
        # If the input tensor has more than 2 dimensions, reshape it to (batch_size, num_features, h, w)
        if x.dim() != 4:
            x = x.view((1, x.size(0), -1, 1, 1))
            x = x.squeeze(0)

        if self.training:
            # Compute the mean and variance of the mini-batch

            
            # Compute the running mean and variance
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * x.mean(dim=(0, 2, 3))
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * x.var(dim=(0, 2, 3))
        
        else: 
            pass 

        # Normalize the input tensor
        x = (x - self.running_mean) / (self.running_var + self.eps).sqrt()
        
        # Apply the scaling and shifting parameters
        x = self.gamma * x + self.beta
        
        return x

