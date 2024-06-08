# (ref) https://pytorch.org/vision/main/generated/torchvision.ops.deform_conv2d.html
"""
      Input (x)
         |
         v
  +--------------+
  | Offset Conv  | 
  | 2 * k * k    |
  +--------------+
         |
         v
     Offset
         |
         v
  +--------------+
  | Deformable   |
  | Convolution  |
  +--------------+
         |
         v
     Output
"""



import torch 
import torch.nn as nn 

import torchvision.ops as ops


class DeformableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0, dilation=1):
        super(DeformableConv2d, self).__init__()
        self.stride = stride
        self.padding = padding
        self.dilation = dilation

        # Offset convolution layer
        self.offset_conv = nn.Conv2d(in_channels, 2 * kernel_size * kernel_size, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation)
        
        # Deformable convolution weights and bias
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size))
        self.bias = nn.Parameter(torch.randn(out_channels))

    def forward(self, x):
        # Calculate offsets
        offset = self.offset_conv(x)
        
        # Apply deformable convolution
        return ops.deform_conv2d(x, offset, self.weight, self.bias, stride=self.stride, padding=self.padding, dilation=self.dilation)












if __name__ == "__main__":
    dim = 96
    H,W= 56, 56
    x = torch.randn(128, dim, H, W) # input 

    kernel_size = 7
    dilation = 2
    stride = 1
    padding = (dilation*(kernel_size-1)-(stride-1)*W+1)//2

    token_mixer = DeformableConv2d(
                    in_channels=dim, 
                    out_channels=dim, 
                    kernel_size=kernel_size, 
                    stride=stride,
                    padding=padding, # To keep the output dimensions the same as the input dimensions                    
                    dilation = dilation,  # Increased dilation rate
                    )
    
    out = token_mixer(x)
    print(out.shape)
