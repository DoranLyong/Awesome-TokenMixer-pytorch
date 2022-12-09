""" Code reference:
    - https://github.com/raoyongming/GFNet
"""
import math 

import torch 
import torch.nn as nn 
from torch.cuda.amp import autocast



class GlobalFilter(nn.Module):
    def __init__(self, dim, h=14, w=8):
        super().__init__()
        self.complex_weight = nn.Parameter(torch.randn(h, w, dim, 2, dtype=torch.float32) * 0.02)

    def forward(self, x, spatial_size=None):
        B, N, C = x.shape
        if spatial_size is None:
            a = b = int(math.sqrt(N))
        else:
            a, b = spatial_size

        x = x.view(B, a, b, C)

        x = x.to(torch.float32)

        x = torch.fft.rfft2(x, dim=(1, 2), norm='ortho')
        weight = torch.view_as_complex(self.complex_weight)
        x = x * weight
        x = torch.fft.irfft2(x, s=(a, b), dim=(1, 2), norm='ortho')

        x = x.reshape(B, N, C)

        return x



if __name__ == "__main__":
    dim = 64 
    H, W = (14, 14)
    x = torch.randn(1, H*W, dim)  # (batch_size, seq_len, hidden_size)

    block = GlobalFilter(dim, h=H, w=(W//2+1))
    out = block(x)
    print(out.shape)
