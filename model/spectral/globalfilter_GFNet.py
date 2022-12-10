""" (ref) https://medium.com/optalysys/attention-fourier-transforms-a-giant-leap-in-transformer-efficiency-58ca0b3c4164
    (ref) https://github.com/pkumivision/FFC/blob/main/model_zoo/ffc.py
"""

import math 

import torch 
import torch.nn as nn 

from timm.models.layers import trunc_normal_


class GlobalFilter(nn.Module):
    """ Original implementation refer to (GFNet, NeurIPS 2021):
        - https://github.com/raoyongming/GFNet
        - https://gfnet.ivg-research.xyz/
    """
    def __init__(self, dim, h=14, w=8):
        super().__init__()
        self.complex_weight = nn.Parameter(torch.randn(h, w, dim, 2, dtype=torch.float32) * 0.02) # (h, w, dim, Re/Im)

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



class FourierUnit(nn.Module):
    """ Custom implementation
    """
    def __init__(self, dim, h=14, w=8):
        super().__init__()
        self.complex_weight = nn.Parameter(torch.randn(dim, h, w, 2, dtype=torch.float32) * 0.02) # (C, H, W, real+imag)
        trunc_normal_(self.complex_weight, std=.02) # init complex weight

    def forward(self, x):
        assert x.ndim == 4
        bias = x.clone()
        B,C,H,W = x.size() 
        
        # == FFT == # 
        x = x.to(torch.float32)
        x = torch.fft.rfft2(x, dim=(2, 3), norm='ortho') # (B, C, H, W//2+1)

        # == Spectral transform == # 
        weight = torch.view_as_complex(self.complex_weight).type(torch.complex64) # ((B, C, H, W//2+1); torch.complex64 for preventing half-precision error of cuFFT 
        x = x * weight

        # == Inverse FFT == #
        x = torch.fft.irfft2(x, s=(H, W), dim=(2, 3), norm='ortho') # (B, C, H, W)
        return x + bias 



if __name__ == "__main__":

    # == Original implementation == #
    dim = 64 
    H, W = (14, 14)
    x = torch.randn(1, H*W, dim)  # (batch_size, seq_len, hidden_size)

    block = GlobalFilter(dim, h=H, w=(W//2+1)) # 
    out = block(x)
    print(out.shape)


    # == Custom implementation == #
    x2 = torch.randn(1, dim, H, W)  # (batch_size, hidden_size, H, W)

    block2 = FourierUnit(dim, h=H, w=(W//2+1))
    out2 = block2(x2)
    print(out2.shape)


