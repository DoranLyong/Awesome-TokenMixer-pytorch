# (ref) https://github.com/raoyongming/HorNet/blob/master/hornet.py
from functools import partial

import torch 
import torch.nn as nn 
import torch.nn.functional as F

from timm.models.layers import trunc_normal_

class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first. 
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with 
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs 
    with shape (batch_size, channels, height, width).
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError 
        self.normalized_shape = (normalized_shape, )
    
    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x

class GlobalLocalFilter(nn.Module):
    def __init__(self, dim, h=14, w=8):
        super().__init__()
        self.dw = nn.Conv2d(dim // 2, dim // 2, kernel_size=3, padding=1, bias=False, groups=dim // 2)
        self.complex_weight = nn.Parameter(torch.randn(dim // 2, h, w, 2, dtype=torch.float32) * 0.02)
        trunc_normal_(self.complex_weight, std=.02)
        self.pre_norm = LayerNorm(dim, eps=1e-6, data_format='channels_first')
        self.post_norm = LayerNorm(dim, eps=1e-6, data_format='channels_first')

    def forward(self, x):
        x = self.pre_norm(x)
        x1, x2 = torch.chunk(x, 2, dim=1)
        x1 = self.dw(x1)

        x2 = x2.to(torch.float32)
        B, C, a, b = x2.shape
        x2 = torch.fft.rfft2(x2, dim=(2, 3), norm='ortho')

        weight = self.complex_weight
        if not weight.shape[1:3] == x2.shape[2:4]:
            weight = F.interpolate(weight.permute(3,0,1,2), size=x2.shape[2:4], mode='bilinear', align_corners=True).permute(1,2,3,0)

        weight = torch.view_as_complex(weight.contiguous())

        x2 = x2 * weight
        x2 = torch.fft.irfft2(x2, s=(a, b), dim=(2, 3), norm='ortho')

        x = torch.cat([x1.unsqueeze(2), x2.unsqueeze(2)], dim=2).reshape(B, 2 * C, a, b)
        x = self.post_norm(x)
        return x

def DWConv(dim, kernel, bias):
    return nn.Conv2d(dim, dim, kernel_size=kernel, padding=(kernel-1)//2 ,bias=bias, groups=dim)


class gnconv(nn.Module): 
    def __init__(self, dim, order=5, gflayer=None, h=14, w=8, s=1.):
        super().__init__()
        self.order = order

        self.dims = [dim // 2 ** i for i in range(order)]
        self.dims.reverse() # reverse order 

        self.proj_in = nn.Conv2d(dim, dim*2, 1) # pw-conv for channel expansion

        if gflayer is None:
            self.dwconv = DWConv(sum(self.dims), 7, True) # 7x7 DW-conv 
        else: 
            # Gobal filter layer 
            pass 

        self.proj_out = nn.Conv2d(dim, dim, 1) # pw-conv for channel fusion 

        self.pws = nn.ModuleList(
            [nn.Conv2d(self.dims[i], self.dims[i+1], 1) # 1x1 pw-conv
            for i in range(order-1)]
        )

        self.scale = s 
        print(f"[gnconv] {order} order with dims= {self.dims} scale={self.scale:.4f}")

    def forward(self, x):
        B,C,H,W = x.size() 

        # == (1) inverted bottleneck - start == #
        fused_x = self.proj_in(x) # channel expansion C -> 2C
        pwa, abc = torch.split(fused_x, (self.dims[0], sum(self.dims)), dim=1)  # in: [2C, H, W]
                                                                                # out: [C//2, H , W] & [C+C//2, H, W]
        # == (2) DW-conv == # 
        dw_abc = self.dwconv(abc) * self.scale # value scaling; [C+C//2, H, W] -> [C+C//2, H, W]
        dw_list = torch.split(dw_abc, self.dims, dim=1) # (e.g) dims=[32,64] => dw_list = [(32,H,W), (64,H,W)]
        
        # == element-wise mul for recursive gating == # 
        x = pwa * dw_list[0] # [dims[0], H, W] * [dims[0], H, W] 

        for i in range(self.order-1): 
            x = self.pws[i](x) * dw_list[i+1] # [dims[i+1], H, W] * [dims[i+1], H, W]

        # == (3) inverted bottleneck - end == #
        x = self.proj_out(x) # channel fusion C -> C 
        return x 


if __name__ == "__main__":
    dim=64
    x = torch.randn(1,dim,7,7)  # stage-1

    s = 1.0/3.0 # value scaling factor
    token_mixer = gnconv(dim=64, order=3, s=s)


    out = token_mixer(x)
    print(out.size())



