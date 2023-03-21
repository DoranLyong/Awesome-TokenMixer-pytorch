# (ref) https://github.com/hkzhang91/parc-net/blob/HEAD/ParC_ConvNets/ParC_convnext.py
import torch 
import torch.nn as nn 
import torch.nn.functional as F

from timm.models.layers import trunc_normal_, DropPath


class ParC_operator(nn.Module):
    def __init__(self, dim, type, global_kernel_size, use_pe=True):
        super().__init__()
        self.type = type  # H or W
        self.dim = dim
        self.use_pe = use_pe
        self.global_kernel_size = global_kernel_size
        self.kernel_size = (global_kernel_size, 1) if self.type == 'H' else (1, global_kernel_size)
        self.gcc_conv = nn.Conv2d(dim, dim, kernel_size=self.kernel_size, groups=dim)

        if use_pe:
            if self.type=='H':
                self.pe = nn.Parameter(torch.randn(1, dim, self.global_kernel_size, 1))
            elif self.type=='W':
                self.pe = nn.Parameter(torch.randn(1, dim, 1, self.global_kernel_size))
            trunc_normal_(self.pe, std=.02)

    def forward(self, x):
        if self.use_pe:
            x = x + self.pe.expand(1, self.dim, self.global_kernel_size, self.global_kernel_size)

        x_cat = torch.cat((x, x[:, :, :-1, :]), dim=2) if self.type == 'H' else torch.cat((x, x[:, :, :, :-1]), dim=3)
        x = self.gcc_conv(x_cat)

        return x


class ParC_ConvNext_Block(nn.Module):
    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6, global_kernel_size=14, use_pe=True):
        super().__init__()
        self.gcc_H = ParC_operator(dim//2, 'H', global_kernel_size, use_pe)
        self.gcc_W = ParC_operator(dim//2, 'W', global_kernel_size, use_pe)
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)  # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)),
                                  requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x_H, x_W = torch.chunk(x, 2, dim=1)
        x_H, x_W = self.gcc_H(x_H), self.gcc_W(x_W)
        x = torch.cat((x_H, x_W), dim=1)
        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)

        x = input + self.drop_path(x)
        return x

class LayerNorm(nn.Module):

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x





if __name__ == "__main__": 
    dim = 96
    stages_rs=[56, 28, 14, 7]  # input resolutions of four stages
    x = torch.randn(1, dim, 56, 56) # input 

    ParC_block = ParC_ConvNext_Block(dim=dim, drop_path=0, layer_scale_init_value=1e-6, global_kernel_size=stages_rs[0], use_pe=True)
    output = ParC_block(x)
    print(output.shape)