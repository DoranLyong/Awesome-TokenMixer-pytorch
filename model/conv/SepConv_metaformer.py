# (ref) https://github.com/sail-sg/metaformer/blob/main/metaformer_baselines.py
import torch
import torch.nn as nn


class SepConv(nn.Module):
    r"""
    Inverted separable convolution from MobileNetV2: https://arxiv.org/abs/1801.04381.
    """
    def __init__(self, dim, expansion_ratio=2, act1_layer=nn.ReLU, act2_layer=nn.Identity, 
        bias=False, kernel_size=7, padding=3, **kwargs, ):
        super().__init__()
        med_channels = int(expansion_ratio * dim) 
        self.pwconv1 = nn.Linear(dim, med_channels, bias=bias) # channel expansion 
        self.act1 = act1_layer()

        self.dwconv = nn.Conv2d( med_channels, med_channels, kernel_size=kernel_size,
                                padding=padding, groups=med_channels, bias=bias) # depthwise conv
        self.act2 = act2_layer()
        self.pwconv2 = nn.Linear(med_channels, dim, bias=bias)

    def forward(self, x):
        x = self.pwconv1(x)
        x = self.act1(x)
        x = x.permute(0, 3, 1, 2)
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)
        x = self.act2(x)
        x = self.pwconv2(x)
        return x

if __name__ == "__main__": 
    dim = 64 
    x = torch.randn(1, 7, 7, dim) # (B, H, W, C)

    token_mixer = SepConv(dim=dim)
    out = token_mixer(x)
    print(out.shape)