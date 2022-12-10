# (ref) https://github.com/sail-sg/metaformer/blob/main/metaformer_baselines.py
import torch
import torch.nn as nn


class Pooling(nn.Module):
    """
    Implementation of pooling for PoolFormer: https://arxiv.org/abs/2111.11418
    Modfiled for [B, H, W, C] input
    """
    def __init__(self, pool_size=3, **kwargs):
        super().__init__()
        self.pool = nn.AvgPool2d(
            pool_size, stride=1, padding=pool_size//2, count_include_pad=False)

    def forward(self, x):
        y = x.permute(0, 3, 1, 2)
        y = self.pool(y)
        y = y.permute(0, 2, 3, 1)
        return y - x


if __name__ == "__main__":
    dim = 64 
    x = torch.randn(1, dim, 7, 7) # (B, C, H, W)

    token_mixer = Pooling(pool_size=3)
    out = token_mixer(x)
    print(out.shape)
