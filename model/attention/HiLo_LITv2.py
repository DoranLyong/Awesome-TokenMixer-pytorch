""" Code reference:
    - https://github.com/ziplab/litv2
    - https://arxiv.org/abs/2205.13213
    - https://github.com/Westlake-AI/openmixup/tree/main/configs/classification/imagenet/lit_v2/
"""
import math 

import torch 
import torch.nn as nn 



class HiLo(nn.Module):
    """ HiLo: High-Low Self-Attention(SA) Module
    """
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, 
                        attn_drop=0., proj_drop=0., window_size=2, alpha=0.5):
        super().__init__()

        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."
        head_dim = int(dim/num_heads)
        self.dim = dim

        # == SA heads in Lo-Fi 
        self.l_heads = int(num_heads * alpha)
        self.l_dim = self.l_heads * head_dim # token dimension in Lo-Fi

        # == SA heads in Hi-Fi
        self.h_heads = num_heads - self.l_heads
        self.h_dim = self.h_heads * head_dim # token dimension in Hi-Fi

        # == local window size. The `s` in the paper.
        self.ws = window_size
        