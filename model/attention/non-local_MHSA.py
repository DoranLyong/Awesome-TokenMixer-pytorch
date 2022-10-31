# (ref) https://naokishibuya.medium.com/transformers-self-attention-1dc3a2719e0a
# (ref) https://github.com/rwightman/pytorch-image-models/blob/HEAD/timm/models/vision_transformer.py#L257

import torch 
import torch.nn as nn 

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


if __name__ == "__main__": 
    dim = 64
    x = torch.randn(1, 7*7, dim) # input 

    # 'dim' should be divisible by num_heads 
    heads = dim//4
    attn_block = Attention(dim=dim, num_heads=heads, qkv_bias=False, attn_drop=0., proj_drop=0.)

    output = attn_block(x)
    print(output.shape)