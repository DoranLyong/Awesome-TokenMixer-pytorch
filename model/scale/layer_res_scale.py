""" (ref) https://github.com/sail-sg/metaformer/blob/main/metaformer_baselines.py
    (ref) https://arxiv.org/abs/2210.13452
"""

import torch 
import torch.nn as nn 

from timm.models.layers import DropPath
from timm.models.layers.helpers import to_2tuple


class Scale(nn.Module):
    """
    Scale vector by element multiplications.
    """
    def __init__(self, dim, init_value=1.0, trainable=True):
        super().__init__()
        self.scale = nn.Parameter(init_value * torch.ones(dim), requires_grad=trainable)

    def forward(self, x):
        return x * self.scale


class TokenMixer(nn.Module): 
    def __init__(self):
        super().__init__()
        self.mixer = nn.Identity()

    def forward(self, x):
        x = self.mixer(x)
        return x 


class Mlp(nn.Module):
    """ MLP as used in MetaFormer models, eg Transformer, MLP-Mixer, PoolFormer, MetaFormer baslines and related networks.
    Mostly copied from timm.
    """
    def __init__(self, dim, mlp_ratio=4, out_features=None, act_layer=nn.GELU, drop=0., bias=False, **kwargs):
        super().__init__()
        in_features = dim
        out_features = out_features or in_features
        hidden_features = int(mlp_ratio * in_features)
        drop_probs = to_2tuple(drop)

        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop_probs[0])
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias)
        self.drop2 = nn.Dropout(drop_probs[1])

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x



class Block(nn.Module): 
    def __init__(self, dim=64, token_mixer=TokenMixer, mlp=Mlp, norm_layer=nn.LayerNorm, 
                drop=0., drop_path=0., layer_scale_init_value=None, res_scale_init_value=None):
        super().__init__()
        
        # == Token Mixer == # 
        self.norm1 = norm_layer(dim)
        self.token_mixer = token_mixer()
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.layer_scale1 = Scale(dim=dim, init_value=layer_scale_init_value) if layer_scale_init_value else nn.Identity()
        self.res_scale1 = Scale(dim=dim, init_value=res_scale_init_value) if res_scale_init_value else nn.Identity()

        # == FFN == #   
        self.norm2 = norm_layer(dim)  
        self.mlp = mlp(dim=dim, drop=drop)
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.layer_scale2 = Scale(dim=dim, init_value=layer_scale_init_value) if layer_scale_init_value else nn.Identity()
        self.res_scale2 = Scale(dim=dim, init_value=res_scale_init_value) if res_scale_init_value else nn.Identity()


    def forward(self, x):
        x = self.res_scale1(x) + self.layer_scale1(self.drop_path1(self.token_mixer(self.norm1(x))))
        x = self.res_scale2(x) + self.layer_scale2(self.drop_path2(self.mlp(self.norm2(x))))
        return x 




if __name__ == "__main__": 
    dim = 64
    seq_len = 16  # HW or time stamp 
    x = torch.randn(1, seq_len, dim)  # (batch_size, seq_len, hidden_size)

    block = Block(dim=dim, layer_scale_init_value=1e-5, res_scale_init_value=None)
    out = block(x)
    print(out.shape)