""" MLP-Mixer
    (ref1) https://github.com/sail-sg/metaformer/blob/main/metaformer_baselines.py
    (ref2) https://arxiv.org/abs/2105.01601
""" 
import torch 
import torch.nn as nn 

from timm.models.layers.helpers import to_2tuple

class Mlp(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks
    Mostly copied from timm.
    """
    def __init__(self, dim, mlp_ratio=4, out_features=None, act_layer=nn.ReLU, drop=0., bias=False, **kwargs):
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



if __name__ == "__main__": 
    dim = 64 
    x = torch.randn(1, 7, 7, dim) # (B, H, W, C)

    token_mixer = Mlp(dim=dim)
    out = token_mixer(x)
    print(out.shape)