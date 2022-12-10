# (ref) https://github.com/sail-sg/metaformer/blob/main/metaformer_baselines.py
import torch
import torch.nn as nn 


class RandomMixing(nn.Module):
    def __init__(self, num_tokens=14*14, **kwargs):
        super().__init__()
        self.random_matrix = nn.parameter.Parameter(
            data=torch.softmax(torch.rand(num_tokens, num_tokens), dim=-1), 
            requires_grad=False)

    def forward(self, x):
        B, H, W, C = x.shape
        x = x.reshape(B, H*W, C)
        x = torch.einsum('mn, bnc -> bmc', self.random_matrix, x)
        x = x.reshape(B, H, W, C)
        return x




if __name__ == "__main__": 
    dim = 64 
    x = torch.randn(1, 7, 7, dim)   # (B, H, W, C)

    token_mixer = RandomMixing(num_tokens=7*7) 
    out = token_mixer(x)
    print(out.shape)
