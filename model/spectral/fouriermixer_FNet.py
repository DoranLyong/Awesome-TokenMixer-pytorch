""" Code reference: 
    - FNet: Mixing Tokens with Fourier Transforms (https://arxiv.org/abs/2105.03824)
    - https://github.com/labmlai/annotated_deep_learning_paper_implementations
    - https://github.com/erksch/fnet-pytorch
    - https://github.com/rishikksh20/FNet-pytorch
    - https://github.com/facebookresearch/xformers/blob/main/xformers/components/attention/fourier_mix.py
    - https://github.com/raoyongming/GFNet
    - https://github.com/google-research/google-research/tree/master/f_net
"""
import torch 
import torch.nn as nn 
from torch.cuda.amp import autocast



class FourierMixer(nn.Module):
    """ Args: 
        --dim (int): dimension of the input
        --seq_len (int): length of the sequence 
        --dropout (float): dropout rate 
    """
    def __init__(self, dim=2, seq_len=16, **kwargs):
        super().__init__()
        self.complex_weight = nn.Parameter(torch.randn(seq_len, (dim//2+1), 2, dtype=torch.float32) * 0.02)

    def forward(self, x):
        B, L, C = x.shape

        with autocast(enabled=False):
            # Guard against autocast / fp16, not supported by torch.fft.fft2
            X = torch.fft.rfft2(x, dim=(1,2), norm='ortho')
            weight = torch.view_as_complex(self.complex_weight)
            X = X * weight
        
        attn = torch.fft.irfft2(X, s=(L,C), dim=(1, 2), norm='ortho')
        return attn 



if __name__ == "__main__":
    dim = 64 
    seq_len = 16 # HW or token length 
    x = torch.randn(1, seq_len, dim)  # (batch_size, seq_len, hidden_size)

    block = FourierMixer(dim=dim, seq_len=seq_len)
    out = block(x)
    print(out.shape)



