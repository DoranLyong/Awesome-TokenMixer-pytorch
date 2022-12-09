""" Code reference: 
    - FNet: Mixing Tokens with Fourier Transforms (https://arxiv.org/abs/2105.03824)
    - https://github.com/labmlai/annotated_deep_learning_paper_implementations
    - https://github.com/erksch/fnet-pytorch
    - https://github.com/rishikksh20/FNet-pytorch
    - https://github.com/facebookresearch/xformers/blob/main/xformers/components/attention/fourier_mix.py
    - https://github.com/raoyongming/GFNet
"""

import torch 
import torch.nn as nn 


def compare(x1, x2):
    try: 
        torch.testing.assert_close(x1, x2, check_stride=False)
        print("Equal")
    except: 
        print("Not equal")


dim = 2
seq_len = 16  # HW 
x = torch.randn(1, seq_len, dim)  # (batch_size, seq_len, hidden_size)

X1 = torch.fft.fft2(x).real
X2 = torch.fft.rfft2(x, dim=(1,2)).real
X3 = torch.fft.rfft2(x, norm='ortho').real
X4 = torch.fft.fft(torch.fft.fft(x, dim=-1), dim=-2).real


compare(X1, X2)
compare(X1, X3)
compare(X1, X4)