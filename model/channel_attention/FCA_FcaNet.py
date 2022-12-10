import torch 
import torch.nn as nn 






class FCA(nn.Module):
    """ Frequency Channel Attention (FCA); https://arxiv.org/abs/2012.11879 (ICCV2021)
        code implementation from : 
        https://github.com/cfzd/FcaNet
    """
    pass 





if __name__ == "__main__": 
    image_size = [52, 52]
    input = torch.rand(1, 64, *image_size)

    gca = FCA(channel=64, ratio=1)
    output = gca(input)
    print(output.shape)