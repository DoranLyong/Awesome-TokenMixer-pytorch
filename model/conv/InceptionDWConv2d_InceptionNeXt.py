# (ref) https://github.com/sail-sg/inceptionnext/blob/main/models/inceptionnext.py
import torch 
import torch.nn as nn 

from timm.models.layers import trunc_normal_, DropPath



class InceptionDWConv2d(nn.Module):
    """ Inception depthweise convolution
    """
    def __init__(self, in_channels, square_kernel_size=3, band_kernel_size=11, branch_ratio=0.125):
        super().__init__()
        
        gc = int(in_channels * branch_ratio) # channel numbers of a convolution branch
        self.dwconv_hw = nn.Conv2d(gc, gc, square_kernel_size, padding=square_kernel_size//2, groups=gc)
        self.dwconv_w = nn.Conv2d(gc, gc, kernel_size=(1, band_kernel_size), padding=(0, band_kernel_size//2), groups=gc)
        self.dwconv_h = nn.Conv2d(gc, gc, kernel_size=(band_kernel_size, 1), padding=(band_kernel_size//2, 0), groups=gc)
        self.split_indexes = (in_channels - 3 * gc, gc, gc, gc)
        
    def forward(self, x):
        x_id, x_hw, x_w, x_h = torch.split(x, self.split_indexes, dim=1)
        return torch.cat(
            (x_id, self.dwconv_hw(x_hw), self.dwconv_w(x_w), self.dwconv_h(x_h)), 
            dim=1,
        )
    

if __name__ == "__main__": 
    dim = 96
    x = torch.randn(128, dim, 56, 56) # input 

    InceptionBlock = InceptionDWConv2d(dim, square_kernel_size=3, band_kernel_size=11, branch_ratio=0.125) # 0.125 = 1/8
    
    output = InceptionBlock(x)
    print(output.shape)