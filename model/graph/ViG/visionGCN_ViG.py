""" (ref) https://github.com/huawei-noah/Efficient-AI-Backbones/tree/master/vig_pytorch
"""

import torch 
import torch.nn as nn 
from torch.nn import Sequential as Seq

from timm.models.layers import DropPath, to_2tuple, trunc_normal_

from gcn_lib import Grapher, act_layer



class FFN(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act='relu', drop_path=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Sequential(
            nn.Conv2d(in_features, hidden_features, 1, stride=1, padding=0),
            nn.BatchNorm2d(hidden_features),
        )
        self.act = act_layer(act)
        self.fc2 = nn.Sequential(
            nn.Conv2d(hidden_features, out_features, 1, stride=1, padding=0),
            nn.BatchNorm2d(out_features),
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        shortcut = x
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.drop_path(x) + shortcut
        return x    #.reshape(B, C, N, 1)



class Downsample(nn.Module):
    """ Convolution-based downsample
    """
    def __init__(self, in_dim=3, out_dim=768):
        super().__init__()        
        self.conv = nn.Sequential(
            nn.Conv2d(in_dim, out_dim, 3, stride=2, padding=1),
            nn.BatchNorm2d(out_dim),
        )

    def forward(self, x):
        x = self.conv(x)
        return x





if __name__ == "__main__":

    H, W = (224//4, 224//4)
    HW = H*W 

    # == Build model == #
    k = 9 # neighbor num (default:9)
    blocks = [2,2,6,2]
    n_blocks = sum(blocks)  # number of basic blocks in the backbone
    channels = [80, 160, 400, 640] # number of channels of deep features

    
    drop_path = 0
    reduce_ratios = [4, 2, 1, 1]
    dpr = [x.item() for x in torch.linspace(0, drop_path, n_blocks)]  # stochastic depth decay rule
    num_knn = [int(x.item()) for x in torch.linspace(k, k, n_blocks)]  # number of knn's k    
    max_dilation = 49 // max(num_knn)


    vig_block = nn.ModuleList([])
    idx=0

    for i in range(len(blocks)):
        if i > 0:
            vig_block.append(Downsample(channels[i-1], channels[i]))
            HW = HW//4 
        for j in range(blocks[i]):
            vig_block += [
                Seq(Grapher(in_channels=channels[i], kernel_size=num_knn[idx], dilation=min(idx // 4 + 1, max_dilation),  conv='mr', act='gelu',
                            norm='batch',  bias=True, stochastic=False, epsilon=0.2, r= reduce_ratios[i], n=H*W, drop_path=dpr[idx], relative_pos=True),
                    FFN(channels[i], channels[i] * 4, act='gelu', drop_path=dpr[idx])
                    )]            
            idx += 1 

    vig_block = Seq(*vig_block)
    

    # == Dummy input == #    
    x = torch.randn(1, channels[0], H, W)  # (B,C,H,W)


    # == Inference == #    
    for l in range(len(vig_block)):
        x = vig_block[l](x)
        print(x.shape)
