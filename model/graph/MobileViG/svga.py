# (ref) https://github.com/SLDGroup/MobileViG

import torch 
import torch.nn as nn 

from timm.models.layers import DropPath


class MRConv4d(nn.Module):
    """
    Max-Relative Graph Convolution (Paper: https://arxiv.org/abs/1904.03751) for dense data type
    
    K is the number of superpatches, therefore hops equals res // K.
    """
    def __init__(self, in_channels, out_channels, K=2):
        super(MRConv4d, self).__init__()
        self.nn = nn.Sequential(
            nn.Conv2d(in_channels * 2, out_channels, 1),
            nn.BatchNorm2d(in_channels * 2),
            nn.GELU()
            )
        self.K = K

    def forward(self, x):
        B, C, H, W = x.shape
            
        x_j = x - x
        for i in range(self.K, H, self.K):
            x_c = x - torch.cat([x[:, :, -i:, :], x[:, :, :-i, :]], dim=2)
            x_j = torch.max(x_j, x_c)
        for i in range(self.K, W, self.K):
            x_r = x - torch.cat([x[:, :, :, -i:], x[:, :, :, :-i]], dim=3)
            x_j = torch.max(x_j, x_r)

        x = torch.cat([x, x_j], dim=1)
        return self.nn(x)
        

class Grapher(nn.Module):
    """
    Grapher module with graph convolution and fc layers
    """
    def __init__(self, in_channels, drop_path=0.0, K=2):
        super(Grapher, self).__init__()
        self.channels = in_channels
        self.K = K

        self.fc1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 1, stride=1, padding=0),
            nn.BatchNorm2d(in_channels),
        )
        self.graph_conv = MRConv4d(in_channels, in_channels * 2, K=self.K)
        self.fc2 = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels, 1, stride=1, padding=0),
            nn.BatchNorm2d(in_channels),
        )  # out_channels back to 1x}
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

       
    def forward(self, x):
        _tmp = x
        x = self.fc1(x)
        x = self.graph_conv(x)
        x = self.fc2(x)
        x = self.drop_path(x) + _tmp

        return x





if __name__ == "__main__":

    H, W = (224//16, 224//16)
    channels = 250

    # == Dummy input == # 
    x = torch.randn(1, channels, H, W)

    # == Build SVGA == # 
    svga_block = Grapher(channels, drop_path=0.0, K=2)

    # == Inference == # 
    output = svga_block(x)
    print(output.shape)