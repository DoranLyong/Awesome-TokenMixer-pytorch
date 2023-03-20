# (ref) https://github.com/Atten4Vis/DemystifyLocalViT/blob/HEAD/models/dwnet.py
import torch 
import torch.nn as nn 
import torch.nn.functional as F




class DynamicDWConv(nn.Module):
    def __init__(self, dim, kernel_size, bias=True, stride=1, padding=1, groups=1, reduction=4):
        super().__init__()
        self.dim = dim
        self.kernel_size = kernel_size
        self.stride = stride 
        self.padding = padding 
        self.groups = groups 

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.conv1 = nn.Conv2d(dim, dim // reduction, 1, bias=False) # 1x1 conv; channel reduction
        self.bn = nn.BatchNorm2d(dim // reduction)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(dim // reduction, dim * kernel_size * kernel_size, 1) # 1x1 convl ; channel expansion 

        if bias:
            self.bias = nn.Parameter(torch.zeros(dim))
        else:
            self.bias = None

    def forward(self, x):
        b, c, h, w = x.shape
        weight = self.conv2(self.relu(self.bn(self.conv1(self.pool(x))))) # (B, C*K*K, 1, 1)
        weight = weight.view(b * self.dim, 1, self.kernel_size, self.kernel_size) # (B * C, 1, K, K)

        x = F.conv2d(x.reshape(1, -1, h, w), weight, self.bias.repeat(b), stride=self.stride, padding=self.padding, groups=b * self.groups)
        x = x.view(b, c, x.shape[-2], x.shape[-1])
        return x










if __name__ == "__main__": 
    dim = 96
    window_size = 7
    x = torch.randn(128, dim, 56, 56) # input 

    DWBlock = DynamicDWConv(dim, kernel_size=window_size, stride=1, padding=window_size // 2, groups=dim)

    output = DWBlock(x)
    print(output.shape)