import torch 
import torch.nn as nn 


class SE(nn.Module):
    """ Squeeze-and-Excitation Networks (SENet); https://arxiv.org/abs/1709.01507 (CVPR2018)
        code implementation from :
        - https://blog.paperspace.com/channel-attention-squeeze-and-excitation-networks/
        - https://github.com/jonnedtc/Squeeze-Excitation-PyTorch
    """

    def __init__(self, channel, reduction_ratio =16):
        super(SE, self).__init__()
        # == Global Average Pooling == #
        self.avg_pool = nn.AdaptiveAvgPool2d(1) # (B,C,H,W) -> (B,C,1,1)
        
        # == Fully Connected Multi-Layer Perceptron (FC-MLP) == # 
        self.mlp = nn.Sequential(
            nn.Linear(channel, channel // reduction_ratio, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction_ratio, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.mlp(y).view(b, c, 1, 1)
        return x * y.expand_as(x) # element-wise multiplication (residual connection)


if __name__ == "__main__": 
    image_size = [52, 52]
    input = torch.rand(1, 64, *image_size)

    se = SE(channel=64)
    output = se(input)
    print(output.shape)