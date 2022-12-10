import torch 
import torch.nn as nn 


class ECA(nn.Module):
    """ Efficient Channel Attention (ECA); https://arxiv.org/abs/1910.03151 (CVPR2019)
        code implementation from :
        https://github.com/BangguWu/ECANet/blob/master/models/eca_module.py
    """
    def __init__(self, channel, k_size=3):
        super(ECA, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1) # (B,C,H,W) -> (B,C,1,1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x)

        # Two different branches of ECA module
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1) # (B,C,1,1) -> (B,1,C) -> (B,C,1,1)

        # Multi-scale information fusion
        y = self.sigmoid(y)
        return x * y.expand_as(x) # element-wise multiplication (residual connection)


if __name__ == "__main__": 
    image_size = [52, 52]
    input = torch.rand(1, 64, *image_size)

    eca = ECA(channel=64)
    output = eca(input)
    print(output.shape)