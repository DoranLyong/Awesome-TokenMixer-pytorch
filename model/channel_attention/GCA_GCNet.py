import torch 
import torch.nn as nn 

from mmcv.cnn import constant_init, kaiming_init
from timm.models.layers import make_divisible


def last_zero_init(m):
    if isinstance(m, nn.Sequential):
        constant_init(m[-1], val=0)
    else:
        constant_init(m, val=0)


class GCA(nn.Module):
    """ Global Context Attention (GCNet); https://arxiv.org/abs/1904.11492 (ICCV2019)
        code implementation from :
        https://github.com/xvjiarui/GCNet/blob/029db5407dc27147eb1d41f62b09dfed8ec88837/mmdet/ops/gcb/context_block.py#L13

        --channel: input channel
        --ratio: channel ratio of the output channel
        --pooling_type: 'att' or 'avg'
        --fusion_types: 'channel_add' or 'channel_mul'
    """
    def __init__(self, channel, ratio, pooling_type='att', fusion_types='channel_add'):
        super(GCA, self).__init__()

        self.in_channel = channel
        self.ratio = ratio
        self.out_channel = make_divisible(channel * ratio, 2)

        self.pooling_type = pooling_type
        self.fusion_type = fusion_types

        # == Global Context Pooling (Modeling) == # 
        if pooling_type == 'att':
            self.conv_mask = nn.Conv2d(channel, 1, kernel_size=1)
            self.softmax = nn.Softmax(dim=2)
        
        elif pooling_type == 'avg':
            self.avg_pool = nn.AdaptiveAvgPool2d(1)

        # == Transform == # 
        self.transform = nn.Sequential(
                nn.Conv2d(self.in_channel, self.out_channel, kernel_size=1), # 1x1 conv 
                nn.LayerNorm([self.out_channel, 1, 1]),
                nn.GELU(), # nn.ReLU(inplace=True),  # yapf: disable
                nn.Conv2d(self.out_channel, self.in_channel, kernel_size=1) # 1x1 conv 
                )

        # == init. param == # 
        self.reset_parameters()

    def reset_parameters(self):
        if self.pooling_type == 'att':
            kaiming_init(self.conv_mask, mode='fan_in')
            self.conv_mask.inited = True
        if self.transform is not None: 
            last_zero_init(self.transform )


    def spatial_pool(self, x):
        """ Global Context Pooling (Modeling) 

        intput: (B,C,H,W)
        output: (B,C,1,1)
        """
        B, C, H, W  = x.size()

        if self.pooling_type == 'att':
            input_x = x # (B,C,H,W)
            input_x = input_x.view(B, C, H * W) # (B,C,H*W)
            input_x = input_x.unsqueeze(dim=1) # (B,1,C,H*W)

            context_mask = self.conv_mask(x) # (B,1,H,W) ; 오잉? 이게 왜 되지? 
            context_mask = context_mask.view(B, 1, H * W) # (B,1,H*W)
            context_mask = self.softmax(context_mask) # Attention_map; (B,1,H*W)
            context_mask = context_mask.unsqueeze(-1) # (B,1,H*W,1)

            context = torch.matmul(input_x, context_mask) # (B,1,C,1)
            context = context.view(B, C, 1, 1) # (B,C,1,1)

        else: 
            context = self.avg_pool(x) # (B,C,1,1)

        return context

        
    def forward(self, x):
        assert x.ndim == 4  # (B,C,H,W)

        # == Global Context Modeling == #
        context = self.spatial_pool(x) # (B,C,1,1)

        # == Transform & Fusion == #
        if self.fusion_type == 'channel_add':
            channel_add_term = self.transform(context) # (B,C,1,1)
            out = x + channel_add_term # (B,C,H,W)
        
        elif self.fusion_type == 'channel_mul':
            channel_mul_term = torch.sigmoid(self.transform(context)) # (B,C,1,1)
            out = x * channel_mul_term # (B,C,H,W)

        return out


if __name__ == "__main__": 
    image_size = [52, 52]
    input = torch.rand(1, 64, *image_size)

    gca = GCA(channel=64, ratio=1)
    output = gca(input)
    print(output.shape)