""" PatchEmbed (= stem Down Sampling)
    (ref1) https://github.com/sail-sg/poolformer/blob/main/models/poolformer.py  
    (ref2) https://github.com/sail-sg/metaformer/blob/main/metaformer_baselines.py
    (ref3) https://github.com/Sense-X/UniFormer/blob/main/image_classification/models/uniformer.py
    (ref4) https://github.com/sail-sg/metaformer/issues/4
"""
import torch 
import torch.nn as nn 

from timm.models.layers.helpers import to_2tuple

class PatchEmbed(nn.Module):
    """ [following ref1]
    Patch Embedding that is implemented by a layer of conv. 
    Input: tensor in shape [B, C, H, W]
    Output: tensor in shape [B, C, H/stride, W/stride]
    """
    def __init__(self, patch_size=16, stride=16, padding=0, 
                 in_chans=3, embed_dim=768, norm_layer=None):
        super().__init__()
        patch_size = to_2tuple(patch_size)
        stride = to_2tuple(stride)
        padding = to_2tuple(padding)
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, 
                              stride=stride, padding=padding)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        x = self.proj(x)
        x = self.norm(x)
        return x


class Downsampling(nn.Module):
    """ [following ref2]
    Downsampling implemented by a layer of convolution.
    """
    def __init__(self, in_channels, out_channels, 
        kernel_size, stride=1, padding=0, 
        pre_norm=None, post_norm=None, pre_permute=False):
        super().__init__()
        self.pre_norm = pre_norm(in_channels) if pre_norm else nn.Identity()
        self.pre_permute = pre_permute
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, 
                              stride=stride, padding=padding)
        self.post_norm = post_norm(out_channels) if post_norm else nn.Identity()

    def forward(self, x):
        x = self.pre_norm(x)
        if self.pre_permute:
            # if take [B, H, W, C] as input, permute it to [B, C, H, W]
            x = x.permute(0, 3, 1, 2)
        x = self.conv(x)
        x = x.permute(0, 2, 3, 1) # [B, C, H, W] -> [B, H, W, C]
        x = self.post_norm(x)
        return x


class PatchEmbed_v2(nn.Module):
    """ [following ref3]
        Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.norm = nn.LayerNorm(embed_dim)
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x)
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2) # [B,C,H,W] -> [B,H*W,C]
        x = self.norm(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous() # [B,H*W,C] -> [B,C,H,W]
        return x


class MyDownsampling(nn.Module):
    """ [following ref2]; https://github.com/sail-sg/metaformer/issues/4
    Downsampling implemented by a layer of convolution.
    """
    def __init__(self, in_channels, out_channels, 
        kernel_size, stride=1, padding=0, separable=False,
        pre_norm=None, post_norm=None, pre_permute=False):
        super().__init__()
        self.pre_norm = pre_norm(in_channels) if pre_norm else nn.Identity()
        self.pre_permute = pre_permute

        if not separable:
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, 
                                  stride=stride, padding=padding)   
        else: 
            self.conv = nn.Sequential(
                nn.Conv2d(in_channels, in_channels, (1,kernel_size), stride=(1, stride), padding=(0, padding), groups=in_channels),
                nn.Conv2d(in_channels, in_channels, (kernel_size,1), stride=(stride, 1), padding=(padding, 0), groups=in_channels),
                nn.Conv2d(in_channels, out_channels, 1),
            )

        self.post_norm = post_norm(out_channels) if post_norm else nn.Identity()

    def forward(self, x):
        x = self.pre_norm(x)
        if self.pre_permute:
            # if take [B, H, W, C] as input, permute it to [B, C, H, W]
            x = x.permute(0, 3, 1, 2)
        x = self.conv(x)
        x = x.permute(0, 2, 3, 1) # [B, C, H, W] -> [B, H, W, C]
        x = self.post_norm(x)
        return x



if __name__ == "__main__": 
    dim = 3
    x = torch.randn(1, 3, 224, 224) 

    # ---------- #
    # == Ref1 == #
    # ---------- # 
    print(f"--- ref1 ---")
    patch_embed = [ PatchEmbed(patch_size=7, stride=4, padding=2, in_chans=3, embed_dim=64), # in_pathch 
                    PatchEmbed(patch_size=3, stride=2, padding=1, in_chans=64, embed_dim=64*2), # down_patch 
                ]
    stem1 = x.clone()
    for i in range(len(patch_embed)):
        stem1 = patch_embed[i](stem1)
        print(stem1.shape)

    # ---------- #
    # == Ref2 == #
    # ---------- # 
    print(f"--- ref2 ---")
    downsample = [  Downsampling(in_channels=3, out_channels=64, kernel_size=7, stride=4, padding=2, pre_norm=None, post_norm=None), # in_patch 
                    Downsampling(in_channels=64, out_channels=64*2, kernel_size=3, stride=2, padding=1, pre_norm=None, post_norm=None, pre_permute=True), # down_patch
                ]
    stem2 = x.clone() 
    for i in range(len(downsample)):
        stem2 = downsample[i](stem2)
        print(stem2.shape)

    # ---------- #
    # == Ref3 == #
    # ---------- #
    print(f"--- ref3 ---")
    patch_embed = [ PatchEmbed_v2(img_size=224, patch_size=4, in_chans=3, embed_dim=64), # in_patch
                    PatchEmbed_v2(img_size=224//2**2, patch_size=2, in_chans=64, embed_dim=64*2), # in_patch
    ]
    stem3 = x.clone() 
    for i in range(len(patch_embed)):
        stem3 = patch_embed[i](stem3)
        print(stem3.shape)


    # -------------------- #
    # == MyDownsampling == #
    # -------------------- #
    print(f"--- MyDownsampling ---")
    downsample = [  MyDownsampling(in_channels=3, out_channels=64, kernel_size=7, stride=4, padding=2, pre_norm=None, post_norm=None), # in_patch 
                    MyDownsampling(in_channels=64, out_channels=64*2, kernel_size=3, stride=2, padding=1, separable=True, pre_norm=None, post_norm=None, pre_permute=True), # down_patch
                ]
    stem2 = x.clone() 
    for i in range(len(downsample)):
        stem2 = downsample[i](stem2)
        print(stem2.shape)
