""" (ref) https://github.com/sail-sg/iFormer
"""
import torch 
import torch.nn as nn 


class HighMixer(nn.Module):
    """ Mixer for high-frequency features
        via 2D convolution & max-pooling
    """
    def __init__(self, dim, kernel_size=3, stride=1, padding=1, **kwargs):
        super().__init__()

        self.cnn_in = cnn_in = dim // 2
        pool_in = dim // 2

        cnn_dim = cnn_in * 2 # expand dim for output 
        pool_dim = pool_in * 2 # expand dim for output 

        # == Branch 1 == # 
        self.conv1 = nn.Conv2d(cnn_in, cnn_dim, kernel_size=1, stride=1, padding=0, bias=False)
        self.proj1 = nn.Conv2d(cnn_dim, cnn_dim, kernel_size=kernel_size, stride=stride, padding=padding, bias=False, groups=cnn_dim)
        self.act1 = nn.GELU()

        # == Branch 2 == #
        self.Maxpool = nn.MaxPool2d(kernel_size, stride=stride, padding=padding)
        self.proj2 = nn.Conv2d(pool_in, pool_dim, kernel_size=1, stride=1, padding=0)
        self.act2 = nn.GELU()

    def forward(self, x): 
        """ x: (B,C,H,W)
        """ 
        # -- Branch 1 
        cx = x[:, :self.cnn_in, :, :].contiguous() # split dim 
        cx = self.conv1(cx)
        cx = self.proj1(cx)
        cx = self.act1(cx)

        # -- Branch 2 
        px = x[:, self.cnn_in:, :, :].contiguous()
        px = self.Maxpool(px)
        px = self.proj2(px)
        px = self.act2(px)

        # -- Assemble 
        hx = torch.cat((cx, px), dim=1)
        return hx


class LowMixer(nn.Module): 
    """ Mixer for low-frequency features 
        via multi-head self-attention (MHSA)
    """
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., pool_size=2, **kwargs):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.dim = dim

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)

        self.pool = nn.AvgPool2d(pool_size, stride=pool_size, padding=0, count_include_pad=False) if pool_size > 1 else nn.Identity()
        self.uppool = nn.Upsample(scale_factor=pool_size) if pool_size > 1 else nn.Identity()

    def att_fun(self, q, k, v, B, N, C):
        """ MHSA 
        """
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(2, 3).reshape(B, C, N)
        return x
    
    def forward(self, x):
        """ x: (B,C,H,W)
        """
        B, _, _, _ = x.shape
        xa = self.pool(x)
        xa = xa.permute(0, 2, 3, 1).view(B, -1, self.dim)  # (B,C,H,W) -> (B, N, C)
        B, N, C = xa.shape

        qkv = self.qkv(xa).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)   # make torchscript happy (cannot use tensor as tuple)
        xa = self.att_fun(q, k, v, B, N, C)
        xa = xa.view(B, C, int(N**0.5), int(N**0.5))#.permute(0, 3, 1, 2)

        xa = self.uppool(xa)
        return xa


class iMixer(nn.Module):
    """ Inception Mixer with HighMixer & LowMixer
    """
    def __init__(self, dim, num_heads=8, qkv_bias=False, 
                        attn_drop=0., proj_drop=0., attention_head=1, pool_size=2, **kwargs):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim = dim // num_heads

        self.low_dim = low_dim = attention_head * head_dim
        self.high_dim = high_dim = dim - low_dim

        self.high_mixer = HighMixer(high_dim)
        self.low_mixer = LowMixer(low_dim, num_heads=attention_head, qkv_bias=qkv_bias, attn_drop=attn_drop, pool_size=pool_size)

        self.conv_fuse = nn.Conv2d(low_dim+high_dim*2, low_dim+high_dim*2, kernel_size=3, stride=1, padding=1, bias=False, groups=low_dim+high_dim*2)
        self.proj = nn.Conv2d(low_dim+high_dim*2, dim, kernel_size=1, stride=1, padding=0)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, H, W, C = x.shape
        x = x.permute(0, 3, 1, 2) # (B,H,W,C) -> (B,C,H,W)

        # -- branch for high-frequency features
        hx = x[:,:self.high_dim,:,:].contiguous()
        hx = self.high_mixer(hx)

        # -- branch for low-frequency features
        lx = x[:,self.high_dim:,:,:].contiguous()
        lx = self.low_mixer(lx)

        x = torch.cat((hx, lx), dim=1)
        x = x + self.conv_fuse(x)
        x = self.proj(x)
        x = self.proj_drop(x)
        x = x.permute(0, 2, 3, 1).contiguous() # (B,C,H,W) -> (B,H,W,C)
        return x



if __name__ == "__main__":
    embed_dims = [96, 192, 320, 384]
    num_heads = [3, 6, 10, 12] # 96//3 , 192//6, 320//10, 384//12
    attention_heads = [1]*3 + [3]*3 + [7] * 4 + [9] * 5 + [11] * 3
    i = 3

    x = torch.randn(1, 7, 7, embed_dims[i]) # input 

    attn = iMixer(dim=embed_dims[i], num_heads=num_heads[i],  qkv_bias=True, 
                    attn_drop=0., attention_head=attention_heads[3+3+4+5+1], pool_size=1)
    output = attn(x)
    print(output.shape)