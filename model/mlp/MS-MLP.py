# (ref) https://github.com/JegZheng/MS-MLP/blob/main/models/ms_mlp.py
import torch 
import torch.nn as nn
import torch.nn.functional as F 

from timm.models.layers import DropPath, to_2tuple, trunc_normal_

class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first. 
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with 
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs 
    with shape (batch_size, channels, height, width).
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError 
        self.normalized_shape = (normalized_shape, )
    
    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x



class MixShiftBlock(nn.Module):
    r""" Mix-Shifting Block.
    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resulotion;(H,W)
        shift_size (int): Shift size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        drop (float, optional): Dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, input_resolution, shift_size, shift_dist, mix_size, layer_scale_init_value=1e-6,
                mlp_ratio=4, drop=0., drop_path=0.,
                act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution 
        self.mlp_ratio = mlp_ratio

        self.shift_size = shift_size
        self.shift_dist = shift_dist
        self.chunk_size = [i.shape[0] for i in torch.chunk(torch.zeros(dim), self.shift_size)] # dim을 shift_size로 등분한 각각의 크기
        
        self.kernel_size = [(ms, ms//2) for ms in mix_size]
        self.dwconv_lr = nn.ModuleList([nn.Conv2d(chunk_dim, chunk_dim, kernel_size = kernel_size[0], padding = kernel_size[1], groups=chunk_dim) for chunk_dim, kernel_size in zip(self.chunk_size, self.kernel_size)])
        self.dwconv_td = nn.ModuleList([nn.Conv2d(chunk_dim, chunk_dim, kernel_size = kernel_size[0], padding = kernel_size[1], groups=chunk_dim) for chunk_dim, kernel_size in zip(self.chunk_size, self.kernel_size)])
        
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, int(mlp_ratio * dim)) # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(int(mlp_ratio * dim), dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)), 
                                    requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        B_, C, H, W = x.shape

        # ==split groups== # 
        xs = torch.chunk(x, self.shift_size, 1) # 채널 방향으로 shift_size만큼 쪼갬 

        # ==shift with pre-defined relative distance== # 
        x_shift_lr = [ torch.roll(x_c, shift, 3) for x_c, shift in zip(xs, self.shift_dist)] # shift 연산 
        x_shift_td = [ torch.roll(x_c, shift, 2) for x_c, shift in zip(xs, self.shift_dist)]

        # == regional mixing == #
        for i in range(self.shift_size):
            x_shift_lr[i] = self.dwconv_lr[i](x_shift_lr[i]) # region mixing 
            x_shift_td[i] = self.dwconv_td[i](x_shift_td[i]) # relative distance 


        # == Aggregation == # 
        x_lr = torch.cat(x_shift_lr, 1) # (B,dim,H,W)
        x_td = torch.cat(x_shift_td, 1) # (B,dim,H,W)
        
        x = x_lr + x_td 

        # == Projection == # 
        x = x.permute(0, 2, 3, 1) # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x  # scale 
        x = x.permute(0, 3, 1, 2) # (N, H, W, C) -> (N, C, H, W)

        x = input + self.drop_path(x)
        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, " \
               f"shift_size={self.shift_size}, mlp_ratio={self.mlp_ratio}"

    def flops(self):
        flops = 0
        H, W = self.input_resolution
        N = H * W
        # dwconv_1 dwconv_2
        for i in range(self.shift_size):
            flops += 2 * (N * self.chunk_size[i] * self.kernel_size[i][0])
        # x_lr + x_td
        flops += N * self.dim
        # norm
        flops += self.dim * H * W
        # pwconv
        flops += 2 * H * W * self.dim * self.dim * self.mlp_ratio
        return flops



if __name__ == "__main__": 
    dim = 64
    x = torch.randn(1, dim,7 ,7) # input 

    ms_block = MixShiftBlock(dim=dim, 
                        input_resolution=(7,7),
                        shift_size=5,
                        shift_dist=[-2,-1,0,1,2], 
                        mix_size=[1,1,3,5,7] 
                        )
                    
    output_ms = ms_block(x)
    print(output_ms.shape)