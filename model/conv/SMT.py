# (ref) https://github.com/AFeng-x/SMT/blob/main/models/smt.py
import torch 
import torch.nn as nn 


class SMT(nn.Module):
    def __init__(self, dim, num_heads=4, expand_ratio=2, act_layer=nn.GELU,
                 proj_drop=0., proj_bias=False, **kwargs):
        super().__init__()

        self.dim = dim
        self.num_heads = num_heads

        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.v = nn.Linear(dim, dim, bias=True)
        self.s = nn.Linear(dim, dim, bias=True)

        self.split_groups=self.dim// num_heads

        for i in range(self.num_heads):
            local_conv = nn.Conv2d(dim//self.num_heads, dim//self.num_heads, kernel_size=(3+i*2), padding=(1+i), stride=1, groups=dim//self.num_heads)
            setattr(self, f"local_conv_{i + 1}", local_conv)

        self.proj = nn.Sequential(
            nn.Conv2d(dim, dim*expand_ratio, kernel_size=1, padding=0, stride=1, groups=self.split_groups, bias=True),
            nn.BatchNorm2d(dim*expand_ratio),
            act_layer(),
            nn.Conv2d(dim*expand_ratio, dim, kernel_size=1, padding=0, stride=1, bias=True))

        # -- Update -- #
        self.proj_out = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, H, W, C = x.shape

        v = self.v(x) # (B, H, W, C)
        s = self.s(x).reshape(B, H, W, self.num_heads, C//self.num_heads).permute(3, 0, 4, 1, 2) # (num_heads, B,  head_size, H, W)

        for i in range(self.num_heads):
            local_conv = getattr(self, f"local_conv_{i + 1}")
            s_i = local_conv(s[i]).reshape(B, self.split_groups, -1, H, W)

            if i == 0:
                s_out = s_i
            else:
                s_out = torch.cat((s_out, s_i), dim=2)

        s_out = s_out.contiguous().view(B, C, H, W)
        s_out = self.proj(s_out).permute(0, 2, 3, 1) # (B, H, W, C)

        x = v * s_out # modulation

        # == Update == #
        x = self.proj_out(x)
        x = self.proj_drop(x)
        return x









if __name__ == "__main__": 
    dim = 64 
    x = torch.randn(1, 7, 7, dim) # (B, H, W, C)

    token_mixer = SMT(dim=dim)
    out = token_mixer(x)
    print(out.shape)  