import torch
import numpy as np
from einops import rearrange, repeat
from einops.layers.torch import Rearrange

class PreNorm(torch.nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = torch.nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(torch.nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(dim, hidden_dim),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_dim, dim),
            torch.nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

# Define multi-head-self-attention
class Attention(torch.nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = torch.nn.Softmax(dim = -1)
        self.dropout = torch.nn.Dropout(dropout)

        self.to_qkv = torch.nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = torch.nn.Sequential(
            torch.nn.Linear(inner_dim, dim),
            torch.nn.Dropout(dropout)
        ) if project_out else torch.nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(torch.nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = torch.nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(torch.nn.ModuleList([
                PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x

# Define model (which extends the NN module)
class custom(torch.nn.Module):
    def __init__(self, num_blocks=20, num_heads=16, hidden_dimension=768):
        super(custom, self).__init__()

        # Attributes
        self.num_patches = 14*14 # 224x224 image is (14 x 14 pacthes of dimension 16x16 each)
        self.patch_size = 16
        self.input_dimension = self.patch_size * self.patch_size * 3
        self.num_blocks = num_blocks
        self.num_heads = num_heads
        self.hidden_dimension = hidden_dimension

        # Patch embedding
        self.patch_embedding = torch.nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = self.patch_size, p2 = self.patch_size),
            torch.nn.LayerNorm(self.input_dimension),
            torch.nn.Linear(self.input_dimension, self.hidden_dimension),
            torch.nn.LayerNorm(self.hidden_dimension),
        )
        
        # Positional embedding
        self.pos_embedding = torch.nn.Parameter(torch.randn(1, self.num_patches, self.hidden_dimension))

        # Transformer encoder blocks
        self.transformer = Transformer(self.hidden_dimension, self.num_blocks, self.num_heads, dim_head=64, mlp_dim=1024, dropout=0.1)

        # Prediction head
        self.head = torch.nn.Conv2d(in_channels=self.hidden_dimension, out_channels=1, kernel_size=1, stride=1, padding=0)

    # Forward
    def forward(self, x):
        b, c, h, w = x.shape

        x = self.patch_embedding(x)

        x += self.pos_embedding
        
        # Transformer Blocks
        x = self.transformer(x)
        
        # Final
        x = x.reshape(b, -1, 14, 14).contiguous()
        x = self.head(x)
        
        return x

#FIN