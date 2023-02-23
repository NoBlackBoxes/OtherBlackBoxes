import torch
import numpy as np
from einops.layers.torch import Rearrange

# Define multi-head-self-attention
class mhsa(torch.nn.Module):
    def __init__(self, d, num_heads=2):
        super(mhsa, self).__init__()
        self.d = d
        self.num_heads = num_heads

        assert d % num_heads == 0, f"Can't divide dimension {d} into {num_heads} heads"

        d_head = int(d / num_heads)
        self.q_mappings = torch.nn.ModuleList([torch.nn.Linear(d_head, d_head) for _ in range(self.num_heads)])
        self.k_mappings = torch.nn.ModuleList([torch.nn.Linear(d_head, d_head) for _ in range(self.num_heads)])
        self.v_mappings = torch.nn.ModuleList([torch.nn.Linear(d_head, d_head) for _ in range(self.num_heads)])
        self.d_head = d_head
        self.softmax =torch.nn.Softmax(dim=-1)

    def forward(self, sequences):
        # Sequences has shape (N, seq_length, token_dim)
        # We go into shape    (N, seq_length, n_heads, token_dim / n_heads)
        # And come back to    (N, seq_length, item_dim)  (through concatenation)
        result = []
        for sequence in sequences:
            seq_result = []
            for head in range(self.num_heads):
                q_mapping = self.q_mappings[head]
                k_mapping = self.k_mappings[head]
                v_mapping = self.v_mappings[head]

                seq = sequence[:, head * self.d_head: (head + 1) * self.d_head]
                q, k, v = q_mapping(seq), k_mapping(seq), v_mapping(seq)

                attention = self.softmax(q @ k.T / (self.d_head ** 0.5))
                seq_result.append(attention @ v)
            result.append(torch.hstack(seq_result))
        return torch.cat([torch.unsqueeze(r, dim=0) for r in result])

# Define transformer block
class block(torch.nn.Module):
    def __init__(self, hidden_dimension, num_heads, mlp_ratio=4):
        super(block, self).__init__()

        # Attributes
        self.hidden_dimension = hidden_dimension
        self.num_heads = num_heads

        self.norm1 = torch.nn.LayerNorm(hidden_dimension)
        self.mhsa = mhsa(hidden_dimension, num_heads)
        self.norm2 = torch.nn.LayerNorm(hidden_dimension)
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(hidden_dimension, mlp_ratio * hidden_dimension),
            torch.nn.GELU(),
            torch.nn.Linear(mlp_ratio * hidden_dimension, hidden_dimension)
        )

    def forward(self, x):
        out = x + self.mhsa(self.norm1(x))
        out = out + self.mlp(self.norm2(out))
        return out

# Define model (which extends the NN module)
class custom(torch.nn.Module):
    def __init__(self, num_blocks=20, num_heads=16, hidden_dimension=256):
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
        self.blocks = torch.nn.ModuleList([block(self.hidden_dimension, self.num_heads) for _ in range(self.num_blocks)])
        
        # Prediction head
        self.head = torch.nn.Conv2d(in_channels=self.hidden_dimension, out_channels=1, kernel_size=1, stride=1, padding=0)

    # Forward
    def forward(self, x):
        b, c, h, w = x.shape

        x = self.patch_embedding(x)

        x += self.pos_embedding
        
        # Transformer Blocks
        for block in self.blocks:
            x = block(x)
        
        # Final
        x = x.reshape(b, -1, 14, 14).contiguous()
        x = self.head(x)
        
        return x

#FIN