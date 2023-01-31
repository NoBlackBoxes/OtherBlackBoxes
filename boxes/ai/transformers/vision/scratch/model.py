import torch
import numpy as np

# Patchify input batch
def patchify(input_batch, num_patches):
    n, c, h, w = input_batch.shape

    patches = torch.zeros(n, num_patches ** 2, h * w * c // num_patches ** 2)
    patch_size = h // num_patches

    for idx, image in enumerate(input_batch):
        for i in range(num_patches):
            for j in range(num_patches):
                patch = image[:, i * patch_size: (i + 1) * patch_size, j * patch_size: (j + 1) * patch_size]
                patches[idx, i * num_patches + j] = patch.flatten()
    return patches

# Positional embeddings
def get_positional_embeddings(sequence_length, d):
    result = torch.ones(sequence_length, d)
    for i in range(sequence_length):
        for j in range(d):
            result[i][j] = np.sin(i / (10000 ** (j / d))) if j % 2 == 0 else np.cos(i / (10000 ** ((j - 1) / d)))
    return result

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
    def __init__(self, num_blocks=20, num_heads=8, hidden_dimension=64, output_dimension=2):
        super(custom, self).__init__()

        # Attributes
        self.num_patches = 14 # 224x224 image is (14 x 14 pacthes of dimensin 16x16 each)
        self.num_blocks = num_blocks
        self.num_heads = num_heads
        self.hidden_dimension = hidden_dimension

        # 1) Linear embedding
        self.input_dimension = 3 * 16 * 16
        self.linear_embedding = torch.nn.Linear(self.input_dimension, self.hidden_dimension)
        
        # 2) Learnable classification token
        self.class_token = torch.nn.Parameter(torch.rand(1, self.hidden_dimension))
        
        # 3) Positional embedding
        self.register_buffer('positional_embeddings', get_positional_embeddings(self.num_patches ** 2 + 1, self.hidden_dimension), persistent=False)
        
        # 4) Transformer encoder blocks
        self.blocks = torch.nn.ModuleList([block(self.hidden_dimension, self.num_heads) for _ in range(self.num_blocks)])
        
        # 5) Classification MLPk
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(self.hidden_dimension, output_dimension),
            torch.nn.Softmax(dim=-1)
        )

    # Forward
    def forward(self, x):
        n, c, h, w = x.shape
        patches = patchify(x, self.num_patches).to(self.positional_embeddings.device)
        
        tokens = self.linear_embedding(patches)

        tokens = torch.cat((self.class_token.expand(n, 1, -1), tokens), dim=1)

        x = tokens + self.positional_embeddings.repeat(n, 1, 1)

        # Transformer Blocks
        for block in self.blocks:
            x = block(x)
        
        # Getting the classification token only
        x = x[:, 0]

        # Classify
        x = self.mlp(x)
        
        return x

#FIN