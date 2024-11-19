# fmt: off

class Attention(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.NH = num_heads
        self.HD = dim // num_heads
        self.scale = self.HD**-0.5
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        B, N, D = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.NH, self.HD)  # (B, N, 3, NH, HD)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).chunk(3)         # (B, NH, N, HD) x 3
        attn = q @ k.transpose(-2, -1)                        # (B, NH, N, N)
        attn = (attn * self.scale).softmax(dim=-1)            # (B, NH, N, N)
        x = (attn @ v).transpose(1, 2).reshape(B, N, D)       # (B, N, D)
        return self.proj(x)
