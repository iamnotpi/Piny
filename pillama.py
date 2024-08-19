import torch 
import torch.nn as nn
from torch.nn import functional as F
from dataclasses import dataclass

# tiny Shakespeare dataset
with open("D:\\Code\\Python\\input.txt", 'r', encoding='utf-8') as file:
    text = file.read()

@dataclass
class PiLlamaConfig:
    n_embd: int = 768
    block_size: int = 1024
    n_heads: int = 12
    kv_heads: int = 4
    n_layers: int = 4
    vocab_size: int = 50257
    flash_attention: bool = True

class RMSNorm(nn.Module):
    def __init__(self, n_embd, eps=1e-6):
        super().__init__()
        self.n_embd = n_embd
        self.gain = nn.Parameter(torch.ones((n_embd)))
        self.eps = eps

    def forward(self, x):
        norm = torch.linalg.norm(x, 2, dim=-1, keepdims=True)
        return x / (norm + self.eps) * self.gain * self.n_embd ** 0.5
        

class GroupedQueryAttention(nn.Module):
    def __init__(self, config: PiLlamaConfig):
        super().__init__()
        self.config = config
        assert config.n_heads > config.kv_heads, "Number of heads must be larger than number of kv heads"
        self.head_size = config.n_embd // config.n_heads
        self.query = nn.Linear(config.n_embd, config.n_heads * self.head_size, bias=False) 
        self.key = nn.Linear(config.n_embd, config.kv_heads * self.head_size, bias=False)
        self.value = nn.Linear(config.n_embd, config.kv_heads * self.head_size, bias=False)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.register_buffer("mask", torch.tril(torch.ones((config.block_size, config.block_size))).view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        B, T, C = x.shape
        q = self.query(x).view(B, T, self.config.n_heads, self.head_size).transpose(1, 2) # (B, nh, T, head_size)
        k = self.key(x).view(B, T, self.config.kv_heads, self.head_size).transpose(1, 2) # (B, nkvh, T, head_size)
        v = self.value(x).view(B, T, self.config.kv_heads, self.head_size).transpose(1, 2) # (B, nkvh, T, head_size)

        k = k.repeat_interleave(self.config.n_heads // self.config.kv_heads, dim=1) # (B, nh, T, head_size)
        v = v.repeat_interleave(self.config.n_heads // self.config.kv_heads, dim=1) # (B, nh, T, head_size)

        if self.config.flash_attention:
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        else:
            attn = (q @ k.transpose(-1, -2) * C ** -0.5)
            attn = F.softmax(attn, dim=-1)
            attn = attn.masked_fill(self.mask[:,:,:T,:T]==0, float('-inf'))
            y = attn @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.c_proj(y)


class MLP(nn.Module):
    def __init__(self, config: PiLlamaConfig):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)

    def forward(self, x):
        x = self.c_fc(x) 
        x = self.gelu(x)
        return self.c_proj(x)


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attn = GroupedQueryAttention(config)
        self.rmsn_attn = RMSNorm(config.n_embd)
        self.mlp = MLP(config)
        self.rmsn_mlp = RMSNorm(config.n_embd)
    
    def forward(self, x):
        x = x + self.attn(self.rmsn_attn(x))
        x = x + self.mlp(self.rmsn_mlp(x))
        return x

class PiLlama(nn.Module):
    pass