import math
import time
import torch 
import torch.nn as nn
from torch.nn import functional as F
from dataclasses import dataclass
from tokenizer import Tokenizer


class DataLoaderLite:
    def __init__(self, input, batch_size, sequence_length, tokenizer):
        self.B = batch_size
        self.T = sequence_length
        with open(input, 'r', encoding='utf-8') as f:
            text = f.read()
        self.tokens = torch.tensor(tokenizer.encode(text, bos=False, eos=False))
        self.current_position = 0

    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_position:self.current_position + B * T + 1]
        X, Y = buf[:-1].view(B, T), buf[1:].view(B, T)
        self.current_position += B * T 
        if self.current_position + B * T + 1 > len(self.tokens):
            self.current_position = 0 
        return X, Y


@dataclass
class PiLlamaConfig:
    n_embd: int = 512
    block_size: int = 128 # todo: replace with 1024
    n_heads: int = 16
    kv_heads: int = 4
    n_layers: int = 16
    vocab_size: int = 16384
    flash_attention: bool = True
    rope_base: int = 10000
    max_lr: int = 6e-4


class RMSNorm(nn.Module):
    def __init__(self, n_embd, eps=1e-6):
        super().__init__()
        self.n_embd = n_embd
        self.gain = nn.Parameter(torch.ones((n_embd)))
        self.eps = eps

    def forward(self, x):
        inv_norm = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * inv_norm * self.gain 
        

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

    def _rotate_half(self, x):
        return torch.stack((-x[..., 1::2], x[..., ::2]), dim=-1).view(x.shape)
    
    def forward(self, x):
        B, T, C = x.shape
        q = self.query(x).view(B, T, self.config.n_heads, self.head_size).transpose(1, 2) # (B, nh, T, head_size)
        k = self.key(x).view(B, T, self.config.kv_heads, self.head_size).transpose(1, 2) # (B, nkvh, T, head_size)
        v = self.value(x).view(B, T, self.config.kv_heads, self.head_size).transpose(1, 2) # (B, nkvh, T, head_size)

        k = k.repeat_interleave(self.config.n_heads // self.config.kv_heads, dim=1) # (B, nh, T, head_size)
        v = v.repeat_interleave(self.config.n_heads // self.config.kv_heads, dim=1) # (B, nh, T, head_size)

        # Rotary positional embedding
        inv_freqs = 1.0 / self.config.rope_base ** (torch.arange(0, self.head_size, 2, device=x.device).float() / self.head_size)
        theta = torch.outer(torch.arange(1, T + 1, device=x.device), inv_freqs)
        pos = torch.repeat_interleave(theta, 2, dim=-1) 

        rotated_q = self._rotate_half(q)
        rotated_k = self._rotate_half(k)
        q = q * torch.cos(pos) + rotated_q * torch.sin(pos)
        k = k * torch.cos(pos) + rotated_k * torch.sin(pos)

        if self.config.flash_attention:
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        else:
            attn = (q @ k.transpose(-1, -2)) * (1.0 / math.sqrt(self.head_size))
            attn = attn.masked_fill(self.mask[:,:,:T,:T]==0, float('-inf'))
            attn = F.softmax(attn, dim=-1)
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
    def __init__(self, config: PiLlamaConfig):
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
    def __init__(self, config: PiLlamaConfig):
        super().__init__()
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layers)]),
            rmsn = RMSNorm(config.n_embd)
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size)
        # Weights sharing scheme
        self.transformer.wte.weight = self.lm_head.weight

    def forward(self, idx, targets=None):
        B, T = idx.size()
        x = self.transformer.wte(idx)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.rmsn(x)
        logits = self.lm_head(x) # (B, T, n_embd)
        loss = None
        if targets is not None: 
            loss = F.cross_entropy(logits.view(B * T, -1), targets.view(-1))
        return logits, loss
    
    def configure_optimizer(self, lr=3e-4, betas=(0.9, 0.95), weight_decay=0.1, fused=True, device_type='cuda'):
        params_dict = {np: p for np, p in self.named_parameters()}
        params_dict = {np: p for np, p in params_dict.items() if p.requires_grad}
        decay_params = [p for np, p in params_dict.items() if p.dim() >= 2]
        nodecay_params = [p for np, p in params_dict.items() if p.dim() < 2]
        param_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        print("Number of weights with decay:", sum(p.numel() for p in decay_params))
        print("Number of weights without decay:", sum(p.numel() for p in nodecay_params))
        use_fused = fused and device_type == "cuda"
        optimizer = torch.optim.AdamW(params=param_groups, lr=lr, betas=betas, fused=use_fused)
        return optimizer
    
class CosineLRScheduler:
    def __init__(self, max_lr, min_lr, warmup_step, max_step):
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.warmup_step = warmup_step
        self.max_step = max_step

    def get_lr(self, it):
        if it < self.warmup_step:
            return self.max_lr * (it + 1) / self.warmup_step
        elif it > self.max_step:
            return self.min_lr
        lr_decay_ratio = (it - self.warmup_step) / (self.max_step - self.warmup_step)
        return self.min_lr + 0.5 * (self.max_lr - self.min_lr) * (1 + math.cos(math.pi * lr_decay_ratio))

device = "cpu" 
if torch.cuda.is_available():
    device = "cuda"
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    device = "mps"
print("Device:", device)

model_config = PiLlamaConfig()
model = PiLlama(model_config)
model = model.to(device)
num_params = sum(p.numel() for p in model.parameters())
print(f"Number of parameters: {num_params}")


max_step = 50
optimizer = model.configure_optimizer(device_type=device)
scheduler = CosineLRScheduler(
    max_lr = model_config.max_lr, 
    min_lr = 0.1 * model_config.max_lr, 
    warmup_step = 10, 
    max_step = max_step
)

# # tiny Shakespeare dataset
input = "D:\\Code\\Python\\input.txt"

tokenizer = Tokenizer('pi_tokenizer.model')
train_loader = DataLoaderLite(input, batch_size=4, sequence_length=model_config.block_size, tokenizer=tokenizer)

# Using TF32 (if available) for faster matrix multiplication 
torch.set_float32_matmul_precision('high')

device_type = 'cuda' if device.startswith('cuda') else 'cpu'
# Mixed precision training with cuda
use_amp = device_type == 'cuda'
if use_amp:
    type = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
else:
    type = torch.float32
scaler = torch.amp.GradScaler('cuda', enabled=(type == torch.float16))
    
model.train()
for step in range(max_step):
    t0 = time.time()
    Xb, Yb = train_loader.next_batch()
    Xb, Yb = Xb.to(device), Yb.to(device)
    with torch.autocast(device_type=device_type, dtype=type, enabled=use_amp):
        logits, loss = model(Xb, Yb)
    lr = scheduler.get_lr(step) 
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    optimizer.zero_grad()
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    t1 = time.time()
    dt = t1 - t0
    tok_per_sec = train_loader.B * train_loader.T / dt
    print(f'Step: {step} | Learning rate: {lr:.4e} | Loss: {loss.item():.6f} | Time: {dt:.2f}ms | Tok/sec: {tok_per_sec:.2f}')