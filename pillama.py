import math
import os
import time

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from dataclasses import dataclass
from torch.distributed import destroy_process_group, init_process_group
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel as DDP

from tokenizer import Tokenizer


def load_tokens(filename):
    np_tokens = np.load(filename).astype(np.int32)
    tokens = torch.tensor(np_tokens, dtype=torch.long)
    return tokens


class DataLoaderLite:
    def __init__(self, batch_size, sequence_length, process_rank, num_processes, split):
        assert split in ['train', 'validation'], "Split must be either 'train' or 'validation'"
        self.B = batch_size
        self.T = sequence_length
        self.process_rank = process_rank
        self.num_processes = num_processes
        data_dir = 'TinyStories'
        self.shards = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if split in f]
        self.num_shards = len(self.shards)
        self.advance()

    def advance(self, reset=False):
        # Advance to the next shard
        self.current_shard = (self.current_shard + 1) % self.num_shards if not reset else 0
        self.tokens = load_tokens(self.shards[self.current_shard])
        self.current_position = self.process_rank * self.B * self.T

    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_position:self.current_position + B * T + 1]
        X, Y = buf[:-1].view(B, T), buf[1:].view(B, T)
        self.current_position += B * T * self.num_processes
        if self.current_position + B * T * self.process_rank + 1 > len(self.tokens):
            self.advance() 
        return X, Y


@dataclass
class PiLlamaConfig:
    n_embd: int = 768
    ffn_dim: int = int(8/3 * n_embd)
    block_size: int = 1024
    n_heads: int = 16
    kv_heads: int = 4
    n_layers: int = 4
    vocab_size: int = 16384
    flash_attention: bool = True
    rope_base: int = 10000
    max_lr: int = 6e-4
    max_batch_size: int = 65536

B, T = 4, 1024

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
        self.w1 = nn.Linear(config.n_embd, config.ffn_dim, bias=False)
        self.w2 = nn.Linear(config.n_embd, config.ffn_dim, bias=False)
        self.w3 = nn.Linear(config.ffn_dim, config.n_embd, bias=False)
        self.silu = nn.SiLU()

    def forward(self, x):
        return self.w3(self.silu(self.w2(x)) * self.w1(x))


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
    def __init__(self, config: PiLlamaConfig, tokenizer: Tokenizer):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layers)]),
            rmsn = RMSNorm(config.n_embd)
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # Weights sharing scheme
        self.transformer.wte.weight = self.lm_head.weight
        self.tokenizer = tokenizer

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
        return logits, loss if loss is not None else logits
    
    def configure_optimizer(self, lr=3e-4, betas=(0.9, 0.95), weight_decay=0.1, fused=True, device_type='cuda'):
        params_dict = {np: p for np, p in self.named_parameters()}
        params_dict = {np: p for np, p in params_dict.items() if p.requires_grad}
        decay_params = [p for np, p in params_dict.items() if p.dim() >= 2]
        nodecay_params = [p for np, p in params_dict.items() if p.dim() < 2]
        param_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        if master_process:
            print("Number of weights with decay:", sum(p.numel() for p in decay_params))
            print("Number of weights without decay:", sum(p.numel() for p in nodecay_params))
        use_fused = fused and device_type == "cuda"
        optimizer = torch.optim.AdamW(params=param_groups, lr=lr, betas=betas, fused=use_fused)
        return optimizer
    
    @torch.no_grad()
    def generate(self, tokens, max_gen_len, top_k, temperature, seed):
        # tokens: A list of token ids
        self.eval()
        output = []
        for ids in tokens: 
            x = torch.tensor(ids, dtype=torch.long).unsqueeze(0)
            while x.size(1) < max_gen_len or x[:, -1] != self.tokenizer.eos_id:
                with torch.no_grad():
                    logits, _ = self(x)
                    logits = logits[:, -1, :]
                    if temperature > 0:
                        logits /= temperature
                    probs = F.softmax(logits, dim=-1)
                    top_k_probs, top_k_indices = torch.topk(probs, top_k, dim=-1)
                    torch.manual_seed(seed)
                    next_token = torch.multinomial(top_k_probs, num_samples=1)
                    xcol = torch.gather(top_k_indices, -1, next_token) # Similar to indexing top_k_indices with next_token
                    x = torch.cat((x, xcol), dim=1)
            output.append(x.tolist())
        return output
    
    def text_generation(self, texts, max_gen_len, top_k, temperature, seed):
        tokens = [self.tokenizer.encode(text, bos=True, eos=False) for text in texts]
        generated = self.generate(tokens, max_gen_len, top_k, temperature, seed)
        decoded = []
        for ids in generated:
            decoded.append(self.tokenizer.decode(ids))
        return decoded
    
    @staticmethod
    def load_checkpoint(checkpoint_path, tokenizer_path='pi_tokenizer.model', device='cpu'):
        assert os.path.isfile(checkpoint_path), f"Checkpoint file not found: {checkpoint_path}"
        checkpoint = torch.load(checkpoint_path, map_location=device)
        tokenizer = Tokenizer(tokenizer_path)
        model = PiLlama(PiLlamaConfig(), tokenizer)
        model.load_state_dict(checkpoint['model'])
        return model
        

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

# https://pytorch.org/docs/stable/elastic/run.html
# torchrun --standalone --nproc_per_node=2 pillama.py
ddp = int(os.environ.get('RANK', -1)) != -1 # Is this a ddp run?
if ddp: 
    assert torch.cuda.is_available(), "CUDA required!"
    init_process_group('nccl') # Backend to use (nccl for distributed GPUs training)
    ddp_rank = int(os.environ['RANK']) # Rank of the current processor (on all nodes) (1 node <-> 1 device)
    ddp_local_rank = int(os.environ['LOCAL_RANK']) # Rank of the current processor on the current node 
    ddp_world_size = int(os.environ['WORLD_SIZE']) # Number of GPUs
    device = f'cuda:{ddp_local_rank}' # 1 machine -> local_rank == rank
    torch.cuda.set_device(device)
    master_process = (ddp_rank == 0)
else:
    ddp_rank = 0
    ddp_local_rank = 0
    ddp_world_size = 1
    master_process = True
    device = "cpu" 
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = "mps"
    print("Device:", device)

device_type = 'cuda' if device.startswith('cuda') else 'cpu'

# Using TF32 (if available) for faster matrix multiplication 
torch.set_float32_matmul_precision('high')

# Mixed precision training with cuda
use_amp = device_type == 'cuda'
if use_amp:
    type = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
else:
    type = torch.float32
scaler = torch.amp.GradScaler('cuda', enabled=(type == torch.float16))

model_config = PiLlamaConfig()
tokenizer = Tokenizer('pi_tokenizer.model')
model = PiLlama(model_config, tokenizer)
model.to(device)
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])
raw_model = model.module if ddp else model
num_params = sum(p.numel() for p in model.parameters())
if master_process:
    print(f"Number of parameters: {num_params}")

total_batch_size = 65536
grad_accum_steps = total_batch_size // (B * T * ddp_world_size)
max_step = 6164

optimizer = raw_model.configure_optimizer(device_type=device)
scheduler = CosineLRScheduler(
    max_lr = model_config.max_lr, 
    min_lr = 0.1 * model_config.max_lr, 
    warmup_step = 10, 
    max_step = max_step
)

train_loader = DataLoaderLite( 
    batch_size = B, 
    sequence_length = T, 
    process_rank = ddp_rank, 
    num_processes = ddp_world_size,
    split = 'train'
)
val_loader = DataLoaderLite( 
    batch_size = B, 
    sequence_length = T, 
    process_rank = ddp_rank, 
    num_processes = ddp_world_size,
    split = 'validation',
)

log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, 'log.txt')
    
for step in range(0):
    t0 = time.time()
    loss_accum = 0.0
    # Save checkpoint every 250 steps
    if (step % 250 == 0 or step == max_step - 1) and master_process:
        checkpoint_path = os.path.join(log_dir, f'pillama_{step}.pt')
        checkpoint = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'step': step
        }
        torch.save(checkpoint, checkpoint_path)
        
    model.train()
    for micro_step in range(grad_accum_steps):
        Xb, Yb = train_loader.next_batch()
        Xb, Yb = Xb.to(device), Yb.to(device)
        # Only sync gradients at the end of a step
        if ddp:
            model.require_backward_grad_sync = (micro_step == grad_accum_steps - 1)
        with torch.autocast(device_type=device_type, dtype=type, enabled=use_amp):
            logits, loss = model(Xb, Yb)
        loss /= grad_accum_steps # Gradients accumulation
        loss_accum += loss.detach() # No need to compute gradients 
        scaler.scale(loss).backward()
    if ddp:
        # Sync the loss between different processes across all machine and all of them receive the same final result
        dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)
    lr = scheduler.get_lr(step) 
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    # Gradient clipping
    scaler.unscale_(optimizer)
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad()
    if device_type == "cuda":
        torch.cuda.synchronize()
    t1 = time.time()
    with torch.no_grad():
        model.eval()
        val_loader.advance(reset=True)
        val_accum = 0.0
        val_accum_steps = 20
        for _ in range(val_accum_steps):
            Xb, Yb = val_loader.next_batch()
            Xb, Yb = Xb.to(device), Yb.to(device)
            with torch.autocast(device_type=device_type, dtype=type, enabled=use_amp):
                _, loss = model(Xb, Yb)
            loss /= val_accum_steps
            val_accum += loss.detach()
        if ddp:
            dist.all_reduce(val_accum, op=dist.ReduceOp.AVG)
    if master_process:
        with open(log_file, 'a') as f:
            f.write(f'Step {step}, training loss {loss_accum.item():.6f}, val loss {val_accum.item():.6f}, norm {norm:.4f}\n')
        dt = t1 - t0
        tok_per_sec = train_loader.B * train_loader.T * grad_accum_steps * ddp_world_size / dt
        print(f'Step: {step} | Loss: {loss_accum.item():.6f} | LR: {lr:.4e} | Time: {1000 * dt:.2f}ms | Tok/sec: {tok_per_sec:.2f}')

prompts = ["Once upon a time, "]
generated = model.text_generation(prompts, max_gen_len=128, top_k=50, temperature=1.2, seed=42)
for text in generated:
    print(text)
if ddp:
    destroy_process_group()