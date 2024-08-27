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
        self.current_shard = 0
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
        if self.current_position + B * T * self.num_processes + 1 > len(self.tokens):
            self.advance() 
        return X, Y


@dataclass
class PiLlamaConfig:
    n_embd: int = 768
    ffn_dim: int = int(8/3 * n_embd)
    block_size: int = 2048
    n_heads: int = 16
    kv_heads: int = 4
    n_layers: int = 4
    vocab_size: int = 16384
    flash_attention: bool = True
    rope_base: int = 10000
    max_lr: int = 9e-4
    max_batch_size: int = 262144

class RMSNorm(nn.Module):
    def __init__(self, n_embd, eps=1e-6):
        super().__init__()
        self.n_embd = n_embd
        self.gain = nn.Parameter(torch.ones((n_embd)))
        self.eps = eps

    def forward(self, x):
        inv_norm = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * inv_norm * self.gain 
        

class KVCache(nn.Module):
    def __init__(self, batch_size, max_seq_length, n_kv_heads, head_size, device):
        super().__init__()
        self.register_buffer("cache_k", torch.zeros((batch_size, n_kv_heads, max_seq_length, head_size), device=device))
        self.register_buffer("cache_v", torch.zeros((batch_size, n_kv_heads, max_seq_length, head_size), device=device))

    def update(self, start_pos, k, v):
        # k shape: (B, n_kv_heads, T, head_size)
        seq_length = k.size(2)
        # Add the new keys and values to the cache
        self.cache_k[:, :, start_pos:start_pos + seq_length] = k
        self.cache_v[:, :, start_pos:start_pos + seq_length] = v
        # Return the updated cache (all keys and values from the beginning up to the current position) 
        k = self.cache_k[:, :, :start_pos + seq_length]    
        v = self.cache_v[:, :, :start_pos + seq_length]
        return k, v
    
def compute_inv_freqs(dim, seq_len, base=10000, device=None):
    inv_freqs = 1.0 / base ** (torch.arange(0, dim, 2, device=device).float() / dim)
    theta = torch.outer(torch.arange(1, seq_len + 1, device=device), inv_freqs)
    pos = torch.repeat_interleave(theta, 2, dim=-1) 
    return pos

def apply_rope(x, inv_freqs):
    # x shape: (B, nh, T, head_size)
    rotated_x = torch.stack((-x[..., 1::2], x[..., ::2]), dim=-1).view(x.shape)
    return x * torch.cos(inv_freqs) + rotated_x * torch.sin(inv_freqs)


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
        # Mask is deprecated, preseved for compatibility with pretrained models
        self.register_buffer("mask", torch.tril(torch.ones((config.block_size, config.block_size))).view(1, 1, config.block_size, config.block_size))
        self.cache = None
    
    def forward(self, x, start_pos, inv_freqs, mask=None):
        # KV cache!!!
        # Given |----cache_len----|----T----|; our model generates the logits for the next token
        # start_pos = cache_len
        B, T, C = x.shape # T is the length of the input sequence 

        # q is the query of the current tokens, k and v are the keys and values from the cache and the current tokens
        # q, k, v of the current tokens
        q = self.query(x).view(B, T, self.config.n_heads, self.head_size).transpose(1, 2) # (B, nh, T, head_size)
        k = self.key(x).view(B, T, self.config.kv_heads, self.head_size).transpose(1, 2) # (B, nkvh, cache_len + T, head_size)
        v = self.value(x).view(B, T, self.config.kv_heads, self.head_size).transpose(1, 2) # (B, nkvh, cache_len + T, head_size)
        
        # RoPE before caching!!!!!
        q = apply_rope(q, inv_freqs)
        k = apply_rope(k, inv_freqs)

        # Update the keys and values with the cache (now containing the keys and values from all previous tokens)
        if self.cache is not None:
            k, v = self.cache.update(start_pos, k, v)

        # Grouped query attention
        k = k.repeat_interleave(self.config.n_heads // self.config.kv_heads, dim=1) # (B, nh, cache_len + T, head_size)
        v = v.repeat_interleave(self.config.n_heads // self.config.kv_heads, dim=1) # (B, nh, cache_len + T, head_size)

        if self.config.flash_attention:
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        else:
            attn = (q @ k.transpose(-1, -2)) * (1.0 / math.sqrt(self.head_size)) # (B, nh, T, cache_len + T)
            if mask is not None:
                attn = attn + mask
            # attn = attn.masked_fill(self.mask[:,:,:T,:T]==0, float('-inf'))
            attn = F.softmax(attn, dim=-1)
            y = attn @ v # (B, nh, T, head_size)
        y = y.transpose(1, 2).contiguous().view(x.shape)
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
    
    def forward(self, x, start_pos, inv_freqs, mask=None):
        x = x + self.attn(self.rmsn_attn(x), start_pos, inv_freqs, mask)
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

        self.inv_freqs = compute_inv_freqs(config.n_embd // config.n_heads, config.block_size, base=config.rope_base, device=self.lm_head.weight.device)

    def forward(self, idx, targets):
        # For use during training
        B, T = idx.size()
        mask = torch.full((T, T), float('-inf'), device=idx.device)
        mask.triu_(diagonal=1)
        inv_freqs = self.inv_freqs[:T]
        x = self.transformer.wte(idx)
        for block in self.transformer.h:
            x = block(x, -1, inv_freqs, mask) # -1 disable the cache
        x = self.transformer.rmsn(x)
        logits = self.lm_head(x) # (B, T, n_embd)
        loss = F.cross_entropy(logits.view(B * T, -1), targets.view(-1))
        return loss 
    
    @torch.inference_mode()
    def forward_inference(self, idx, start_pos):
        # For use during inference
        # start_pos: The starting position in the sequence of the generated tokens
        T = idx.size(1)
        mask = torch.full((T, T), float('-inf'), device=idx.device)
        mask.triu_(diagonal=1)
        # Since we are using KVCache, we need to pad the mask with zeros
        mask = torch.cat([torch.zeros((T, start_pos), device=idx.device), mask], dim=-1) # (T, T + start_pos)
        inv_freqs = self.inv_freqs[start_pos:start_pos + T]
        x = self.transformer.wte(idx)
        for block in self.transformer.h:
            x = block(x, start_pos, inv_freqs, mask)
        x = self.transformer.rmsn(x)
        logits = self.lm_head(x)
        return logits
    
    def configure_optimizer(self, lr=9e-4, betas=(0.9, 0.95), weight_decay=0.1, fused=True, device_type='cuda', master_process=False):
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
    def generate(self, tokens, max_gen_len, top_p, temperature, generator):
        # tokens: List[List[int]]: List of tokenized texts
        # max_gen_len: Maximum length of the generated text
        self.eval()
        device = self.lm_head.weight.device

        # Setup KVCache for each block
        min_prompt_len = min(len(t) for t in tokens)
        max_prompt_len = max(len(t) for t in tokens)
        assert max_prompt_len <= self.config.block_size
        total_len = min(max_prompt_len + max_gen_len, self.config.block_size)
        batch_size = len(tokens)

        for block in self.transformer.h:
            block.attn.cache = KVCache(
                batch_size = batch_size,
                max_seq_length = total_len,
                n_kv_heads = self.config.kv_heads,
                head_size = self.config.n_embd // self.config.n_heads,
                device = device
            )
        
        # Store all the given tokens in a tensor of size (batch_size, total_len)
        tokens_tensor = torch.full((batch_size, total_len), self.tokenizer.pad_id, dtype=torch.long, device=device)
        for i, token_ids in enumerate(tokens):
            tokens_tensor[i, :len(token_ids)] = torch.tensor(token_ids, dtype=torch.long)
        input_mask = tokens_tensor != self.tokenizer.pad_id # Mask the padding tokens
        eos_reached = torch.zeros(batch_size, dtype=torch.bool, device=device)

        prev_pos = 0 # Starting position of the generated tokens
        for cur_pos in range(min_prompt_len, total_len): 
            x = tokens_tensor[:, prev_pos:cur_pos] # Either (B, min_prompt_len) or (B, 1)
            logits = self.forward_inference(x, prev_pos) # Either (B, min_prompt_len) or (B, 1)
            logits = logits[:, -1, :] # (B, 1, vocab_size)
            if temperature > 0:
                logits = logits / temperature
            next_tokens = sample_top_p(logits, top_p, generator) # (B, 1)
            next_tokens = next_tokens.view(-1) # (B,)
            # We don't want the model to generate only eos token and terminate early
            eos_reached |= (next_tokens == self.tokenizer.eos_id) & (~input_mask[:, cur_pos])
            # Update tokens tensor with the newly generated token only if the tokens in the current position are not padding tokens
            next_tokens = torch.where(input_mask[:, cur_pos], tokens_tensor[:, cur_pos], next_tokens)
            tokens_tensor[:, cur_pos] = next_tokens
            prev_pos = cur_pos
            # Stop generate when encounters an EOS token
            if eos_reached.all():
                break

        output = []
        for i, token_ids in enumerate(tokens_tensor.tolist()):
            # Crop to max_gen_len
            token_ids = token_ids[:max_gen_len + len(tokens[i])]
            try:
                eos_pos = token_ids.index(self.tokenizer.eos_id)
                token_ids = token_ids[:eos_pos]
            except:
                pass
            output.append(token_ids)
        
        # Clear cache
        for block in self.transformer.h:
            block.attn.cache = None

        return output
    
    def text_generation(self, text_prompts, max_gen_len=None, top_p=0.9, temperature=1.0, generator=None):
        if max_gen_len is None:
            max_gen_len = self.config.block_size - 1 # -1 for the BOS token that is always prepended
        tokens = [self.tokenizer.encode(text, bos=True, eos=False) for text in text_prompts]
        generated = self.generate(tokens, max_gen_len, top_p, temperature, generator)
        decoded = [self.tokenizer.decode(ids) for ids in generated]
        return decoded
    
    @staticmethod
    def load_checkpoint(checkpoint_path, tokenizer_path='pi_tokenizer.model', device=None):
        assert os.path.isfile(checkpoint_path), f"Checkpoint file not found: {checkpoint_path}"
        checkpoint = torch.load(checkpoint_path, map_location=device)
        tokenizer = Tokenizer(tokenizer_path)
        model = PiLlama(PiLlamaConfig(), tokenizer)
        model.load_state_dict(checkpoint['model'])
        return model
        
def sample_top_p(logits, top_p, generator):
    # logits shape: (B, T, vocab_size)
    probs = F.softmax(logits, dim=-1)
    sorted_probs, sorted_indices = torch.sort(probs, descending=True)
    cum_probs = torch.cumsum(sorted_probs, dim=-1)
    # Shift cum_probs by one position to the right
    mask = cum_probs - sorted_probs > top_p 
    sorted_probs.masked_fill_(mask, 0.0)
    sorted_probs /= sorted_probs.sum(dim=-1, keepdim=True)
    sampled_indices = torch.multinomial(sorted_probs, num_samples=1, generator=generator) # Indices from sorted_probs
    # Match the indices with the original indices from sorted_indices
    sampled_indices = torch.gather(sorted_indices, -1, sampled_indices) 
    return sampled_indices

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

def main():
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

    # Training hyperparameters
    B, T = 32, 2048
    total_batch_size = 262144
    grad_accum_steps = total_batch_size // (B * T * ddp_world_size)
    val_size = 4541735 # Magic number
    val_accum_steps = val_size // (B * T * ddp_world_size)
    max_step = 6164 # Approx. 4 epochs
    warmup_step = 50

    device_type = 'cuda' if device.startswith('cuda') else 'cpu'

    # Using TF32 (if available) for faster matrix multiplication 
    torch.set_float32_matmul_precision('high')

    # Mixed precision training with cuda
    use_amp = device_type == 'cuda'
    if use_amp:
        type = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    else:
        type = torch.float32
    scaler = torch.cuda.amp.GradScaler(enabled=(type == torch.float16))

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

    optimizer = raw_model.configure_optimizer(device_type=device, master_process=master_process)
    scheduler = CosineLRScheduler(
        max_lr = model_config.max_lr, 
        min_lr = 0.1 * model_config.max_lr, 
        warmup_step = warmup_step, 
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
    log_val_file = os.path.join(log_dir, 'log_val.txt')

    for step in range(max_step):
        t0 = time.time()
        loss_accum = 0.0
        # Save checkpoint and eval every 250 steps
        if ((step != 0 and step % 250 == 0) or step == max_step - 1):
            with torch.no_grad():
                model.eval()
                val_loader.advance(reset=True)
                val_accum = 0.0
                for _ in range(val_accum_steps):
                    Xb, Yb = val_loader.next_batch()
                    Xb, Yb = Xb.to(device), Yb.to(device)
                    with torch.autocast(device_type=device_type, dtype=type, enabled=use_amp):
                        loss = model(Xb, Yb)
                    loss /= val_accum_steps
                    val_accum += loss.detach()
                if ddp:
                    dist.all_reduce(val_accum, op=dist.ReduceOp.AVG)
            if master_process:
                with open(log_val_file, 'a') as f:
                    f.write(f'Step {step}, val loss {val_accum.item():.6f}\n')
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
                loss = model(Xb, Yb)
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
        if master_process:
            with open(log_file, 'a') as f:
                f.write(f'Step {step}, training loss {loss_accum.item():.6f}, norm {norm:.4f}\n')
            dt = t1 - t0
            tok_per_sec = train_loader.B * train_loader.T * grad_accum_steps * ddp_world_size / dt
            print(f'Step: {step} | Loss: {loss_accum.item():.6f} | LR: {lr:.4e} | Time: {1000 * dt:.2f}ms | Tok/sec: {tok_per_sec:.2f}')

    if ddp:
        destroy_process_group()

if __name__ == '__main__':
    main()