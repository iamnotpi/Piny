import os
from datasets import load_dataset
from tokenizer import Tokenizer
import multiprocessing as mp
import numpy as np
from tqdm import tqdm

shard_size = int(1e8) # 100M tokens per shard
local_dir = 'TinyStories'
tokenizer_path = 'pi_tokenizer.model'
tokenizer = Tokenizer(tokenizer_path)

DATA_CACHE_DIR = os.path.join(os.getcwd(), local_dir)
os.makedirs(DATA_CACHE_DIR, exist_ok=True)

ts = load_dataset("roneneldan/TinyStories")

def tokenize(doc):
    tokens = tokenizer.encode(doc, bos=True, eos=True)
    tokens_np = np.array(tokens)
    assert (0 <= tokens_np).all() and (tokens_np < 2 ** 16).all(), "Token dictionary to large for uint16"
    tokens_np_uint16 = tokens_np.astype(np.uint16)
    return tokens_np_uint16

def write_datafile(filename, tokens_np):
    np.save(filename, tokens_np)

def main():
    nprocs = max(1, os.cpu_count() // 2)
    for split, split_dataset in ts.items():
        with mp.Pool(nprocs) as pool:
            shard_index = 0
            # Preallocate buffer
            all_tokens_np = np.zeros((shard_size,))
            token_count = 0
            progress_bar = None
            
            for tokens in pool.imap(tokenize, split_dataset['text'], chunksize=16):
                # Still enough place for new tokens
                if token_count + len(tokens) < shard_size:
                    # Add tokens to shard
                    all_tokens_np[token_count:token_count + len(tokens)] = tokens
                    token_count += len(tokens)
                    # Update progress bar
                    if progress_bar is None:
                        progress_bar = tqdm(total=shard_size, unit="tokens", desc=f"Shard {shard_index}")
                    progress_bar.update(len(tokens)) 
                else:
                    filename = os.path.join(DATA_CACHE_DIR, f'tinystories_{split}_{shard_index}')
                    remainder = shard_size - token_count # How many spaces left in the current shard?
                    progress_bar.update(remainder)
                    all_tokens_np[token_count:token_count + remainder] = tokens[:remainder]
                    write_datafile(filename, all_tokens_np)
                    shard_index += 1
                    progress_bar = None
                    token_count = len(tokens) - remainder
                    all_tokens_np[0:token_count] = tokens[remainder:]
            if token_count != 0:
                filename = os.path.join(DATA_CACHE_DIR, f"tinystories_{split}_{shard_index}")
                write_datafile(filename, all_tokens_np[:token_count])
    
if __name__ == "__main__":
    main()