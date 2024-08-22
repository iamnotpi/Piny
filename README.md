# Pi-Llama
## Introduction
Pi-Llama is a small language model that is capable of telling stories.
## Model architecture 
## Hyperparameters
## TODOs
- [x] Implement RoPE
- [x] Implement SwiGLU activation function
- [x] Change hyperparameters (except for optimizer-related)
- [x] Build Llama tokenizer (using SentencePiece instead of tiktoken for user-defined vocab size)
- [x] Implement training loop 
- [ ] Implement gradients accumulation for training with larger batch size
- [ ] Implement DDP
- [ ] Load and encode TinyStories dataset
- [ ] Build a dataloader
- [ ] Complete README.md