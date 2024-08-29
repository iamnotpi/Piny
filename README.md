# Piny
## Introduction
Piny is a small language model that is capable of telling stories.
## Model architecture 
Piny is a 37M language model trained on TinyStories dataset. Its architecture adapts the original Transformer architecture, with these modifications:
- Grouped Query Attention (GQA) without bias
- SwiGLU 
- RMSNorm instead of LayerNorm
- RoPE 
- Pre-activation
## Hyperparameters
The model hyperparameters are listed in the table below.
| **Hyperparameter**            | **Value**  |
|-------------------------------|------------|
| **Vocab Size**           | 16,384     |
| **Model Dim**           | 768        |
| **FFN Dim** | 2,048  |
| **Number of  Heads** | 16         |
| **Number of KV Heads** | 4      |
| **Number of Layers**          | 4          |
| **RoPE Base**                 | 10,000     |
## Training details
Piny was trained for approximately 6 epochs using the AdamW optimizer. The optimizer was configured with \($\beta_1$ = 0.9\), \($\beta_2$ = 0.95\), and a weight decay of 0.1. The initial learning rate was set to 9e-4.

Training began with a linear warm-up phase for the first 50 steps, gradually increasing the learning rate to its peak. After the warm-up, a cosine learning rate scheduler was employed to decay the learning rate, reducing it to 10% of the initial value over 6,000 steps. Once this threshold was reached, the model continued training at this reduced learning rate until the completion of the training process.
## Result
The model achieved a cross-entropy loss of 1.2891 on the validation set, demonstrating its capability to generate concise and coherent short stories.
## Example completion
- **Prompt:** "Once"
- **Model Completion:**
> upon a time, there was a little boy named Timmy. Timmy loved to play outside in the park. One day, Timmy found a whistle on the ground. He picked it up and blew it, but it wouldn't make any noise. Timmy thought it was a silly whistle, but he still wanted to keep playing with it. Timmy decided to throw the whistle up high and catch it. He ran around the park, pretending he was a superhero. But as he threw the whistle up high, he saw a big dog staring at him. Timmy got scared and dropped the whistle. The dog ran away, but Timmy was sad that he lost his whistle. From that day on, Timmy learned to be more careful with his toys. He still liked to play outside, but he never threw a whistle again. And whenever he saw a silly dog, he just made sure to be very careful.
- **Prompt:** "The quick brown fox jumps over the lazy dog. "
- **Model Completion:**
>The little dog was scared and ran away. The old fox ran after the fox, waving his claws. He wanted to eat the old fox. The little fox was sad, but he still had his apples in his bag. He wanted to restore the apples to the old fox. The old fox saw the juicy apples and smiled. He was happy to have some apples. The fox ate the apples and felt full. He was glad that the old fox had helped restore the apples.
## Limitations
Due to GPU constraints, we trained the model only once without performing any hyperparameter tuning. As a result, the validation loss is relatively high compared to even smaller models used by the authors of the TinyStories dataset. While the model demonstrates some ability to generate stories, the outputs lack coherence, as illustrated in the example above. 
## TODOs
- [x] Implement RoPE
- [x] Implement SwiGLU activation function
- [x] Change hyperparameters (except for optimizer-related)
- [x] Build Llama tokenizer (using SentencePiece instead of tiktoken for user-defined vocab size)
- [x] Implement training loop 
- [x] Implement gradients accumulation for training with larger batch size
- [x] Implement DDP
- [x] Add graphs for hyperparameters tuning
- [x] Load and encode TinyStories dataset
- [x] Build a dataloader
- [ ] Add data shuffling
- [ ] Tune hyperparams
## References
- https://github.com/karpathy/nano-llama31
- https://github.com/karpathy/nanoGPT
