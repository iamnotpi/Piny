import os
from sentencepiece import SentencePieceTrainer

vocab_size = 16384
input = os.path.join(os.getcwd(), "TinyStories/tinystories_spm.txt")
model_prefix = 'pi_tokenizer'

SentencePieceTrainer.train(
    input = input, 
    model_prefix = model_prefix, 
    model_type = 'bpe', 
    vocab_size = vocab_size, 
    byte_fallback = True,
    split_by_number = True
)