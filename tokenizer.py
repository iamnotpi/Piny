import os
from sentencepiece import SentencePieceProcessor, SentencePieceTrainer

class Tokenizer:
    def __init__(self, model_path=None):
        self.sp_model = None
        if model_path:
            self.load_model(model_path)

    def load_model(self, model_path):
        if not os.path.isfile(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        self.sp_model = SentencePieceProcessor(model_file=model_path)
        self.vocab_size = self.sp_model.vocab_size()
        self.bos_id = self.sp_model.bos_id()
        self.eos_id = self.sp_model.eos_id()

    def train(self, input_file, model_prefix='pi_tokenizer', vocab_size=16384):
        SentencePieceTrainer.train(
            input=input_file, 
            model_prefix=model_prefix, 
            model_type='bpe', 
            vocab_size=vocab_size, 
            byte_fallback=True,
            split_digits=True
        )
        self.load_model(f"{model_prefix}.model")

    def encode(self, text, bos=False, eos=False):
        if not self.sp_model:
            raise ValueError("Model is not loaded.")
        
        token_ids = self.sp_model.encode_as_ids(text)
        if bos:
            token_ids = [self.bos_id] + token_ids
        if eos:
            token_ids = token_ids + [self.eos_id]
        return token_ids

    def decode(self, token_ids):
        if not self.sp_model:
            raise ValueError("Model is not loaded.")
        
        return self.sp_model.decode_ids(token_ids)