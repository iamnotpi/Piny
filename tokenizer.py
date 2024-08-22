import os
from sentencepiece import SentencePieceProcessor, SentencePieceTrainer

class Tokenizer:
    def __init__(self, model_path):
        assert os.path.isfile(model_path)
        self.sp_model = SentencePieceProcessor(model_file=model_path)
        self.vocab_size = self.sp_model.vocab_size()
        self.bos_id: int = self.sp_model.bos_id()
        self.eos_id: int = self.sp_model.eos_id()

    def train(self, input, model_prefix='pi_tokenizer', vocab_size=16384):
        SentencePieceTrainer.train(
            input = input, 
            model_prefix = model_prefix, 
            model_type = 'bpe', 
            vocab_size = vocab_size, 
            byte_fallback = True,
            split_by_number = True
        )
        self.vocab_size = vocab_size

    def encode(self, text, bos, eos):
        t = self.sp_model.encode_as_ids(text)
        # Whether to prepend the beginning-of-sequence token
        if bos: 
            t = [self.bos_id] + t
        # Whether to prepend the end-of-sequence token
        if eos:
            t = t + [self.eos_id]
        return t

    def decode(self, idx):
        return self.sp_model.decode_ids(idx)