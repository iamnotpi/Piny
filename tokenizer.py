import os
from sentencepiece import SentencePieceProcessor

class Tokenizer:
    def __init__(self, model_path):
        assert os.path.isfile(model_path)
        self.sp_model = SentencePieceProcessor(model_file=model_path)
        self.bos_id: int = self.sp_model.bos_id()
        self.eos_id: int = self.sp_model.eos_id()

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