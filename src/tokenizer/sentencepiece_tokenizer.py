from .tokenizer import Tokenizer
import numpy as np
import sentencepiece as spm

class SentencePieceTokenizer(Tokenizer):
    def __init__(self, model_path):
        self.sp = spm.SentencePieceProcessor(model_file=model_path)

    def num_tokens(self):
        return self.sp.get_piece_size()

    def encode(self, text):
        return self.sp.encode(text)

    def decode(self, token_list):
        return self.sp.decode(token_list)