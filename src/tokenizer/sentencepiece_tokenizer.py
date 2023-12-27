from tokenizer import Tokenizer
import numpy as np
import sentencepiece as spm

class SentencePieceTokenizer(Tokenizer):
    def __init__(self, model_path):
        self.sp = spm.SentencePieceTokenizer(model_file=model_path)

    def num_tokens(self):
        return self.sp.get_piece_size()

    def encode(self, text):
        return self.encode(text)

    def decode(self, token_list):
        return self.decode(token_list)