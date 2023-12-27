from .tokenizer import Tokenizer
import numpy as np

class ByteTokenizer(Tokenizer):
    def num_tokens(self):
        return 256

    def encode(self, text):
        return [int(i) for i in text.encode(encoding='utf-8')]

    def decode(self, token_list):
        return np.array(token_list).astype(dtype='uint8').tobytes().decode('utf-8', 'replace')