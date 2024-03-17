from abc import ABC, abstractmethod

class Tokenizer(ABC):
    @abstractmethod
    def vocab_size(self):
        raise NotImplementedError

    @abstractmethod
    def encode(text):
        raise NotImplementedError

    @abstractmethod
    def decode(token_list):
        raise NotImplementedError