import json
import os


class Tokenizer:
    """
    tokenizer class will tokenize the input interms of charater level 
    based on the the given text_data to get_encoding function it will use vacobalory
    if not it will use the default encodings to encode and decode data
    """

    def __init__(self):
        pass

    def get_encoding(self, model: str = None):

        self.vocab = Tokenizer.get_char_vocab(f"{model}/vocab.json")
        self.stoi, self.itos = self.vocab
        self.n_vocab = len(self.stoi)
        # print(self.vocab))
        # print(self.n_vocab)

    def set_encoding(self, model: str, data: str):
        """
        toake text or string data and segregate it into vocab and store it in model
        """
        # handelling folder not existerror
        if not os.path.exists(f"./{model}"):
            os.makedirs(f"./{model}")
        Tokenizer.set_char_vocab(f"{model}/vocab.json", data)

    def encode(self, s: str) -> list[int]:
        # encoder: take a string char , output a list of integers
        return [self.stoi[c] for c in s]

    def decode(self, l: list[int]) -> str:
        # decoder: take a list of integers, output a string
        return ''.join([self.itos[str(i)] for i in l])

    @classmethod
    def get_char_vocab(cls, path: str) -> list:
        """
        text data file -> string data -> list vocab (array)
        """
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            stoi = data["stoi"]
            itos = data["itos"]
        return stoi, itos

    @classmethod
    def set_char_vocab(cls, path: str, data: str) -> None:
        """
        string data -> vocab -> text data file (write list(array) into file)
        """
        data_chars = sorted(list(set(data)))
        stoi = {ch: i for i, ch in enumerate(data_chars)}
        itos = {i: ch for i, ch in enumerate(data_chars)}
        vocab = {"stoi": stoi, "itos": itos}
        with open(path, "w", encoding="utf-8") as f:
            json.dump(vocab, f)
            # print("writen data string : ",len(data_string))

    def set_dicts(dict1, dict2, filename):
        data = {'dict1': dict1, 'dict2': dict2}
        with open(filename, 'w') as file:
            json.dump(data, file)

    def get_dicts(filename):
        with open(filename, 'r') as file:
            data = json.load(file)
            dict1 = data['dict1']
            dict2 = data['dict2']
        return dict1, dict2
