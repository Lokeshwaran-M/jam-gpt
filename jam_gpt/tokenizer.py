import json
import os
from jam_gpt import Data


class Tokenizer:
    """
    tokenizer class will tokenize the input interms of charater level 
    based on the the given text_data to get_encoding function it will use vacobalory
    if not it will use the default encodings to encode and decode data
    """
    def __init__(self):
        pass

    def get_encoding(self, model:str=None):
        
        self.chars = Tokenizer.get_char_vocab(f"{model}/vocab.json")
        self.n_vocab = len(self.chars)
    
        # print("".join(self.chars))
        # print(self.n_vocab)
        self.stoi = {ch: i for i, ch in enumerate(self.chars)}
        self.itos = {i: ch for i, ch in enumerate(self.chars)}
    
    def set_encoding(self,model:str,data:str):
        """
        toake text or string data and segregate it into vocab and store it in model
        """
        # handelling folder not existerror
        if not os.path.exists(f"./{model}"):
            os.makedirs(f"./{model}")
        Tokenizer.set_char_vocab(f"{model}/vocab.json",data)

    def encode(self,s:str)->list[int]:
        # encoder: take a string, output a list of integers
        return [self.stoi[c] for c in s]
    
    def decode(self,l:list[int])->str:
        # decoder: take a list of integers, output a string
        return ''.join([self.itos[i] for i in l])
    
    @classmethod
    def get_char_vocab(cls, path: str) -> list:
        """
        text data file -> string data -> list vocab (array)
        """
        with open(path, "r", encoding="utf-8") as f:
            text_data = json.load(f)
        return text_data    

    @classmethod
    def set_char_vocab(cls, path: str, data: str) -> None:
        """
        string data -> vocab -> text data file (write list(array) into file)
        """
        data_chars= sorted(list(set(data)))
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data_chars,f)
            # print("writen data string : ",len(data_string))









