import json
import os
import shutil

class Tokenizer:
    """
    tokenizer class will tokenize the input interms of charater level 
    based on the the given text_data to get_encoding function it will use vacobalory
    if not it will use the default encodings to encode and decode data
    """

    def __init__(self):
        pass

    def get_encoding(self, model: str = None):

        self.vocab = Tokenizer.get_char_vocab(model)
        self.stoi, self.itos = self.vocab
        self.n_vocab = len(self.stoi)
        # print(self.vocab))
        # print(self.n_vocab)

    def set_encoding(self, model: str, data: str):
        """
        toake text or string data and segregate it into vocab and store it in model
        """
        # handelling folder not existerror

        Tokenizer.set_char_vocab(model, data)

    def encode(self, s: str) -> list[int]:
        # encoder: take a string char , output a list of integers
        enc_list = []
        for c in s:
            if c not in self.stoi:
                self.stoi[c] = max(self.stoi.values())+1
                self.n_vocab += 1
            enc_list.append(self.stoi[c])
        return enc_list

    def decode(self, l: list[int]) -> str:
        # decoder: take a list of integers, output a string
        return ''.join([self.itos[str(i)] for i in l])
    
    @classmethod
    def store_vocab(cls,smodel,md_name: str):  
        """ 
        args : 
        smodel = source model name
        md_name = destination model name
        source model -> destination model
        """
        if not os.path.exists(f"./{md_name}"):
            os.makedirs(f"./{md_name}")
        spath = f"{smodel}/vocab.json" 
        dpath = f"{md_name}/vocab.json"  
        shutil.copy(spath, dpath)
        
    @classmethod
    def get_char_vocab(cls, model: str):
        """
        json file -> dict -> dict,dict
        """
        path =f"{model}/vocab.json"
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            stoi = data["stoi"]
            itos = data["itos"]
        return stoi, itos

    @classmethod
    def set_char_vocab(cls, model: str, data: str) -> None:
        """
        string data -> vocab -> dict,dict -> dict -> json file
        """
        if not os.path.exists(f"./{model}"):
            os.makedirs(f"./{model}")
        path = f"{model}/vocab.json"
        data_chars = sorted(list(set(data)))
        stoi = {ch: i for i, ch in enumerate(data_chars)}
        itos = {i: ch for i, ch in enumerate(data_chars)}
        vocab = {"stoi": stoi, "itos": itos}
        with open(path, "w", encoding="utf-8") as f:
            json.dump(vocab, f)
            # print("writen data string : ",len(data_string))
