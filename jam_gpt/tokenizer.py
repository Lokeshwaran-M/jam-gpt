

class Tokenizer:
    """
    tokenizer class will tokenize the input interms of charater level 
    based on the the given text_data to get_encoding function it will use vacobalory
    if not it will use the default encodings to encode and decode data
    """
    def __init__(self):
        pass
    
    def get_encoding(self, text_data: str = None):
        if not text_data:
            self.chars = ['\n', ' ', '!', '"', '$', '%', '&', "'", '(', ')', '*', ',', '-', '.', '/', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', ':', ';', '<', '=', '>', '?', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', '[', '\\', ']', '^', '_', '`', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', '{', '|', '~', '£', '©', '»', '—', '™']
            self.n_vocab = len(self.chars)
        else:
            self.chars = sorted(list(set(text_data)))
            self.n_vocab = len(self.chars)
        
        # print("".join(self.chars))
        # print(self.n_vocab)
        self.stoi = {ch: i for i, ch in enumerate(self.chars)}
        self.itos = {i: ch for i, ch in enumerate(self.chars)}

    def encode(self,s:str)->list[int]:
        # encoder: take a string, output a list of integers
        return [self.stoi[c] for c in s]
    
    def decode(self,l:list[int])->str:
        # decoder: take a list of integers, output a string
        return ''.join([self.itos[i] for i in l])
    









