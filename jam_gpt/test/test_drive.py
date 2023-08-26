#  testing Data and Tokenizer calsses

from ..data import Data
from ..tokenizer import Tokenizer

tok = Tokenizer()

def t_tokenizer(path,text_input = "test sample data"):

    text = Data.get(path)
    tok.set_encoding("love",text)
    tok.get_encoding("love")
    vocab_size = tok.n_vocab
    print(vocab_size)
    enc = tok.encode(text_input)
    print(enc)
    dec = tok.decode(enc)
    print(dec)

    # # output ::
    # 2
    # [75, 60, 74, 75, 1, 74, 56, 68, 71, 67, 60, 1, 59, 56, 75, 56]
    # test sample data
                        


