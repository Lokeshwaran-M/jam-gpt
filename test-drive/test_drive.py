#  testing Data and Tokenizer calsses

from ..data import Data
from ..tokenizer import Tokenizer

tok = Tokenizer()

def t_tokenizer(path,text_input = "test sample data"):

    text = Data.get(path)
    tok.set_encoding("md-test-01",text)
    tok.get_encoding("md-test-01")
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
                        
# # to run :
# from jam_gpt.test import test_drive as td

# td.t_tokenizer("data.txt")

