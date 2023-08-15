from jam_gpt import data
from jam_gpt import Tokenizer



tok = Tokenizer()

def test(path):

    text = data.get(path)
    tok.get_encoding(text)
    enc = tok.encode("test sample $^&~~data")
    print(enc)
    dec = tok.decode(enc)
    print(dec)



path = "./data-set/love/texts/The-Art-of-Loving-Erich-Fromm_text.txt"
test(path)