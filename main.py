from jam_gpt import Data
from jam_gpt import Tokenizer



tok = Tokenizer()

def main(path):

    text = Data.get(path)
    tok.set_encoding("love",text)
    tok.get_encoding("love")
    enc = tok.encode("test sample $^&~~data")
    print(enc)
    dec = tok.decode(enc)
    print(dec)
    



path = "./data-set/love/texts/The-Art-of-Loving-Erich-Fromm_text.txt"
if __name__ == "__main__":
    main(path)