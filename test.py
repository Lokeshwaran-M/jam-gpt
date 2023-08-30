from jam_gpt import Data
from jam_gpt import Tokenizer
from jam_gpt import config
from jam_gpt import lm
from jam_gpt import Model

tok = Tokenizer()

path = "data.txt"

# data collection
data = Data.get(path)

# tokanization
# tok.set_encoding("md-test-01", data)
tok.get_encoding("md-test-01")

# setting parameters
config.vocab_size = tok.n_vocab

# model genration
test_model = Model()

test_model.set_model(lm.BigramLM())
# test_model.set_data(Data.train_test_split(tok.encode(data)))
# test_model.optimize()
# test_model.train()

test_model.load_model("md-test-01")
pmt = tok.encode("i love you")
print(tok.decode(test_model.generate(pmt)))
