

from jam_gpt import Data
from jam_gpt import Tokenizer
from jam_gpt import Config
from jam_gpt import LM
from jam_gpt import Model

tok = Tokenizer()


def test(path):

    # data collection
    data = Data.get(path)

    # tokanization
    tok.set_encoding("test", data)
    tok.get_encoding("test")

    # setting parameters
    args = Config.pass_args()
    args[0] = tok.n_vocab
    print(tok.n_vocab)

    # model genration
    test_model = Model()
    lm = LM()

    test_model.set_parameters(args)
    lm.set_parameters(args)
    
    m = test_model.set_model(lm.BigramLanguageModel())

    test_model.set_data(Data.train_test_split(tok.encode(data)))

    test_model.optimize()

    test_model.train()

    print(tok.decode(test_model.generate()))


path = "data.txt"
test(path)


