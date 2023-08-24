
import torch

from jam_gpt import Data
from jam_gpt import Tokenizer
from jam_gpt import Config
from jam_gpt import LM
from jam_gpt import Model

tok = Tokenizer()


def gen_model(path):

    # data collection
    data = Data.get(path)

    # tokanization
    tok.set_encoding("love", data)
    tok.get_encoding("love")

    # setting parameters
    args = Config.pass_args()
    args[0] = tok.n_vocab

    # model generation
    gen = Model()

    gen.set_parameters(args)
    LM.set_parameters(args)
    gen.set_model(LM.BigramLanguageModel)

    gen.set_data(Data.train_test_split(tok.encode(data)))

    m = gen.model.to(gen.device)

    gen.optimize()

    gen.train()

   # generate from the model
    context = torch.zeros((1, 1), dtype=torch.long, device=gen.device)

    print(tok.decode(m.generate(context, max_new_tokens=500)[0].tolist()))





path = "./data-set/love/texts/The-Art-of-Loving-Erich-Fromm_text.txt"

if __name__ == "__main__":
    gen_model(path)