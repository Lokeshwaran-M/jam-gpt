

from jam_gpt import Data, Tokenizer, Config, LM, Model

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


