
import torch
import torch.nn as nn
from torch.nn import functional as F

from jam_gpt import Data
from jam_gpt import Tokenizer
from jam_gpt import Config
from jam_gpt import LM


tok = Tokenizer()


class Model:
    def __init__(self):

        torch.manual_seed(1337)

        self.vocab_size, self.batch_size, self.block_size, self.max_iters, self.eval_interval, self.learning_rate, self.device, self.eval_iters, self.n_embd, self.n_head, self.n_layer, self.dropout = Config.pass_args()

        self.model = None

        self.train_data = None
        self.test_data = None

    def set_model(self, model):
        self.model = model

    def set_data(self, train_data, test_data):
        self.train_data = train_data
        self.test_data = test_data

    def set_parameters(self, args):
        self.vocab_size, self.batch_size, self.block_size, self.max_iters, self.eval_interval, self.learning_rate, self.device, self.eval_iters, self.n_embd, self.n_head, self.n_layer, self.dropout = args

    def get_batch(self, split):
        # generate small batch of data of input -> x and targets -> y
        data = self.train_data if split == 'train' else self.test_data
        ix = torch.randint(len(data) - self.block_size, (self.batch_size,))
        x = torch.stack([data[i:i+self.block_size] for i in ix])
        y = torch.stack([data[i+1:i+self.block_size+1] for i in ix])
        x, y = x.to(self.device), y.to(self.device)
        return x, y

    @torch.no_grad()
    def estimate_loss(self):
        # estimates the loss of model by eval using test data
        out = {}
        self.model.eval()
        for split in ['train', 'val']:
            losses = torch.zeros(self.eval_iters)
            for k in range(self.eval_iters):
                X, Y = self.get_batch(split)
                logits, loss = self.model(X, Y)
                losses[k] = loss.item()
            out[split] = losses.mean()
        self.model.train()
        return out

    def optimize(self):
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=self.learning_rate)

    def train(self, max_iters=None):
        if not max_iters:
            max_iters = self.max_iters
        for iter in range(max_iters):

            # every once in a while evaluate the loss on train and val sets
            if iter % self.eval_interval == 0:
                losses = self.estimate_loss()
                print(
                    f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

            # sample a batch of data
            xb, yb = self.get_batch('train')

            # evaluate the loss
            logits, loss = self.model(xb, yb)
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            self.optimizer.step()


tok = Tokenizer()


def test(path):

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
