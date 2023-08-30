
import os
import pickle
import torch
from . import config


torch.manual_seed(1337)


class Model:
    """ class model to work on lm models """

    def __init__(self):
        [self.vocab_size, self.batch_size, self.block_size, self.max_iters, self.eval_interval, self.learning_rate,
            self.device, self.eval_iters, self.n_embd, self.n_head, self.n_layer, self.dropout] = config.pass_args()

        self.model = None

        self.train_data = None
        self.test_data = None

    def set_parameters(self, args: list):
        [self.vocab_size, self.batch_size, self.block_size, self.max_iters, self.eval_interval, self.learning_rate,
            self.device, self.eval_iters, self.n_embd, self.n_head, self.n_layer, self.dropout] = args

    def get_parameters(self):
        return [self.vocab_size, self.batch_size, self.block_size, self.max_iters, self.eval_interval, self.learning_rate, self.device, self.eval_iters, self.n_embd, self.n_head, self.n_layer, self.dropout]

    def set_model(self, model):
        self.model = model
        self.m = self.model.to(self.device)
        # print the number of parameters in the model
        print("vocab size : ", self.vocab_size)
        print("parameters : ", sum(p.numel()
              for p in self.m.parameters())/1e6, " M")
        return self.m

    def set_data(self, data):
        self.train_data = data[0]
        self.test_data = data[1]

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

    def generate(self, prompt, max_new_tokens=500):
        # generate from the model
        # context = torch.zeros((1, 1), dtype=torch.long, device=self.device)
        tensor_prompt = (torch.tensor(
            prompt, dtype=torch.long, device=self.device)[None, ...])

        return self.m.generate(tensor_prompt, max_new_tokens)[0].tolist()

    def save_model(self, model_name, model_format="bin"):

        # to save model
        if not os.path.exists(f"./{model_name}"):
            os.makedirs(f"./{model_name}")
        path = f"{model_name}/{model_name}.{model_format}"
        if model_format == "bin" or model_format == "pt":
            torch.save(self.model.state_dict(), path)
        elif model_format == "pkl":
            with open(path, 'wb') as f:
                pickle.dump(self.model, f)
        else:
            print(f"given model format : {model_format} is not supported")

        # to save config info
        config.set_args(self.get_parameters())
        config.store_config(f"{model_name}/config.json")

    def load_model(self, model_name, model_format="bin"):
        # to load model
        path = f"{model_name}/{model_name}.{model_format}"
        if model_format == "bin" or model_format == "pt":
            self.model.load_state_dict(torch.load(path))
        elif model_format == "pkl":
            with open(path, 'rb') as f:
                self.model = pickle.load(f)

        # to load config info
        config_data = config.retrieve_config(f"{model_name}/config.json")
        self.set_parameters(config_data["config_args"])
        config.set_args(config_data["config_args"])

        self.model.eval()
        # return self.model
