import torch
import torch.nn as nn
from torch.nn import functional as F 
from jam_gpt import Tokenizer 

tok = Tokenizer()

class Model :
    def __init__(self,data:str) -> None:

        self.data = data
        # hyperparameters
        self.batch_size 
        self.block_size
        self.max_iters
        self.eval_interval
        self.learning_rate
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.eval_iters
        self.n_embd
        self.n_head
        self.n_layer
        self.dropout

    def train_test_split_percent(self,split_percent:int=90):
        # split the data into train ans test based on split percentage
        # split_percent -> persentage of split
        self.tensor_data = torch.tensor(tok.encode(self.data),dtype=torch.long)
        n = int((split_percent/100)*len(self.data))
        self.train_data = self.data[:n]
        self.test_data = self.data[n:]

    def get_batch(split):
        # generate small batch of data of input -> x and targets -> y
        pass

    @torch.no_grad()
    def estimate_loss():
        # estimates the loss of model by eval using test data
        pass

    







