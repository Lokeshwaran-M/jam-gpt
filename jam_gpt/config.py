import torch


class Config:
    def __init__(self) -> None:
        # -----------------------------------------------------------
        # hyperparameters

        self.vocab_size = 0
        self.batch_size = 16
        self.block_size = 32
        self.max_iters = 5000
        self.eval_interval = 100
        self.learning_rate = 1e-3
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.eval_iters = 200
        self.n_embd = 64
        self.n_head = 4
        self.n_layer = 4
        self.dropout = 0.0

        # -----------------------------------------------------------

    @classmethod
    def pass_args(cls):
        instance = cls()
        return [instance.vocab_size, instance.batch_size, instance.block_size, instance.max_iters, instance.eval_interval, instance.learning_rate, instance.device, instance.eval_iters, instance.n_embd, instance.n_head, instance.n_layer, instance.dropout]

    @classmethod
    def get_args(cls):
        instance = cls()
        return instance

    @classmethod
    def set_args(cls, args):
        instance = cls()
        instance.vocab_size, instance.batch_size, instance.block_size, instance.max_iters, instance.eval_interval, instance.learning_rate, instance.device, instance.eval_iters, instance.n_embd, instance.n_head, instance.n_layer, instance.dropout = args



def test():
    x = Config.pass_args()
    print(x)
    x[0] = 96
    Config.set_args(x)
    y = Config.pass_args()
    print(y)


    # [0, 16, 32, 5000, 100, 0.001, 'cuda', 200, 64, 4, 4, 0.0]
    # [96, 16, 32, 5000, 100, 0.001, 'cuda', 200, 64, 4, 4, 0.0]

