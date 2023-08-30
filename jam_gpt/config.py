import json
import torch


# -----------------------------------------------------------#
# hyperparameters

vocab_size = 0
batch_size = 16
block_size = 32
max_iters = 5000
eval_interval = 100
learning_rate = 1e-3
device = "cuda" if torch.cuda.is_available() else "cpu"
eval_iters = 200
n_embd = 64
n_head = 4
n_layer = 4
dropout = 0.0

# -----------------------------------------------------------#


def pass_args():
    return [vocab_size, batch_size, block_size, max_iters, eval_interval, learning_rate, device, eval_iters, n_embd, n_head, n_layer, dropout]


def get_args():
    args = pass_args()
    arg_names = ['vocab_size', 'batch_size', 'block_size', 'max_iters', 'eval_interval',
                 'learning_rate', 'device', 'eval_iters', 'n_embd', 'n_head', 'n_layer', 'dropout']
    print("# -------------------------------------#\n# hyperparameters")
    max_arg_name_length = max(len(arg) for arg in arg_names)
    for arg_name, arg_value in zip(arg_names, args):
        padding = ' ' * (max_arg_name_length - len(arg_name))
        print(f"{arg_name} {padding} :  {arg_value}")
    print("# -------------------------------------#")
    return args


def set_args(args: list):
    global vocab_size, batch_size, block_size, max_iters, eval_interval, learning_rate, device, eval_iters, n_embd, n_head, n_layer, dropout
    [vocab_size, batch_size, block_size, max_iters, eval_interval,
        learning_rate, device, eval_iters, n_embd, n_head, n_layer, dropout] = args


def variables_to_dict():
    config_dict = {
        "vocab_size": vocab_size,
        "batch_size": batch_size,
        "block_size": block_size,
        "max_iters": max_iters,
        "eval_interval": eval_interval,
        "learning_rate": learning_rate,
        "device": device,
        "eval_iters": eval_iters,
        "n_embd": n_embd,
        "n_head": n_head,
        "n_layer": n_layer,
        "dropout": dropout
    }
    return config_dict


def store_config(path, args=pass_args()):
    config_dict = variables_to_dict()
    with open(path, "w") as config_file:
        config_data = {"config_args": args, "config_dict": config_dict}
        json.dump(config_data, config_file)


def retrieve_config(path):
    try:
        with open(path, "r") as config_file:
            config_data = json.load(config_file)
        return config_data
    except FileNotFoundError:
        print("warning : config.json not found using default args")
        config_data = {"config_args": pass_args()}
        return config_data


get_args()
