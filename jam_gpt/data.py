import torch
from . import config


class Data:
    """ just a data preprocessor """
    def __init__(self) -> None:
        pass

    @classmethod
    def get(cls, path: str) -> str:
        """
        text data file -> string data

        """
        with open(path, "r", encoding="utf-8") as f:
            text_data = f.read()
        return text_data

    @classmethod
    def set(cls, path: str, data: str) -> None:
        """
        string data -> text data file
        """
        with open(path, "w", encoding="utf-8") as f:
            f.write(data)
            print("writen data : ", len(data))

    @classmethod
    def train_test_split(cls, data, split_percent: int = 90):
        # split the data into train ans test based on split percentage

        tensor_data = torch.tensor(
            data, dtype=torch.long, device=config.device)
        n = int((split_percent/100)*len(tensor_data))
        train_data = tensor_data[:n]
        test_data = tensor_data[n:]

        return [train_data, test_data]


def test(path):
    Data.set(path, "ghsdubvujsjbnjsnjbnsjnbljnlbnls")
    d = Data.get(path)
    print(d)
    Data.set_vocab(path, d)
    l = Data.get_vocab(path)
    print(l)
