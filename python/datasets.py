import csv
from os import PathLike
from pathlib import Path
from typing import Union

import torch
from torch.utils.data import Dataset


class FizzBuzzDataset(Dataset):
    def __init__(self, data_file: str, label_file: str) -> None:
        self.data = self.read_data(Path(data_file).absolute())
        self.label = self.read_label(Path(label_file).absolute())

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.data[idx], self.label[idx]

    @staticmethod
    def read_data(f: Union[str, PathLike]) -> list[torch.Tensor]:
        data = []
        with open(f, newline="") as inf:
            reader = csv.reader(inf)
            for line in reader:
                data.append(torch.tensor([int(i) for i in line], dtype=torch.float32))
        return data

    @staticmethod
    def read_label(f: Union[str, PathLike]) -> list[torch.Tensor]:
        labels = []
        with open(f, newline="") as inf:
            reader = csv.reader(inf)
            for line in reader:
                labels.append(torch.tensor(int(line[0])))
        return labels


training_data = FizzBuzzDataset("../data/train_data.csv", "../data/train_label.csv")
test_data = FizzBuzzDataset("../data/test_data.csv", "../data/test_label.csv")
