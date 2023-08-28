import random
from pathlib import Path
from typing import List, Tuple

from unidecode import unidecode

import torch
from pytorch_lightning import LightningDataModule
from rich.progress import track
from torch.utils.data import DataLoader, Dataset


def get_vocab_block_size(datapath: str) -> Tuple[List[str], int, List[str]]:
    """
    Returns the vocabulary, block size, and list of words from a given data file.

    Args:
        datapath (str): The path to the data file.

    Returns:
        Tuple[List[str], int, List[str]]: A tuple containing the vocabulary as a list of strings,
        the block size as an integer, and the list of words as a list of strings.
    """
    with open(datapath, "r", encoding="utf-8") as f:
        next(f)  # ignore first line
        words = set()
        vocab = set()
        block_size = 0
        while True:
            next_line = f.readline()
            if not next_line:
                break

            next_line = unidecode(next_line.split(",")[0].strip().lower())
            vocab = vocab.union(set(next_line))
            block_size = max(block_size, len(next_line))
            words.add(next_line)
    return sorted(list(vocab) + ["."]), block_size, list(words)


class NGramDataset(Dataset):
    def __init__(self, X: torch.Tensor, Y: torch.Tensor) -> None:
        self.X = X
        self.Y = Y

    def __getitem__(self, idx):
        x = self.X[idx]
        y = self.Y[idx]
        return x, y

    def __len__(self):
        return len(self.X)


class NGramDataModule(LightningDataModule):
    def __init__(self, datapath: Path, batch_size: int, num_workers: int, verbose: bool = True):
        super().__init__()
        self.datapath = datapath
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.verbose = verbose

    def prepare_data(self):
        """Downloads a file from a given URL and saves it to a specified file path."""
        if not self.datapath.exists():
            # fetch file
            import requests

            url = "https://raw.githubusercontent.com/datasets-br/prenomes/master/data/nomes-censos-ibge.csv"
            response = requests.get(url, allow_redirects=True)
            if response.status_code == 200:
                with open(self.datapath, "w", encoding="utf-8") as f:
                    f.write(response.text)

    def setup(self, stage):
        self.vocab, self.block_size, words = get_vocab_block_size(self.datapath)
        self.len_shortest_word = min(len(word) for word in words)

        # here are all the unique characters that occur in this text
        self.vocab_size = len(self.vocab)

        # create a mapping from characters to integers
        stoi = {ch: i for i, ch in enumerate(self.vocab)}

        self.dataset_size = len(words)
        # first 90% of words will be train, rest val
        self.train_size = int(0.9 * self.dataset_size)
        self.test_size = self.dataset_size - self.train_size

        random.shuffle(words)  # shuffle words to prevent a bad train/val split
        train_set, val_set = [], []
        self.total_ngrams = 0
        # generate tokenized n-grams and put them in the train or val list
        for idx, word in track(enumerate(words), description="n-gramizing"):
            context = [stoi["."]] * self.block_size
            data = train_set if idx <= self.train_size else val_set
            for ch in word + ".":
                ix = stoi[ch]
                self.total_ngrams += 1
                data.append(context + [ix])
                context = context[1:] + [ix]

        train_set, val_set = torch.tensor(train_set), torch.tensor(val_set)
        # separate last column, containing targets
        self.train_split = NGramDataset(train_set[:, :-1], train_set[:, -1])
        self.val_split = NGramDataset(val_set[:, :-1], val_set[:, -1])

        if self.verbose:
            print(self)

    def __repr__(self) -> str:
        return (
            f"Vocabulary size: {self.vocab_size}\n"
            f"Words in dataset: {self.dataset_size}\n"
            f"Training set size: {self.train_size}, test set size: {self.test_size}\n"
            f"Shortest word: {self.len_shortest_word}, longest word: {self.block_size}\n"
            f"Number of n-grams: {self.total_ngrams}"
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_split,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            worker_init_fn=_init_loader_seed,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_split,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            worker_init_fn=_init_loader_seed,
        )


def _init_loader_seed(worker_id):
    """Initialize the seed for the loader.

    Parameters:
        worker_id (int): The ID of the worker.
    """
    random.seed(random.getstate()[1][0] + worker_id)
