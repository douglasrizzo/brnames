import random
from pathlib import Path

import torch
from pytorch_lightning import LightningDataModule
from rich.progress import track
from sklearn.utils import compute_class_weight
from torch.utils.data import DataLoader, Dataset


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

    def __init__(self, datapath: Path, batch_size: int, num_workers: int):
        super().__init__()
        self.datapath = datapath
        self.batch_size = batch_size
        self.num_workers = num_workers

    def prepare_data(self):
        if not self.datapath.exists():
            # fetch file
            import requests

            url = "https://raw.githubusercontent.com/datasets-br/prenomes/master/data/nomes-censos-ibge.csv"
            response = requests.get(url, allow_redirects=True)
            if response.status_code == 200:
                with open(self.datapath, "w", encoding="utf-8") as f:
                    f.write(response.text)

    def setup(self, stage):
        # read the csv, get only names
        with open(self.datapath, "r", encoding="utf-8") as f:
            words = [line.split(",")[0].lower() for line in f.readlines()]
        # remove file header
        words = words[1:]
        print(f"Words before sanitizing: {len(words)}")

        # sanitize data
        expected_vocab = "abcdefghijklmnopqrstuvwxyz"
        for name in track(words, description="Sanitizing"):
            if any(c not in expected_vocab for c in name):
                words.remove(name)
        print(f"Words after sanitizing: {len(words)}")
        words = list(set(words))
        print(f"Words after removing duplicates: {len(words)}")

        shortest = min(len(word) for word in words)
        longest = max(len(word) for word in words)
        print(f"Shortest word: {shortest}, longest word: {longest}")

        # how many letters we'll see to predict the next one
        self.block_size = longest

        text = "".join(words)

        # here are all the unique characters that occur in this text
        chars = sorted(list(set("." + text)))
        self.vocab_size = len(chars)

        # create a mapping from characters to integers
        stoi = {ch: i for i, ch in enumerate(chars)}

        random.shuffle(words)  # shuffle to prevent a bad train/val split
        n = int(0.9 * len(words))  # first 90% of words will be train, rest val
        train_set, val_set = [], []
        n_examples = 0
        # generate tokenized n-grams and put them in the train or val list
        for idx, word in track(enumerate(words), description="n-gramizing"):
            context = [stoi["."]] * self.block_size
            data = train_set if idx <= n else val_set
            for ch in word + ".":
                ix = stoi[ch]
                n_examples += 1
                data.append(context + [ix])
                context = context[1:] + [ix]
        print(f"Number of n-grams: {n_examples}")

        train_set, val_set = torch.tensor(train_set), torch.tensor(val_set)
        # separate last column, containing targets
        self.train_X, self.train_Y = train_set[:, :-1], train_set[:, -1]
        self.val_X, self.val_Y = val_set[:, :-1], val_set[:, -1]

    def train_dataloader(self):
        train_split = NGramDataset(self.train_X, self.train_Y)
        return DataLoader(
            train_split,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            worker_init_fn=_init_loader_seed,
        )

    def val_dataloader(self):
        val_split = NGramDataset(self.val_X, self.val_Y)
        return DataLoader(
            val_split,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            worker_init_fn=_init_loader_seed,
        )

    def compute_class_weights(self) -> torch.Tensor:
        self.prepare_data()
        self.setup(stage="fit")
        return torch.tensor(
            compute_class_weight(
                class_weight="balanced",
                classes=self.train_Y.unique().numpy(),
                y=self.train_Y.numpy(),
            ),
            dtype=torch.float32,
        )


def _init_loader_seed(worker_id):
    random.seed(random.getstate()[1][0] + worker_id)
    # def test_dataloader(self):
    #     test_split = NGramDataset(self.test_X, self.test_Y)
    #     return DataLoader(test_split)
    # def teardown(self):
    #     # clean up after fit or test
    #     # called on every process in DDP
