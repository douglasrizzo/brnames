import argparse
from argparse import ArgumentParser
from pathlib import Path
from typing import Dict, Iterable, List, Literal, Optional, Tuple, Union

import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from .gpt import GPTLanguageModel


class Config:

    @staticmethod
    def karpathy() -> "Config":
        return Config(
            "data.csv",
            64,
            256,
            5000,
            500,
            200,
            False,
            384,
            6,
            6,
            0.2,
            "adamw",
            5e-3,
            0.9,
            3e-4,
            10,
            2,
            (0.9, 0.999),
        )

    def __init__(
        self,
        datapath: str,
        batch_size: int,
        max_iters: int,
        eval_interval: int,
        eval_iters: int,
        amp: bool,
        n_embd: int,
        n_head: int,
        n_layer: int,
        dropout: float,
        optimizer: str,
        weight_decay: float,
        momentum: float,
        lr: float,
        lr_patience: int,
        lr_factor: float,
        betas: Tuple[float, float],
        gen: bool,
    ):
        self.batch_size = batch_size
        self.max_iters = max_iters
        self.eval_interval = eval_interval
        self.eval_iters = eval_iters
        self.amp = amp
        self.n_embd = n_embd
        self.n_head = n_head
        self.n_layer = n_layer
        self.dropout = dropout
        self.optimizer = optimizer
        self.weight_decay = weight_decay
        self.momentum = momentum
        self.lr = lr
        self.lr_patience = lr_patience
        self.lr_factor = lr_factor
        self.betas = betas
        self.gen = gen

        datapath = Path(datapath)
        if not datapath.exists():
            # fetch file
            import requests

            url = "https://raw.githubusercontent.com/datasets-br/prenomes/master/data/nomes-censos-ibge.csv"
            response = requests.get(url, allow_redirects=True)
            if response.status_code == 200:
                with open(datapath, "w", encoding="utf-8") as f:
                    f.write(response.text)

        # read the csv, get only names
        with open(datapath, "r", encoding="utf-8") as f:
            words = [line.split(",")[0].lower() for line in f.readlines()]
        # remove file header
        words = words[1:]
        print(f"Words before sanitizing: {len(words)}")

        # sanitize data
        expected_vocab = "abcdefghijklmnopqrstuvwxyz"
        for name in tqdm(words, desc="Sanitizing"):
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
        self.__stoi = {ch: i for i, ch in enumerate(chars)}
        self.__itos = {i: ch for i, ch in enumerate(chars)}

        X, Y = [], []
        for word in tqdm(words, desc="n-gramizing"):
            context = [self.__stoi["."]] * self.block_size
            for ch in word + ".":
                ix = self.__stoi[ch]
                X.append(context)
                Y.append(ix)
                context = context[1:] + [ix]

        X = torch.tensor(X)
        Y = torch.tensor(Y)

        print(f"Dataset size: {X.shape[0]}, block size: {X.shape[1]}")

        # Train and test splits
        n = int(0.9 * X.shape[0])  # first 90% will be train, rest val
        self.train_X, self.train_Y = X[:n], Y[:n]
        self.val_X, self.val_Y = X[n:], Y[n:]

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.session_name = f"{self.vocab_size}_{self.block_size}_{n_embd}_{n_head}_{n_layer}"
        self.writer = SummaryWriter(str(Path("runs") / self.session_name))

    def encode(self, s: str):
        """Take a string, output a list of integers."""
        return [self.__stoi[c] for c in s]

    def decode(self, l: List[int]):
        """Take a list of integers, output a string"""
        return "".join([self.__itos[i] for i in l])

    def build_optimizer(
        self,
        model_params: Union[Iterable[torch.Tensor], Dict[str, torch.Tensor]],
    ):
        if self.optimizer == "sgd":
            optimizer = torch.optim.SGD(
                params=model_params,
                lr=self.lr,
                momentum=self.momentum,
                weight_decay=self.weight_decay,
            )
        elif self.optimizer == "adam":
            optimizer = torch.optim.Adam(params=model_params,
                                         lr=self.lr,
                                         betas=self.betas,
                                         weight_decay=self.weight_decay)
        elif self.optimizer == "adamw":
            optimizer = torch.optim.AdamW(params=model_params,
                                          lr=self.lr,
                                          betas=self.betas,
                                          weight_decay=self.weight_decay)
        else:
            raise ValueError(f"Unrecognized optimizer '{self.optimizer}'")
        return optimizer

    # data loading
    def get_batch(self, split: Literal["train", "val"]):
        # generate a small batch of data of inputs x and targets y
        data_X = self.train_X if split == "train" else self.val_X
        data_Y = self.train_Y if split == "train" else self.val_Y
        idx = torch.randint(len(data_X), size=(self.batch_size, ))
        x = data_X[idx]
        y = data_Y[idx]
        x, y = x.to(self.device), y.to(self.device)
        return x, y

    @torch.no_grad()
    def estimate_loss(self, model: torch.nn.Module) -> Dict[Literal["train", "val"], torch.Tensor]:
        out = {}
        model.eval()
        for split in ["train", "val"]:
            losses = torch.zeros(self.eval_iters)
            for k in range(self.eval_iters):
                X, Y = self.get_batch(split)
                _, loss = model(X, Y)
                losses[k] = loss.item()
            out[split] = losses.mean()
        model.train()
        return out

    def posprocess_generated_words(self, output: torch.Tensor) -> List[str]:
        samples = [self.decode(out).strip(".") for out in output.tolist()]
        samples = [
            sample[:sample.find(".")] if sample.find(".") != -1 else sample for sample in samples]

        return samples


def get_config() -> Config:
    parser = ArgumentParser(
        "Language model trainer",
        description="Train a language model on a list of names to predict more names",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    group = parser.add_argument_group("Training parameters")
    group.add_argument(
        "--datapath",
        type=str,
        default="data.csv",
        help="Number of training examples utilized in one iteration.",
    )
    group.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Number of training examples utilized in one iteration.",
    )
    group.add_argument(
        "--max_iters",
        type=int,
        default=5000,
        help="",
    )
    group.add_argument(
        "--eval_interval",
        type=int,
        default=500,
        help="",
    )
    group.add_argument(
        "--eval_iters",
        type=int,
        default=200,
        help="",
    )
    group.add_argument(
        "--amp",
        dest="amp",
        action="store_true",
        help="Enable PyTorch Automatic Mixed Precision (with GradScaler and autocast) during training",
    )
    group.add_argument(
        "--gen",
        dest="gen",
        action="store_true",
        help="Generate a sample of names into a text file and exit.",
    )

    group = parser.add_argument_group("Model parameters")
    group.add_argument(
        "--n_embd",
        type=int,
        default=384,
        help="Number of embedding dimensions",
    )
    group.add_argument(
        "--n_head",
        type=int,
        default=6,
        help="Number of self-attention heads by multi-head self-attention block",
    )
    group.add_argument(
        "--n_layer",
        type=int,
        default=6,
        help="Number of multi-head self-attention blocks in the model",
    )
    group.add_argument(
        "--dropout",
        type=float,
        default=0.2,
        help="Dropout probability (for all layers in the model)",
    )

    group = parser.add_argument_group("Optimizer parameters")
    group.add_argument(
        "--optimizer",
        type=str,
        choices=["sgd", "adam", "adamw"],
        default="adamw",
        help="Select the optimizer method",
    )
    group.add_argument(
        "--weight_decay",
        type=float,
        default=5e-3,
        help="Weight decay",
    )
    group.add_argument(
        "--momentum",
        type=float,
        default=0.9,
        help="Nesterov momentum for SGD",
    )
    group.add_argument(
        "--lr",
        type=float,
        default=3e-4,
        help="Learning rate",
    )
    group.add_argument(
        "--lr_patience",
        type=int,
        default=10,
        help="Learning rate scheduler plateau patience",
    )
    group.add_argument(
        "--lr_factor",
        type=float,
        default=.2,
        help="Learning rate scheduler plateau factor",
    )
    group.add_argument(
        "--betas",
        nargs=2,
        type=float,
        default=[0.9, 0.999],
        help="Beta values for the Adam family of algorithms",
    )
    args = parser.parse_args()
    return Config(**vars(args))


if __name__ == "__main__":
    torch.manual_seed(1337)
    config = get_config()
    model = GPTLanguageModel(
        config.vocab_size,
        config.block_size,
        config.n_embd,
        config.n_head,
        config.dropout,
        config.n_layer,
    )
    model = model.to(config.device)

    # create a PyTorch optimizer
    optimizer = config.build_optimizer(model.parameters())

    # and a learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                              mode="min",
                                                                patience=config.lr_patience,
                                                              verbose=True,
                                                                factor=config.lr_factor)

    ckpt_path = Path("weights")
    ckpt_path.mkdir(exist_ok=True)
    ckpt_path = ckpt_path / f"{config.session_name}.pth"

    last_iter = 0
    if ckpt_path.is_file():
        try:
            state_dicts = torch.load(ckpt_path)
            model.load_state_dict(state_dicts["model"])
            optimizer.load_state_dict(state_dicts["optimizer"])
            lr_scheduler.load_state_dict(state_dicts["lr_scheduler"])
            last_iter = state_dicts["iter"]
            print("Loaded checkpoint file")
            if config.gen:
                sample = config.posprocess_generated_words(
                    model.generate_many(device=config.device, n=1000))
                with open("sample.txt", "w", encoding="utf-8") as f:
                    f.write("\n".join(sample))
                exit()

        except RuntimeError:
            print("Unable to load checkpoint file")

    # print the number of parameters in the model
    print(sum(p.numel() for p in model.parameters()) / 1e6, "M parameters")

    scaler = torch.cuda.amp.GradScaler(enabled=config.amp)

    best_val_loss = float("inf")
    for iter in tqdm(range(config.max_iters)):
        # every once in a while evaluate the loss on train and val sets
        if iter % config.eval_interval == 0 or iter == config.max_iters - 1:
            losses = config.estimate_loss(model)
            print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
            config.writer.add_scalar("Training/Loss", losses["train"], last_iter + iter)
            config.writer.add_scalar("Val/Loss", losses["val"], last_iter + iter)
            lr_scheduler.step(losses["val"])
            # generate from the model, remove leading and trailing . and
            # keep only stuff before the first remaining ., if it exists
            samples = config.posprocess_generated_words(model.generate_many(config.device, n=12))
            samples = " ".join(samples)
            print(f"Samples: {samples}")
            for sample in samples:
                config.writer.add_text("Samples", samples, last_iter + iter)

            if losses["val"] < best_val_loss:
                best_val_loss = losses["val"].item()
                print("Saving best model checkpoint")
                torch.save(
                    {
                        "iter": last_iter + iter,
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "lr_scheduler": lr_scheduler.state_dict(), },
                    ckpt_path,
                )

        # sample a batch of data
        xb, yb = config.get_batch("train")

        # evaluate the loss
        with torch.cuda.amp.autocast(enabled=config.amp):
            logits, loss = model(xb, yb)
        # Scales loss. Calls backward() on scaled loss to create scaled gradients.
        scaler.scale(loss).backward()
        # scaler.step() first unscales the gradients of the optimizer's assigned params.
        # If these gradients do not contain infs or NaNs, optimizer.step() is then called,
        # otherwise, optimizer.step() is skipped.
        scaler.step(optimizer)
        # Updates the scale for next iteration.
        scaler.update()
        # Clear gradient parameters
        optimizer.zero_grad(set_to_none=True)
