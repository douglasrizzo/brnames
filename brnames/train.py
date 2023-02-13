import argparse
from argparse import ArgumentParser
from pathlib import Path
from typing import Dict, List, Literal, Tuple

import torch

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from .data import NGramDataModule
from .model import Transformer


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
        precision: int,
        n_embd: int,
        n_head: int,
        n_layer: int,
        dropout: float,
        optimizer: str,
        weight_decay: float,
        momentum: float,
        lr: float,
        betas: Tuple[float, float],
        lr_patience: int,
        lr_factor: float,
        gen: int,
    ):
        self.datapath = Path(datapath)
        self.batch_size = batch_size
        self.max_iters = max_iters
        self.eval_interval = eval_interval
        self.eval_iters = eval_iters
        self.precision = precision
        self.n_embd = n_embd
        self.n_head = n_head
        self.n_layer = n_layer
        self.dropout = dropout
        self.optimizer = optimizer
        self.weight_decay = weight_decay
        self.momentum = momentum
        self.lr = lr
        self.betas = betas
        self.lr_patience = lr_patience
        self.lr_factor = lr_factor
        self.gen = gen

    def encode(self, s: str):
        """Take a string, output a list of integers."""
        return [self.__stoi[c] for c in s]

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

    def hparams_dict(self):
        return {
            'batch_size': self.batch_size,
            'optimizer': self.optimizer,
            'n_embd': self.n_embd,
            'n_head': self.n_head,
            'n_layer': self.n_layer,
            'amp': self.amp,
            'dropout': self.dropout,
            'weight_decay': self.weight_decay,
            'lr_patience': self.lr_patience,
            'lr_factor': self.lr_factor, }


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
        "--precision",
        type=int,
        default=16,
        help="Floating point precision to use.",
    )
    group.add_argument(
        "--gen",
        type=int,
        default=None,
        help="Generate names into a text file and exit.",
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
    datamodule = NGramDataModule(config.datapath, config.batch_size, 8)
    model = Transformer(
        27,
        15,
        config.n_embd,
        config.n_head,
        config.dropout,
        config.n_layer,
        config.optimizer,
        config.weight_decay,
        config.momentum,
        config.betas,
        config.lr,
        config.lr_patience,
        config.lr_factor,
    )
    trainer = pl.Trainer(
        accelerator='gpu',
        max_epochs=config.max_iters,
        val_check_interval=250,
        precision=config.precision,
        limit_val_batches=200,
        callbacks=[EarlyStopping(monitor="Loss/Val", mode="min", patience=config.lr_patience * 2)],
    )
    trainer.fit(model, datamodule=datamodule)
