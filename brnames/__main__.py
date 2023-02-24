import argparse
from argparse import ArgumentParser
from pathlib import Path
from typing import Optional, Tuple

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint, RichProgressBar
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger

from .data import NGramDataModule
from .model import ACTIVATIONS, Transformer


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
            16,
            384,
            6,
            6,
            0.2,
            "relu",
            "layer",
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
        max_epochs: int,
        precision: int,
        ce_weights: bool,
        n_embd: int,
        n_head: int,
        n_layer: int,
        dropout: float,
        activation: str,
        parallel_sa: bool,
        optimizer: str,
        weight_decay: float,
        momentum: float,
        lr: float,
        lr_scheduler: str,
        betas: Tuple[float, float],
        amsgrad: bool,
        lr_patience: int,
        lr_factor: float,
        es_patience: int,
        gen: Optional[Tuple[str, str]],
    ):
        self.datapath = Path(datapath)
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.precision = precision
        self.ce_weights = ce_weights
        self.n_embd = n_embd
        self.n_head = n_head
        self.n_layer = n_layer
        self.dropout = dropout
        self.activation = activation
        self.parallel_sa = parallel_sa
        self.optimizer = optimizer
        self.weight_decay = weight_decay
        self.momentum = momentum
        self.lr = lr
        self.lr_scheduler = lr_scheduler
        self.betas = betas
        self.amsgrad = amsgrad
        self.lr_patience = lr_patience
        self.lr_factor = lr_factor
        self.es_patience = es_patience
        try:
            if gen is None:
                self.gen = gen
            else:
                self.gen = Path(gen[0]), int(gen[1])
        except:
            print(gen, gen is None)
            raise


def get_config() -> Config:
    parser = ArgumentParser(
        "Language model trainer",
        description="Train a language model on a list of names to predict more names.",
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
        "--logger",
        type=str,
        default="wandb",
        choices=["wandb", "tensorboard"],
        help="Logger to use, either `wandb` or `tensorboard`.",
    )
    group.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Number of training examples utilized in one iteration.",
    )
    group.add_argument(
        "--max_epochs",
        type=int,
        default=50000,
        help="Maximum number of training epochs.",
    )
    group.add_argument(
        "--precision",
        type=int,
        default=16,
        help="Floating point precision to use.",
    )
    group.add_argument(
        "--gen",
        nargs=2,
        type=str,
        default=None,
        help=
        "Generate names into a text file and exit. Arg 1 is the path to the checkpoint file, arg 2 is the n of samples ot generate.",
    )
    group.add_argument(
        "--ce_weights",
        dest="ce_weights",
        action="store_true",
        help="Compute class weights for cross-entropy function using the training data.",
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
    group.add_argument(
        "--activation",
        type=str,
        choices=ACTIVATIONS.keys(),
        default="relu",
        help="Activation function to used by the feed-forward modules inside each block.",
    )
    group.add_argument(
        "--sequential-sa",
        dest="parallel_sa",
        action="store_false",
        help="Use sequential implementation of multi-head self-attention.",
        default=True,
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
        help="Initial learning rate",
    )
    group.add_argument(
        "--lr_scheduler",
        type=str,
        default="reduce_on_plateau",
        help="Learning rate scheduler",
    )
    group.add_argument(
        "--lr_patience",
        type=int,
        default=10,
        help="ReduceLROnPlateau patience (evaluation steps)",
    )
    group.add_argument(
        "--lr_factor",
        type=float,
        default=0.2,
        help="Learning rate scheduler multiplicative factor",
    )
    group.add_argument(
        "--es_patience",
        type=int,
        default=20,
        help="Early stopping patience (evaluation steps)",
    )
    group.add_argument(
        "--betas",
        nargs=2,
        type=float,
        default=[0.9, 0.999],
        help="Beta values for the Adam family of algorithms",
    )
    group.add_argument(
        "--amsgrad",
        dest="amsgrad",
        action="store_true",
        help="Use AMSGrad variant of optimizer, if available.",
    )
    args = parser.parse_args()
    return Config(**vars(args))


if __name__ == "__main__":
    config = get_config()

    if config.gen is not None:
        # load checkpoint
        model = Transformer.load_from_checkpoint(config.gen[0]).to("cuda").eval()
        samples = model.posprocess_generated_words(model.generate(config.gen[1]))
        with open("sample.txt", "w", encoding="utf-8") as f:
            f.write("\n".join(samples))
        exit()

    torch.manual_seed(1337)
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
        config.activation,
        config.parallel_sa,
        config.amsgrad,
        datamodule.compute_class_weights() if config.ce_weights else None,
        config.lr_scheduler,
    )
    trainer = pl.Trainer(
        logger=WandbLogger(project="brnames", path="./logs/wandb", log_model="all")
        if config.logger == "wandb" else TensorBoardLogger(save_dir="./logs/tensorboard", log_graph=True),
        accelerator="gpu",
        max_epochs=config.max_epochs,
        val_check_interval=0.5,
        precision=config.precision,
        auto_scale_batch_size=True,
        callbacks=[
            RichProgressBar(),
            EarlyStopping(
                monitor="Loss/Val",
                mode="min",
                patience=config.es_patience,
            ),
            ModelCheckpoint(
                filename="epoch={epoch}-val_loss={Loss/Val:.4f}",
                auto_insert_metric_name=False,
                monitor="Loss/Val",
                save_top_k=3,
                mode="min",
            ),
            LearningRateMonitor(), ],
    )
    trainer.tune(model, datamodule=datamodule)
    trainer.fit(model, datamodule=datamodule)