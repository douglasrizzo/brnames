import argparse
from argparse import ArgumentParser
from pathlib import Path
from random import shuffle
from typing import Any, Dict

import pytorch_lightning as pl
import ray
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from ray import air, tune
from ray.air.integrations.wandb import WandbLoggerCallback
from ray.tune.integration.pytorch_lightning import TuneReportCallback
from ray.tune.schedulers import ASHAScheduler

from .data import NGramDataModule
from .model import ACTIVATIONS, Transformer

WANDB_PROJECT_NAME = "brnames"
WANDB_GROUP_NAME = "tune"


def get_config() -> Dict[str, Any]:
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
        "--tune",
        dest="tune",
        action="store_true",
        help="Do a hyperparameter search using Ray Tune.",
    )
    group.add_argument(
        "--wandb",
        dest="wandb",
        action="store_true",
        help="Log results to wandb in addition to tensorboard.",
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
    args.datapath = Path(args.datapath)
    if args.gen is not None:
        args.gen = Path(args.gen[0]), int(args.gen[1])
    return vars(args)


def train_single(config: Dict[str, Any], data_path: Path, max_epochs: int) -> None:
    datamodule = NGramDataModule(data_path, 1, 8)
    model = Transformer(config)
    # tune automatically logs the values listed in TuneReportCallback, so we don't configure loggers to Lightning
    trainer = pl.Trainer(
        logger=False,
        gpus=1,
        max_epochs=max_epochs,
        val_check_interval=0.5,
        precision=16,
        auto_scale_batch_size=True,
        enable_progress_bar=False,
        callbacks=[
            TuneReportCallback(["Loss/Val", "Loss/Train"], on="validation_end"),
            ModelCheckpoint(
                filename="epoch={epoch}-val_loss={Loss/Val:.4f}",
                auto_insert_metric_name=False,
                monitor="Loss/Val",
                save_top_k=1,
                mode="min",
            ),
            EarlyStopping("Loss/Val", 0.001, 30), ],
    )
    trainer.tune(model, datamodule=datamodule)
    trainer.fit(model, datamodule=datamodule)


def gen_n_head(spec) -> int:
    valid_n_head = [x for x in [2, 3, 4, 5, 6] if spec.config.n_embd % x == 0]
    shuffle(valid_n_head)
    return valid_n_head[0]


def train_tune(config: Dict[str, Any]):
    tune_config = {
        "optimizer": "adamw",
        "weight_decay": tune.choice([5e-3, 1e-3]),
        "momentum": None,
        "lr": tune.choice([2e-4, 3.5e-4, 5e-4, 6.5e-4, 8e-4]),
        "lr_scheduler": "reduce_on_plateau",
        "lr_factor": 0.2,
        "lr_patience": 10,
        "betas": [0.9, 0.999],
        "vocab_size": 27,
        "block_size": 15,
        "amsgrad": True,
        "parallel_sa": True,
        "n_embd": tune.choice([128, 256, 384, 512]),
        "n_head": tune.sample_from(gen_n_head),
        "dropout": tune.choice([0.1, 0.2, 0.25, 0.3, 0.4, 0.5]),
        "activation": tune.choice(["selu", "silu"]),
        "n_layer": tune.choice([2, 3, 4, 5, 6]), }

    resources_per_trial = {"cpu": 8, "gpu": 1}
    train_fn_with_parameters = tune.with_parameters(
        train_single,
        data_path=config["datapath"],
        max_epochs=config["max_epochs"],
    )

    scheduler = ASHAScheduler(max_t=config["max_epochs"], grace_period=1, reduction_factor=2)

    # configure wandb callback if chosen, logging to TensorBoard is done automatically by tune
    ray_callbacks = ([
        WandbLoggerCallback(project=WANDB_PROJECT_NAME, group=WANDB_GROUP_NAME), ]
                     if config["wandb"] else [])
    tuner = tune.Tuner(
        tune.with_resources(
            train_fn_with_parameters,
            resources=resources_per_trial,
        ),
        tune_config=tune.TuneConfig(
            metric="Loss/Val",
            mode="min",
            scheduler=scheduler,
            num_samples=20,
        ),
        run_config=air.RunConfig(name="brnames_asha", callbacks=ray_callbacks),
        param_space=tune_config,
    )

    # NOTE letting Tune start its own cluster is buggy, so its best to manually start it
    # with ray start --head and then init it manually with ray.init(address="auto")
    ray.init(address="auto")
    results = tuner.fit()

    print("Best hyperparameters found were: ", results.get_best_result().config)


if __name__ == "__main__":
    config = get_config()
    config["vocab_size"] = 27
    config["block_size"] = 15

    if config["gen"] is not None:
        # load checkpoint
        model = Transformer.load_from_checkpoint(config["gen"][0]).to("cuda").eval()
        samples = model.posprocess_generated_words(model.generate(config["gen"][1]))
        with open("sample.txt", "w", encoding="utf-8") as f:
            f.write("\n".join(samples))
        exit()
    else:
        torch.manual_seed(1337)
        if config["tune"]:
            train_tune(config)
        else:
            train_single(config, config["datapath"], config["max_epochs"])
