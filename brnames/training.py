from pathlib import Path
from random import choice
from typing import Any, Dict

import pytorch_lightning as pl
import ray
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from ray import air, tune
from ray.air.integrations.wandb import WandbLoggerCallback
from ray.tune.integration.pytorch_lightning import TuneReportCheckpointCallback
from ray.tune.schedulers import ASHAScheduler

from .data import NGramDataModule
from .model import Transformer


WANDB_PROJECT_NAME = "brnames"
WANDB_GROUP_NAME = "tune"


def train_single(config: Dict[str, Any], data_path: Path, max_epochs: int) -> None:
    datamodule = NGramDataModule(data_path, 512, 8)
    model = Transformer(config)
    # tune automatically logs the values listed in TuneReportCallback, so we don't configure loggers to Lightning
    trainer = pl.Trainer(
        logger=False,
        accelerator="gpu",
        max_epochs=max_epochs,
        val_check_interval=0.5,
        precision=16,
        auto_scale_batch_size=True,
        enable_progress_bar=False,
        callbacks=[
            TuneReportCheckpointCallback(["Loss/Val", "Loss/Train"], on="validation_end"),
            EarlyStopping("Loss/Val", 0.001, 30),
        ],
    )
    trainer.tune(model, datamodule=datamodule)
    trainer.fit(model, datamodule=datamodule)


def train_tune(train_config: Dict[str, Any], param_space: Dict[str, Any]):
    train_fn_with_parameters = tune.with_parameters(
        train_single,
        data_path=train_config["datapath"],
        max_epochs=train_config["max_epochs"],
    )
    scheduler = ASHAScheduler(max_t=train_config["max_epochs"], grace_period=1, reduction_factor=2)
    # configure wandb callback if chosen, logging to TensorBoard is done automatically by tune
    ray_callbacks = (
        [
            WandbLoggerCallback(project=WANDB_PROJECT_NAME, group=WANDB_GROUP_NAME),
        ]
        if train_config["wandb"]
        else []
    )

    tuner = tune.Tuner(
        tune.with_resources(
            train_fn_with_parameters,
            resources={"cpu": 8, "gpu": 1},
        ),
        tune_config=tune.TuneConfig(
            metric="Loss/Val",
            mode="min",
            scheduler=scheduler,
            num_samples=train_config["tune"],
        ),
        run_config=air.RunConfig(
            name="brnames_asha",
            callbacks=ray_callbacks,
            checkpoint_config=air.CheckpointConfig(1, "Loss/Val", "min"),
        ),
        param_space=param_space,
    )

    # NOTE letting Tune start its own cluster is buggy, so its best to manually start it
    # with ray start --head and then init it manually with ray.init(address="auto")
    ray.init(address="auto")
    results = tuner.fit()

    print("Best hyperparameters found were: ", results.get_best_result().config)
