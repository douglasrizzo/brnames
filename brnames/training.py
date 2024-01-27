from pathlib import Path
from typing import Any

import pytorch_lightning as pl
import ray
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.progress.rich_progress import RichProgressBar
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from pytorch_lightning.tuner import Tuner
from ray import air, tune
from ray.air.integrations.wandb import WandbLoggerCallback
from ray.tune.integration.pytorch_lightning import TuneReportCheckpointCallback
from ray.tune.schedulers import ASHAScheduler

from .data import NGramDataModule
from .model import Transformer

WANDB_PROJECT_NAME = "brnames"
WANDB_GROUP_NAMES = {"tune": "tune", "standalone": "standalone"}


def train_single(config: dict[str, Any], data_path: Path, max_epochs: int, in_tune: bool = False) -> None:
  """Train a single model using the given configuration and data from a CSV file.

  Parameters:
      config (Dict[str, Any]): The configuration dictionary containing the model parameters.
      data_path (Path): The path to the CSV file containing the training data.
      max_epochs (int): The maximum number of epochs to train the model.
      in_tune (bool, optional): Whether to use Tune for hyperparameter tuning. Defaults to False.
  """
  datamodule = NGramDataModule(data_path, 512, 8)
  model = Transformer(config)

  callbacks: list[pl.Callback] = [
    EarlyStopping("Loss/Val", 0.001, 30),
  ]
  logger = False
  # if using Tune, create its own callback, otherwise configure Lightning loggers
  if in_tune:
    callbacks.append(TuneReportCheckpointCallback(["Loss/Val", "Loss/Train"], on="validation_end"))
  else:
    callbacks.append(RichProgressBar())
    logger = (
      WandbLogger(project=WANDB_PROJECT_NAME, group=WANDB_GROUP_NAMES["standalone"])
      if config["wandb"]
      else TensorBoardLogger(save_dir=".")
    )

  # tune automatically logs the values listed in TuneReportCallback, so we don't configure loggers to Lightning
  trainer = pl.Trainer(
    logger=logger,
    accelerator="gpu",
    max_epochs=max_epochs,
    val_check_interval=0.5,
    precision=16,
    enable_progress_bar=not in_tune,
    callbacks=callbacks,
  )
  # Create a tuner for the trainer
  tuner = Tuner(trainer)
  # Auto-scale batch size by growing it exponentially (default)
  tuner.scale_batch_size(model, datamodule=datamodule)
  trainer.fit(model, datamodule=datamodule)


def train_tune(train_config: dict[str, Any], param_space: dict[str, Any]) -> None:
  """Train and tune a model using the given training configuration and parameter space.

  Parameters:
      train_config (Dict[str, Any]): A dictionary containing the training configuration.
          - datapath (str): The path to the CSV file containing the training data.
          - max_epochs (int): The maximum number of epochs for training.
          - wandb (bool): Whether to use Wandb for logging.
          - tune (int): The number of samples for hyperparameter tuning.
      param_space (Dict[str, Any]): A dictionary containing the parameter space for hyperparameter tuning.
  """
  train_fn_with_parameters = tune.with_parameters(
    train_single,
    data_path=train_config["datapath"],
    max_epochs=train_config["max_epochs"],
    in_tune=True,
  )
  scheduler = ASHAScheduler(max_t=train_config["max_epochs"], grace_period=1, reduction_factor=2)
  # configure wandb callback if chosen, logging to TensorBoard is done automatically by tune
  ray_callbacks = (
    [
      WandbLoggerCallback(project=WANDB_PROJECT_NAME, group=WANDB_GROUP_NAMES["tune"]),
    ]
    if train_config["wandb"]
    else []
  )

  tuner = tune.Tuner(
    tune.with_resources(
      train_fn_with_parameters,
      resources={"cpu": 8, "gpu": 1},  # resources for a single task
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

  # attempt to connect to an existing Ray cluster, otherwise start our own
  try:
    print("Attempting to connect to an existing Ray cluster...")
    ray.init(address="auto")
  except ConnectionError:
    print("No Ray cluster found, starting a new one...")
    ray.init()
  results = tuner.fit()

  print("Best hyperparameters found were: ", results.get_best_result().config)
