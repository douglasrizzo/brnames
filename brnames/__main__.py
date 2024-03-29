import argparse
import sys
from argparse import ArgumentParser
from pathlib import Path
from random import choice
from typing import Any

import torch
from ray import tune

from brnames.data import get_vocab_block_size

from .model import ACTIVATIONS, Transformer
from .training import train_single, train_tune


def get_config() -> dict[str, Any]:
  """Retrieve configuration options from command-line arguments and returns them as a dictionary."""
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
    type=int,
    default=None,
    help="If a positive integer is passed, do a hyperparameter search using Ray Tune with the given sample size.",
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
    help=(
      "Generate names into a text file and exit. Arg 1 is the path to the checkpoint file, arg 2 is the n of samples "
      "to generate."
    ),
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
  args.datapath.parent.mkdir(parents=True, exist_ok=True)

  args.vocab, args.block_size, _ = get_vocab_block_size(args.datapath)

  if args.gen is not None:
    args.gen = Path(args.gen[0]), int(args.gen[1])
  return vars(args)


def main() -> None:
  """Execute the main entrypoint for the program.

  It loads the configuration, checks if a checkpoint needs to be loaded, performs model generation, and writes the
  generated samples to a file. If no checkpoint needs to be loaded, it sets the random seed, and either tunes the model
  or trains a single model based on the configuration.
  """
  config = get_config()

  if config["gen"] is not None:
    # load checkpoint
    model = Transformer.load_from_checkpoint(config["gen"][0]).to("cuda").eval()
    samples = model.posprocess_generated_words(model.generate(config["gen"][1]))
    with Path("sample.txt").open(mode="w", encoding="utf-8") as f:
      f.write("\n".join(samples))
    sys.exit()
  else:
    torch.manual_seed(1337)
    if config["tune"] is not None:
      param_space = {
        "optimizer": config["optimizer"],
        "weight_decay": tune.choice([5e-3, 1e-3]),
        "momentum": config["momentum"],
        "lr": tune.choice([2e-4, 3.5e-4, 5e-4, 6.5e-4, 8e-4]),
        "lr_scheduler": "reduce_on_plateau",
        "lr_factor": config["lr_factor"],
        "block_size": config["block_size"],
        "vocab": config["vocab"],
        "lr_patience": config["lr_patience"],
        "betas": config["betas"],
        "amsgrad": config["amsgrad"],
        "parallel_sa": config["parallel_sa"],
        "n_embd": tune.choice([128, 256, 384, 512]),
        "n_head": tune.sample_from(
          lambda spec: choice(Transformer.validate_n_head([2, 3, 4, 5, 6], spec.config.n_embd))
        ),
        "dropout": tune.choice([0.1, 0.2, 0.25, 0.3, 0.4, 0.5]),
        "activation": tune.choice(["selu", "silu"]),
        "n_layer": tune.choice([2, 3, 4, 5, 6]),
      }

      train_tune(config, param_space)
    else:
      train_single(config, config["datapath"], config["max_epochs"])


if __name__ == "__main__":
  main()
