from abc import ABC
from typing import Any, Dict, List, Union

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler

ACTIVATIONS = {
    "relu": nn.ReLU,
    "gelu": nn.GELU,
    "silu": nn.SiLU,
    "selu": nn.SELU,
    "leaky_relu": nn.LeakyReLU,
    "sigmoid": nn.Sigmoid,
    "tanh": nn.Tanh,
}


class SelfAttentionHead(nn.Module):
    """One head of self-attention"""

    def __init__(self, n_embd: int, head_size: int, block_size: int, dropout: float):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # input of size (batch, time-step, channels)
        # output of size (batch, time-step, head size)
        B, T, C = x.shape
        k = self.key(x)  # (B,T,hs)
        q = self.query(x)  # (B,T,hs)
        # compute attention scores ("affinities")
        wei = q @ k.transpose(-2, -1) * k.shape[-1] ** -0.5  # (B, T, hs) @ (B, hs, T) -> (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))  # (B, T, T)
        wei = F.softmax(wei, dim=-1)  # (B, T, T)
        wei = self.dropout(wei)
        # perform the weighted aggregation of the values
        v = self.value(x)  # (B,T,hs)
        out = wei @ v  # (B, T, T) @ (B, T, hs) -> (B, T, hs)
        return out


class MultiHeadAttention(ABC, nn.Module):
    """Multiple heads of self-attention in parallel."""

    @staticmethod
    def __closest_numerator(n: int, m: int) -> int:
        q = n // m
        # 1st possible closest number
        n1 = m * q
        # 2nd possible closest number
        n2 = m * (q + 1) if (n * m) > 0 else m * (q - 1)
        return n1 if abs(n - n1) < abs(n - n2) else n2

    def __init__(self, n_embd: int, n_head: int):
        super().__init__()
        if n_embd % n_head != 0:
            best_n_embd = MultiHeadAttention.__closest_numerator(n_embd, n_head)
            raise ValueError(
                f"n_embd ({n_embd}) is not perfectly divisible by n_head ({n_head}). Closest n_embed is {best_n_embd}."
            )

        self.n_embd = n_embd
        self.n_head = n_head
        self.head_size = n_embd // n_head
        print(f"For n_embd ({n_embd}) and n_head ({n_head}), head_size will be {self.head_size}.")


class SequentialMultiHeadAttention(MultiHeadAttention):
    """Multiple heads of self-attention in parallel (in the model architecture), computed sequentially."""

    def __init__(self, n_embd: int, block_size: int, n_head: int, dropout: float):
        super().__init__(n_embd, n_head)
        self.heads = nn.ModuleList(
            [SelfAttentionHead(n_embd, self.head_size, block_size, dropout) for _ in range(n_head)]
        )
        self.proj = nn.Linear(self.head_size * n_head, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out


class ParallelMultiHeadAttention(MultiHeadAttention):
    """
    Parallelized version of the of the multi-head self-attention from https://github.com/karpathy/nanoGPT,
    in which Q,K,V for all heads are computed at the same time.
    """

    def __init__(
        self, n_embd: int, block_size: int, n_head: int, dropout: float, flash: bool = True
    ):
        super().__init__(n_embd, n_head)
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(n_embd, 3 * n_embd, bias=False)
        self.register_buffer(
            "tril",
            torch.tril(torch.ones(block_size, block_size)).view(1, 1, block_size, block_size),
        )

        self.dropout_p = dropout

        self.attn_dropout = nn.Dropout(dropout)
        self.proj = nn.Linear(n_embd, n_embd)
        self.res_dropout = nn.Dropout(dropout)

        self.flash = hasattr(torch.nn.functional, "scaled_dot_product_attention") and flash
        if not self.flash:
            # causal mask to ensure that attention is only applied to the left in the input sequence
            self.register_buffer(
                "tril",
                torch.tril(torch.ones(block_size, block_size)).view(1, 1, block_size, block_size),
            )
            self.tril: torch.Tensor  # unnecessary, this is to appease linters

        if hasattr(torch.nn.functional, "scaled_dot_product_attention"):
            if flash:
                print("WARNING: using flash attention.")
            else:
                print(
                    "WARNING: flash attention is available, but will use slow attention at user's request."
                )
        elif flash:
            print(
                "WARNING: flash attention usage was requested, but is not available. "
                "Flash Attention requires PyTorch >= 2.0. Will use slow attention instead."
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # input of size (batch, time-step, channels)
        # output of size (batch, time-step, head size)
        B, T, C = x.shape

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)

        # compute attention scores ("affinities")
        if self.flash:
            # efficient attention using Flash Attention CUDA kernels
            out = torch.nn.functional.scaled_dot_product_attention(
                q, k, v, attn_mask=None, dropout_p=self.dropout_p, is_causal=True
            )
        else:
            att = (
                q @ k.transpose(-2, -1) * k.shape[-1] ** -0.5
            )  # (B, T, hs) @ (B, hs, T) -> (B, T, T)
            att = att.masked_fill(self.tril[:T, :T] == 0, float("-inf"))  # (B, T, T)
            att = F.softmax(att, dim=-1)  # (B, T, T)
            att = self.attn_dropout(att)
            # perform the weighted aggregation of the values
            out = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        out = (
            out.transpose(1, 2).contiguous().view(B, T, C)
        )  # re-assemble all head outputs side by side
        out = self.res_dropout(self.proj(out))
        return out


class FeedForward(nn.Module):
    """A simple linear layer followed by a non-linearity"""

    def __init__(
        self,
        n_embd: int,
        dropout: float,
        activation: str,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            ACTIVATIONS[activation](),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Block(nn.Module):
    """Transformer block: communication followed by computation"""

    def __init__(
        self,
        n_embd: int,
        block_size: int,
        n_head: int,
        dropout: float,
        activation: str,
        parallel: bool = False,
    ):
        # n_embd: embedding dimension, n_head: the number of heads we'd like
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        if parallel:
            self.sa = ParallelMultiHeadAttention(n_embd, block_size, n_head, dropout)
        else:
            self.sa = SequentialMultiHeadAttention(n_embd, block_size, n_head, dropout)
        self.ln2 = nn.LayerNorm(n_embd)
        self.ffwd = FeedForward(n_embd, dropout, activation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class Transformer(pl.LightningModule):

    itos = {i: ch for i, ch in enumerate(sorted(list(set(".abcdefghijklmnopqrstuvwxyz"))))}

    def __init__(
        self,
        config: Dict[str, Any],
    ):
        super().__init__()
        self.save_hyperparameters()
        self.activation = config["activation"]
        self.amsgrad = config["amsgrad"]
        self.betas = config["betas"]
        self.block_size = config["block_size"]
        self.lr = config["lr"]
        self.lr_factor = config["lr_factor"]
        self.lr_patience = config["lr_patience"]
        self.lr_scheduler = config["lr_scheduler"]
        self.momentum = config["momentum"]
        self.optimizer = config["optimizer"]
        self.parallel_sa = config["parallel_sa"]
        self.weight_decay = config["weight_decay"]
        self.itos = {i: ch for i, ch in enumerate(config["vocab"])}
        vocab_size = len(config["vocab"])

        # used by Lightning to log graph
        self.example_input_array = torch.zeros((1, config["block_size"]), dtype=torch.long)
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, config["n_embd"])
        self.position_embedding_table = nn.Embedding(config["block_size"], config["n_embd"])
        self.blocks = nn.Sequential(
            *[
                Block(
                    config["n_embd"],
                    config["block_size"],
                    config["n_head"],
                    config["dropout"],
                    config["activation"],
                    config["parallel_sa"],
                )
                for _ in range(config["n_layer"])
            ]
        )
        self.ln_f = nn.LayerNorm(config["n_embd"])  # final layer norm
        self.lm_head = nn.Linear(config["n_embd"], vocab_size)

        # better init, not covered in the original GPT video, but important, will cover in followup video
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            # NOTE not as good as N(0,0.02) init
            # torch.nn.init.kaiming_normal_(module.weight, nonlinearity=self.activation, a=0.1)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        B, T = idx.shape
        # idx and targets are both (B,T) tensor of integers
        tok_emb = self.token_embedding_table(idx)  # (B,T,C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=self.device))  # (T,C)
        x = tok_emb + pos_emb  # (B,T,C)
        x = self.blocks(x)  # (B,T,C)
        x = self.ln_f(x)  # (B,T,C)
        logits = self.lm_head(x)  # (B,T,vocab_size)
        return logits

    def training_step(self, batch, batch_idx) -> torch.Tensor:
        X, Y = batch
        logits = self(X)
        loss = F.cross_entropy(logits[:, -1, :], Y)
        self.log("Loss/Train", loss)
        return loss

    def validation_step(self, batch, batch_idx) -> torch.Tensor:
        X, Y = batch
        logits = self(X)
        loss = F.cross_entropy(logits[:, -1, :], Y)
        self.log("Loss/Val", loss)
        return loss

    # def validation_epoch_end(self, outputs) -> None:
    #     words = self.posprocess_generated_words(self.generate(10))
    #     if hasattr(self.logger, "log_text"):
    #         self.logger.log_text(key="samples", columns=["name"], data=[[name] for name in words])
    #     else:
    #         print(f"Sample: {', '.join(words)}")

    def configure_optimizers(
        self,
    ) -> Dict[str, Union[Optimizer, _LRScheduler, torch.optim.lr_scheduler.ReduceLROnPlateau, str]]:
        if self.optimizer == "sgd":
            optimizer = torch.optim.SGD(
                params=self.parameters(),
                lr=self.lr,
                momentum=self.momentum,
                weight_decay=self.weight_decay,
            )
        elif self.optimizer == "adam":
            optimizer = torch.optim.Adam(
                params=self.parameters(),
                lr=self.lr,
                betas=self.betas,
                weight_decay=self.weight_decay,
                amsgrad=self.amsgrad,
            )
        elif self.optimizer == "adamw":
            optimizer = torch.optim.AdamW(
                params=self.parameters(),
                lr=self.lr,
                betas=self.betas,
                weight_decay=self.weight_decay,
                amsgrad=self.amsgrad,
            )
        else:
            raise ValueError(f"Unrecognized optimizer '{self.optimizer}'")

        if self.lr_scheduler == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=self.max_epochs / 10,
                eta_min=1e-7,
            )
        elif self.lr_scheduler == "reduce_on_plateau":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                "min",
                self.lr_factor,
                self.lr_patience,
                threshold=1e-3,
                threshold_mode="abs",
                min_lr=1e-6,
            )
        elif self.lr_scheduler == "exponential":
            scheduler = torch.optim.lr_scheduler.ExponentialLR(
                optimizer,
                gamma=self.lr_factor,
            )
        elif self.lr_scheduler == "cosine_restarts":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer,
                T_0=30,
                eta_min=1e-7,
            )
        elif self.lr_scheduler == "constant":
            scheduler = torch.optim.lr_scheduler.LambdaLR(
                optimizer,
                lambda _: 1.0,
            )
        else:
            raise ValueError(f"Unrecognized lr_scheduler '{self.lr_scheduler}'")

        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "monitor": "Loss/Val",
        }

    @torch.no_grad()
    def generate(self, n: int = 2):
        inpute = torch.tensor([[0] * self.block_size] * n, device=self.device)
        # generate until first character is not a . anymore
        while inpute[0, 0] == 0:
            logits = self(inpute)  # (B, C)
            # to generate the next character of ther word, we only care about the last the step in the logits
            logits = logits[:, -1, :]  # (B, C)
            probs = F.softmax(logits, dim=-1)  # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
            if idx_next.ndim == 1:
                idx_next.unsqueeze_(0)
            # append sampled index to the running sequence
            inpute = torch.cat([inpute[:, 1:], idx_next], dim=1)
        return inpute

    @staticmethod
    def validate_n_head(n_head: List[int], n_embd: int) -> List[int]:
        return [x for x in n_head if n_embd % x == 0]

    def decode(self, int_list: List[int]) -> str:
        """Take a list of integers, output a string"""
        return "".join([self.itos[i] for i in int_list])

    def posprocess_generated_words(self, output: torch.Tensor) -> List[str]:
        # strip the start/end token from the beginning and end of each sequence
        samples = [self.decode(out).strip(".") for out in output.tolist()]
        # find the first start/end token in each sequence, if it erxists, and strip everything after it
        samples = [
            sample[: sample.find(".")] if sample.find(".") != -1 else sample for sample in samples
        ]

        return samples
