from abc import ABC
from typing import List, Tuple, Optional

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch import nn

ACTIVATIONS = {
    "relu": nn.ReLU,
    "gelu": nn.GELU,
    "leaky_relu": nn.LeakyReLU,
    "sigmoid": nn.Sigmoid,
    "tanh": nn.Tanh, }


class SelfAttentionHead(nn.Module):
    """One head of self-attention"""

    def __init__(self, n_embd: int, head_size: int, block_size: int, dropout: float):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # input of size (batch, time-step, channels)
        # output of size (batch, time-step, head size)
        B, T, C = x.shape
        k = self.key(x)  # (B,T,hs)
        q = self.query(x)  # (B,T,hs)
        # compute attention scores ("affinities")
        wei = q @ k.transpose(-2, -1) * k.shape[-1]**-0.5  # (B, T, hs) @ (B, hs, T) -> (B, T, T)
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
        self.heads = nn.ModuleList([
            SelfAttentionHead(n_embd, self.head_size, block_size, dropout) for _ in range(n_head)])
        self.proj = nn.Linear(self.head_size * n_head, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out


class ParallelMultiHeadAttention(MultiHeadAttention):
    """
    Parallelized version of the of the multi-head self-attention from https://github.com/karpathy/nanoGPT,
    in which Q,K,V for all heads are computed at the same time.
    """

    def __init__(self, n_embd: int, block_size: int, n_head: int, dropout: float):
        super().__init__(n_embd, n_head)
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(n_embd, 3 * n_embd, bias=False)
        self.register_buffer(
            "tril",
            torch.tril(torch.ones(block_size, block_size)).view(1, 1, block_size, block_size))

        self.attn_dropout = nn.Dropout(dropout)
        self.proj = nn.Linear(n_embd, n_embd)
        self.res_dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor):
        # input of size (batch, time-step, channels)
        # output of size (batch, time-step, head size)
        B, T, C = x.shape

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)

        # compute attention scores ("affinities")
        wei = q @ k.transpose(-2, -1) * k.shape[-1]**-0.5  # (B, T, hs) @ (B, hs, T) -> (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))  # (B, T, T)
        wei = F.softmax(wei, dim=-1)  # (B, T, T)
        wei = self.attn_dropout(wei)
        # perform the weighted aggregation of the values
        out = wei @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        out = out.transpose(1, 2).contiguous().view(B, T,
                                                    C)  # re-assemble all head outputs side by side
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

    def forward(self, x: torch.Tensor):
        return self.net(x)


class Block(nn.Module):
    """Transformer block: communication followed by computation"""

    def __init__(self,
                 n_embd: int,
                 block_size: int,
                 n_head: int,
                 dropout: float,
                 activation: str,
                 parallel: bool = False):
        # n_embd: embedding dimension, n_head: the number of heads we'd like
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        if parallel:
            self.sa = ParallelMultiHeadAttention(n_embd, block_size, n_head, dropout)
        else:
            self.sa = SequentialMultiHeadAttention(n_embd, block_size, n_head, dropout)
        self.ln2 = nn.LayerNorm(n_embd)
        self.ffwd = FeedForward(n_embd, dropout, activation)

    def forward(self, x: torch.Tensor):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class Transformer(pl.LightningModule):

    itos = {i: ch for i, ch in enumerate(sorted(list(set(".abcdefghijklmnopqrstuvwxyz"))))}

    def __init__(
        self,
        vocab_size: int,
        block_size: int,
        n_embd: int,
        n_head: int,
        dropout: float,
        n_layer: int,
        optimizer: str,
        weight_decay: float,
        momentum: float,
        betas: Tuple[float, float],
        lr: float,
        lr_patience: int,
        lr_factor: float,
        activation: str = "relu",
        parallel_sa: bool = True,
        amsgrad: bool = False,
        ce_weights: Optional[torch.Tensor] = None,
        lr_scheduler:str="reduce_on_plateau"
    ):
        super().__init__()
        self.save_hyperparameters()
        self.optimizer = optimizer
        self.weight_decay = weight_decay
        self.momentum = momentum
        self.lr = lr
        self.betas = betas
        self.amsgrad = amsgrad
        self.lr_patience = lr_patience
        self.lr_factor = lr_factor
        self.block_size = block_size
        self.activation = activation
        self.parallel_sa = parallel_sa
        # used by Lightning to log graph
        self.example_input_array = torch.zeros((1, block_size), dtype=torch.long)
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[
            Block(n_embd, block_size, n_head, dropout, activation, parallel_sa)
            for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)  # final layer norm
        self.lm_head = nn.Linear(n_embd, vocab_size)
        self.ce_weights = ce_weights
        self.lr_scheduler = lr_scheduler

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

    def forward(self, idx: torch.Tensor):
        B, T = idx.shape
        # idx and targets are both (B,T) tensor of integers
        tok_emb = self.token_embedding_table(idx)  # (B,T,C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=self.device))  # (T,C)
        x = tok_emb + pos_emb  # (B,T,C)
        x = self.blocks(x)  # (B,T,C)
        x = self.ln_f(x)  # (B,T,C)
        logits = self.lm_head(x)  # (B,T,vocab_size)
        return logits

    def training_step(self, batch, batch_idx):
        X, Y = batch
        logits = self(X)
        # train/val steps run on different devices, so we need to move the class weights to the same device
        loss = F.cross_entropy(logits[:, -1, :], Y, weight=self.ce_weights.to(self.device)if self.ce_weights is not None else None)
        self.log("Loss/Train", loss.item())
        return loss

    def validation_step(self, batch, batch_idx):
        X, Y = batch
        logits = self(X)
        # train/val steps run on different devices, so we need to move the class weights to the same device
        loss = F.cross_entropy(logits[:, -1, :], Y, weight=self.ce_weights.to(self.device)if self.ce_weights is not None else None)
        self.log("Loss/Val", loss.item())
        return loss

    def validation_epoch_end(self, outputs) -> None:
        words = self.posprocess_generated_words(self.generate(10))
        if hasattr(self.logger, 'log_text'):
            self.logger.log_text(key="samples", columns=["name"], data=[[name] for name in words])
        else:
            print(f"Sample: {', '.join(words)}")

    def configure_optimizers(self):
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
                T_max=self.max_epochs/10,
                eta_min=1e-7,
            )
        elif self.lr_scheduler =="reduce_on_plateau":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                "min",
                self.lr_factor,
                self.lr_patience,
                threshold=1e-3,
                threshold_mode="abs",
                min_lr=1e-6,)
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
        else:
            raise ValueError(f"Unrecognized lr_scheduler '{self.lr_scheduler}'")
        
        return {
            "optimizer":
            optimizer,
            "lr_scheduler": scheduler,
            "monitor":
            "Loss/Val", }

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
    def decode(l: List[int]):
        """Take a list of integers, output a string"""
        return "".join([Transformer.itos[i] for i in l])

    @staticmethod
    def posprocess_generated_words(output: torch.Tensor) -> List[str]:
        samples = [Transformer.decode(out).strip(".") for out in output.tolist()]
        samples = [
            sample[:sample.find(".")] if sample.find(".") != -1 else sample for sample in samples]

        return samples
