from abc import ABC
from typing import Any, Self

import pytorch_lightning as pl
import torch
from torch import nn
from torch.nn import functional

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
  """One head of self-attention."""

  def __init__(self: Self, n_embd: int, head_size: int, block_size: int, dropout: float) -> None:
    """Initialize a self-attention head layer.

    Args:
        self (Self): the object itself
        n_embd (int): the dimension of the input embedding
        head_size (int): the dimension of each head
        block_size (int): the size of the block
        dropout (float): the dropout rate
    """
    super().__init__()
    self.key = nn.Linear(n_embd, head_size, bias=False)
    self.query = nn.Linear(n_embd, head_size, bias=False)
    self.value = nn.Linear(n_embd, head_size, bias=False)
    self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))

    self.dropout = nn.Dropout(dropout)

  def forward(self: Self, x: torch.Tensor) -> torch.Tensor:
    """A function to perform forward pass of the self-attention mechanism.

    Args:
        self (Self): The instance of the class.
        x (torch.Tensor): Input tensor of size (batch, time-step, channels).

    Returns:
        Output tensor of size (batch, time-step, head size).
    """
    # input of size (batch, time-step, channels)
    # output of size (batch, time-step, head size)
    _b, t, _c = x.shape
    k = self.key(x)  # (B,T,hs)
    q = self.query(x)  # (B,T,hs)
    # compute attention scores ("affinities")
    wei = q @ k.transpose(-2, -1) * k.shape[-1] ** -0.5  # (B, T, hs) @ (B, hs, T) -> (B, T, T)
    wei = wei.masked_fill(self.tril[:t, :t] == 0, float("-inf"))  # (B, T, T)
    wei = functional.softmax(wei, dim=-1)  # (B, T, T)
    wei = self.dropout(wei)
    # perform the weighted aggregation of the values
    v = self.value(x)  # (B,T,hs)
    return wei @ v  # (B, T, T) @ (B, T, hs) -> (B, T, hs)


class MultiHeadAttention(ABC, nn.Module):
  """Multiple heads of self-attention."""

  @staticmethod
  def __closest_numerator(n: int, m: int) -> int:
    """Calculate the closest integer to `n` that is a multiple of `m`.

    Parameters:
        n (int): The number to find the closest multiple of `m` for.
        m (int): The divisor to find the closest multiple of.

    Returns:
        int: The closest integer to `n` that is a multiple of `m`.
    """
    q = n // m
    # 1st possible closest number
    n1 = m * q
    # 2nd possible closest number
    n2 = m * (q + 1) if (n * m) > 0 else m * (q - 1)
    return n1 if abs(n - n1) < abs(n - n2) else n2

  def __init__(self: Self, n_embd: int, n_head: int) -> None:
    """Initialize a multi-head attention layer with multiple heads.

    Args:
        self (Self): the object itself.
        n_embd (int): the embedding dimension.
        n_head (int): the number of attention heads.
    """
    super().__init__()
    if n_embd % n_head != 0:
      best_n_embd = MultiHeadAttention.__closest_numerator(n_embd, n_head)
      msg = f"n_embd ({n_embd}) is not perfectly divisible by n_head ({n_head}). Closest n_embed is {best_n_embd}."
      raise ValueError(msg)

    self.n_embd = n_embd
    self.n_head = n_head
    self.head_size = n_embd // n_head
    print(f"For n_embd ({n_embd}) and n_head ({n_head}), head_size will be {self.head_size}.")


class SequentialMultiHeadAttention(MultiHeadAttention):
  """Multiple heads of self-attention in parallel (in the model architecture), computed sequentially."""

  def __init__(self: Self, n_embd: int, block_size: int, n_head: int, dropout: float) -> None:
    """Initialize a multi-head attention layer with multiple heads that work sequentially.

    Args:
        self (Self): The object itself.
        n_embd (int): The embedding dimension.
        block_size (int): The block size.
        n_head (int): The number of heads.
        dropout (float): The dropout probability.
    """
    super().__init__(n_embd, n_head)
    self.heads = nn.ModuleList([SelfAttentionHead(n_embd, self.head_size, block_size, dropout) for _ in range(n_head)])
    self.proj = nn.Linear(self.head_size * n_head, n_embd)
    self.dropout = nn.Dropout(dropout)

  def forward(self: Self, x: torch.Tensor) -> torch.Tensor:
    """Forward function to perform a concatenation operation on the output of multiple heads.

    Args:
        self (Self): The instance of the class.
        x (torch.Tensor): The input tensor.

    Returns:
        torch.Tensor: The output tensor after applying the concatenation operation and dropout.
    """
    out = torch.cat([h(x) for h in self.heads], dim=-1)
    return self.dropout(self.proj(out))


class ParallelMultiHeadAttention(MultiHeadAttention):
  """Parallel version of the of the multi-head self-attention where Q,K,V for all heads are computed at the same time.

  Based on https://github.com/karpathy/nanoGPT.
  """

  def __init__(self: Self, n_embd: int, block_size: int, n_head: int, dropout: float, flash: bool = True) -> None:
    """Initialize a multi-head attention layer with multiple heads that work in parallel.

    Args:
        self (Self): the module instance
        n_embd (int): the embedding dimension
        block_size (int): the size of the block
        n_head (int): the number of heads
        dropout (float): the dropout probability
        flash (bool): whether to use flash attention (defaults to True).
    """
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
        print("WARNING: flash attention is available, but will use slow attention at user's request.")
    elif flash:
      print(
        "WARNING: flash attention usage was requested, but is not available. "
        "Flash Attention requires PyTorch >= 2.0. Will use slow attention instead."
      )

  def forward(self: Self, x: torch.Tensor) -> torch.Tensor:
    """Perform the forward pass through the layer.

    Args:
        x (torch.Tensor): Input tensor of size (batch, time-step, channels).

    Returns:
        torch.Tensor: Output tensor of size (batch, time-step, head size).
    """
    # input of size (batch, time-step, channels)
    # output of size (batch, time-step, head size)
    b, t, c = x.shape

    # calculate query, key, values for all heads in batch and move head forward to be the batch dim
    q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
    k = k.view(b, t, self.n_head, c // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
    q = q.view(b, t, self.n_head, c // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
    v = v.view(b, t, self.n_head, c // self.n_head).transpose(1, 2)  # (B, nh, T, hs)

    # compute attention scores ("affinities")
    if self.flash:
      # efficient attention using Flash Attention CUDA kernels
      with torch.backends.cuda.sdp_kernel(enable_math=False):
        out = torch.nn.functional.scaled_dot_product_attention(
          q, k, v, attn_mask=None, dropout_p=self.dropout_p, is_causal=True
        )
    else:
      att = q @ k.transpose(-2, -1) * k.shape[-1] ** -0.5  # (B, T, hs) @ (B, hs, T) -> (B, T, T)
      att = att.masked_fill(self.tril[:t, :t] == 0, float("-inf"))  # (B, T, T)
      att = functional.softmax(att, dim=-1)  # (B, T, T)
      att = self.attn_dropout(att)
      # perform the weighted aggregation of the values
      out = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
    out = out.transpose(1, 2).contiguous().view(b, t, c)  # re-assemble all head outputs side by side
    return self.res_dropout(self.proj(out))


class FeedForward(nn.Module):
  """A simple linear layer followed by a non-linearity."""

  def __init__(
    self: Self,
    n_embd: int,
    dropout: float,
    activation: str,
  ) -> None:
    """Initialize the feed-forward layer with the specified parameters.

    Args:
        self (Self): The object instance itself.
        n_embd (int): The number of embeddings.
        dropout (float): The dropout rate.
        activation (str): The activation function name.
    """
    super().__init__()
    self.net = nn.Sequential(
      nn.Linear(n_embd, 4 * n_embd),
      ACTIVATIONS[activation](),
      nn.Linear(4 * n_embd, n_embd),
      nn.Dropout(dropout),
    )

  def forward(self: Self, x: torch.Tensor) -> torch.Tensor:
    """Perform a forward pass through the feed-forward layer.

    Args:
        self (Self): The object instance.
        x (torch.Tensor): The input tensor.

    Returns:
        torch.Tensor: The output tensor.
    """
    return self.net(x)


class Block(nn.Module):
  """Transformer block: communication followed by computation."""

  def __init__(
    self: Self,
    n_embd: int,
    block_size: int,
    n_head: int,
    dropout: float,
    activation: str,
    parallel: bool = False,
  ) -> None:
    """Initialize a Transformer block.

    Args:
        self (Self): the object itself
        n_embd (int): embedding dimension
        block_size (int): size of the block
        n_head (int): the number of heads
        dropout (float): the dropout rate
        activation (str): the activation function
        parallel (bool, optional): flag to indicate parallel processing (default is False)
    """
    # n_embd: embedding dimension, n_head: the number of heads we'd like
    super().__init__()
    self.ln1 = nn.LayerNorm(n_embd)
    if parallel:
      self.sa = ParallelMultiHeadAttention(n_embd, block_size, n_head, dropout)
    else:
      self.sa = SequentialMultiHeadAttention(n_embd, block_size, n_head, dropout)
    self.ln2 = nn.LayerNorm(n_embd)
    self.ffwd = FeedForward(n_embd, dropout, activation)

  def forward(self: Self, x: torch.Tensor) -> torch.Tensor:
    """Perform a forward pass through the block.

    Args:
        self (Self): The object itself.
        x (torch.Tensor): The input tensor.

    Returns:
        torch.Tensor: The output tensor.
    """
    x = x + self.sa(self.ln1(x))
    return x + self.ffwd(self.ln2(x))


class Transformer(pl.LightningModule):
  """A Transformer network implemented as a PyTorch Lightning module."""

  def __init__(
    self: Self,
    config: dict[str, Any],
  ) -> None:
    """Initialize the model with the given configuration parameters.

    Args:
        self (Self): The object itself.
        config (dict[str, Any]): A dictionary containing the configuration parameters.
    """
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
    self.itos = dict(enumerate(config["vocab"]))
    vocab_size = len(config["vocab"])

    # used by Lightning to log graph
    self.example_input_array = torch.zeros((1, config["block_size"]), dtype=torch.long)
    # each token directly reads off the logits for the next token from a lookup table
    self.token_embedding_table = nn.Embedding(vocab_size, config["n_embd"])
    self.position_embedding_table = nn.Embedding(config["block_size"], config["n_embd"])
    self.blocks = nn.Sequential(*[
      Block(
        config["n_embd"],
        config["block_size"],
        config["n_head"],
        config["dropout"],
        config["activation"],
        config["parallel_sa"],
      )
      for _ in range(config["n_layer"])
    ])
    self.layer_norm = nn.LayerNorm(config["n_embd"])  # final layer norm
    self.lm_head = nn.Linear(config["n_embd"], vocab_size)

    # better init, not covered in the original GPT video, but important, will cover in followup video
    self.apply(self._init_weights)

  @staticmethod
  def _init_weights(module: nn.Module) -> None:
    """Initialize the weights of the given module.

    Args:
        module (nn.Module): The module whose weights need to be initialized.
    """
    if isinstance(module, nn.Linear):
      torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)  # Initialize weights from N(0, 0.02)
      if module.bias is not None:
        torch.nn.init.zeros_(module.bias)  # Initialize bias to zeros
    elif isinstance(module, nn.Embedding):
      torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)  # Initialize weights from N(0, 0.02)

  def forward(self: Self, idx: torch.Tensor) -> torch.Tensor:
    """Forward function to process input and return the language model head output.

    Args:
        self (Self): the class instance
        idx (torch.Tensor): input tensor of shape (B,T) containing integers

    Returns:
        torch.Tensor: output tensor of shape (B,T,C) from the language model head
    """
    _b, t = idx.shape
    # idx and targets are both (B,T) tensor of integers
    tok_emb = self.token_embedding_table(idx)  # (B,T,C)
    pos_emb = self.position_embedding_table(torch.arange(t, device=self.device))  # (T,C)
    x = tok_emb + pos_emb  # (B,T,C)
    x = self.blocks(x)  # (B,T,C)
    x = self.layer_norm(x)  # (B,T,C)
    return self.lm_head(x)  # (B,T,vocab_size)

  def training_step(self: Self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:  # noqa: ARG002
    """Takes in a batch of data and the batch index and performs a training step.

    Args:
        batch (tuple[torch.Tensor, torch.Tensor]): A tuple containing the input data (X) and the target labels (Y).
        batch_idx (int): The index of the current batch.

    Returns:
        torch.Tensor: The loss value computed during the training step.
    """
    data, targets = batch
    logits = self(data)
    loss = functional.cross_entropy(logits[:, -1, :], targets)
    self.log("Loss/Train", loss)
    return loss

  def validation_step(self: Self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:  # noqa: ARG002
    """Perform a validation step using the given batch data and batch index.

    Args:
        self (Self): The class instance.
        batch (tuple[torch.Tensor, torch.Tensor]): The input data and targets in a tuple.
        batch_idx (int): The index of the batch.

    Returns:
        torch.Tensor: The loss value calculated during the validation step.
    """
    data, targets = batch
    logits = self(data)
    loss = functional.cross_entropy(logits[:, -1, :], targets)
    self.log("Loss/Val", loss)
    return loss

  def on_validation_epoch_end(self: Self) -> None:
    """Called at the end of each validation epoch.

    This function post-processes the generated words by calling the `posprocess_generated_words` method
    with the generated words from the `generate` method. It then logs the generated words using the `logger`
    object if it has a `log_text` attribute, otherwise it prints the generated words to the console.
    """
    words = self.posprocess_generated_words(self.generate(10))
    if hasattr(self.logger, "log_text"):
      self.logger.log_text(key="samples", columns=["name"], data=[[name] for name in words])
    else:
      print(f"Sample: {', '.join(words)}")

  def configure_optimizers(self: Self) -> dict[str, Any]:
    """Configures the optimizers and learning rate schedulers for the model.

    Returns:
        A dictionary containing the configured optimizer, learning rate scheduler, and monitor.
            - optimizer: The configured optimizer.
            - lr_scheduler: The configured learning rate scheduler.
            - monitor: The name of the monitor.

    Raises:
        ValueError: If an unrecognized optimizer or learning rate scheduler is provided.
    """
    # Configure optimizer
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
      msg = f"Unrecognized optimizer '{self.optimizer}'"
      raise ValueError(msg)

    # Configure learning rate scheduler
    if self.lr_scheduler == "cosine":
      scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=self.max_epochs / 10,
        eta_min=1e-7,
      )
    elif self.lr_scheduler == "reduce_on_plateau":
      scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=self.lr_factor,
        patience=self.lr_patience,
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
        lr_lambda=lambda _: 1.0,
      )
    else:
      msg = f"Unrecognized lr_scheduler '{self.lr_scheduler}'"
      raise ValueError(msg)

    return {
      "optimizer": optimizer,
      "lr_scheduler": scheduler,
      "monitor": "Loss/Val",
    }

  @torch.no_grad()
  def generate(self: Self, n: int = 2) -> torch.Tensor:
    """Generates a sequence of tokens using the current model.

    Args:
        n (int): The number of sequences to generate. Defaults to 2.

    Returns:
        torch.Tensor: The generated sequence of tokens.
    """
    inpute = torch.tensor([[0] * self.block_size] * n, device=self.device)
    # generate until first character is not a . anymore
    while inpute[0, 0] == 0:
      logits = self(inpute)  # (B, C)
      # to generate the next character of ther word, we only care about the last the step in the logits
      logits = logits[:, -1, :]  # (B, C)
      probs = functional.softmax(logits, dim=-1)  # (B, C)
      # sample from the distribution
      idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
      if idx_next.ndim == 1:
        idx_next.unsqueeze_(0)
      # append sampled index to the running sequence
      inpute = torch.cat([inpute[:, 1:], idx_next], dim=1)
    return inpute

  @staticmethod
  def validate_n_head(n_head: list[int], n_embd: int) -> list[int]:
    """Validate the number of heads for a given embedding size.

    Args:
        n_head (List[int]): A list of integers representing the number of heads.
        n_embd (int): An integer representing the embedding size.

    Returns:
        List[int]: A list of valid number of heads that evenly divide the embedding size.
    """
    return [x for x in n_head if n_embd % x == 0]

  def decode(self: Self, int_list: list[int]) -> str:
    """Decode the given list of integers into a string representation.

    Parameters:
        int_list (List[int]): A list of integers to be decoded.

    Returns:
        str: The decoded string representation of the given integers.
    """
    return "".join([self.itos[i] for i in int_list])

  def posprocess_generated_words(self: Self, output: torch.Tensor) -> list[str]:
    """Process a bunch of tokens to turn them into words.

    This function strips the start/end token from the beginning and end of each sequence in the given output tensor and
    returns a list of processed words.

    Parameters:
    - output (torch.Tensor): The output tensor containing the generated words.

    Returns:
    - List[str]: A list of processed words with the start/end tokens removed.
    """
    # strip the start/end token from the beginning and end of each sequence
    samples = [self.decode(out).strip(".") for out in output.tolist()]
    # find the first start/end token in each sequence, if it erxists, and strip everything after it
    return [sample[: sample.find(".")] if sample.find(".") != -1 else sample for sample in samples]
