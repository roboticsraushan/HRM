from __future__ import annotations

import torch
from torch import nn, Tensor, is_tensor
import torch.nn.functional as F
from torch.nn import Embedding, Linear, Module, ModuleList
from torch.utils._pytree import tree_map

from einops import rearrange

from x_transformers import Encoder

from adam_atan2_pytorch import AdamAtan2

# helper functions

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

def divisible_by(num, den):
    return (num % den) == 0

def tree_map_tensor(sample, fn):
    return tree_map(lambda t: t if not is_tensor(t) else fn(t), sample)

# modules

class HRM(Module):
    def __init__(
        self,
        *,
        dim,
        num_tokens,
    ):
        super().__init__()

        self.to_input_embed = Embedding(num_tokens, dim)

        self.to_pred = Linear(dim, num_tokens, bias = False)

    def forward(
        self,
        seq,
        hiddens: tuple[Tensor, ...] | None = None,
        *,
        labels = None,
        detach_hiddens = True
    ):

        if detach_hiddens:
            hiddens = tree_map_tensor(hiddens, lambda t: t.detach())

        tokens = self.to_input_embed(seq)

        pred = self.to_pred(tokens)

        # if labels passed in, cross entropy loss

        if not exists(labels):
            return pred, hiddens

        loss = F.cross_entropy(
            rearrange(pred, 'b n l -> b l n'),
            labels
        )

        return loss, (pred, hiddens)
