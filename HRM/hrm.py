from __future__ import annotations

import torch
from torch import nn, Tensor, tensor, is_tensor
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
        networks: list[Module | dict],
        *,
        dim,
        num_tokens,
        reasoning_steps = 2,                                    # N in the paper - the number of forward evals for the last network (highest hierarchy) above
        relative_period: int | tuple[int, ...] = 2    # the relative period for each network evaluation call to the one just previous - in the paper, they do 2 networks with a period of 2
    ):
        super().__init__()

        # input

        self.to_input_embed = Embedding(num_tokens, dim)

        # allow for any number of hierarchical modules

        # order in hierarchy should be from low to high

        self.networks = ModuleList([])

        for network in networks:
            if isinstance(network, dict):
                network = Encoder(**network)

            self.networks.append(network)

        assert len(self.networks) > 0

        # setup how frequent each network is called
        # the first network (lowest in the hierarchy) should be called every iteration

        num_higher_networks = len(self.networks) - 1

        if not isinstance(relative_period, tuple):
            relative_period = (relative_period,) * num_higher_networks

        # implied that first network is called always

        if len(relative_period) == (len(self.networks) - 1):
            relative_period = (1, *relative_period)

        # for the paper, they did (low: 1, high: 2) -

        assert len(relative_period) == len(self.networks) and relative_period[0] == 1

        self.evaluate_networks_at = tensor(relative_period).cumprod(dim = -1).tolist()

        self.reasoning_steps = reasoning_steps
        self.lowest_steps_per_reasoning_step = self.evaluate_networks_at[-1]

        # output

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

        # seq to input tokens

        tokens = self.to_input_embed(seq)

        # network as they proposed - following figure 4

        with torch.no_grad():
            for index in range(self.reasoning_steps * self.lowest_steps_per_reasoning_step - 1):
                iteration = index + 1

                for network, evaluate_network_at in zip(self.networks, self.evaluate_networks_at):

                    if not divisible_by(iteration, evaluate_network_at):
                        continue

                    tokens = network(tokens)

        # 1-step gradient learning

        for network in self.networks:
            tokens = network(tokens)

        # to output prediction

        pred = self.to_pred(tokens)

        # if labels passed in, cross entropy loss

        if not exists(labels):
            return pred, hiddens

        loss = F.cross_entropy(
            rearrange(pred, 'b n l -> b l n'),
            labels
        )

        return loss, (pred, hiddens)
