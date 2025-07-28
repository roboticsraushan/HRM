from __future__ import annotations
from contextlib import nullcontext

import torch
import torch.nn.functional as F
from torch import nn, Tensor, tensor, is_tensor, stack
from torch.nn import Embedding, Linear, Module, ModuleList
from torch.utils._pytree import tree_map

from einops import rearrange, repeat

from x_transformers import Encoder

from adam_atan2_pytorch import AdamAtan2

# helper functions

def exists(v):
    return v is not None

def first(arr):
    return arr[0]

def last(arr):
    return arr[-1]

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
        reasoning_steps = 2,                          # N in the paper - the number of forward evals for the last network (highest hierarchy) above
        relative_period: int | tuple[int, ...] = 2,   # the relative period for each network evaluation call to the one just previous - in the paper, they do 2 networks with a period of 2
        ignore_index = -1
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

        self.num_networks = len(self.networks)
        assert self.num_networks > 0

        # setup how frequent each network is called
        # the first network (lowest in the hierarchy) should be called every iteration

        num_higher_networks = self.num_networks - 1

        if not isinstance(relative_period, tuple):
            relative_period = (relative_period,) * num_higher_networks

        # implied that first network is called always

        if len(relative_period) == (self.num_networks - 1):
            relative_period = (1, *relative_period)

        # for the paper, they did (low: 1, high: 2) -

        assert len(relative_period) == self.num_networks and relative_period[0] == 1

        self.evaluate_networks_at = tensor(relative_period).cumprod(dim = -1).tolist()

        self.reasoning_steps = reasoning_steps
        self.lowest_steps_per_reasoning_step = last(self.evaluate_networks_at)

        # output

        self.to_pred = Linear(dim, num_tokens, bias = False)

        # loss related

        self.ignore_index = ignore_index

    def forward(
        self,
        seq,
        hiddens: tuple[Tensor, ...] | None = None,
        *,
        labels = None,
        detach_hiddens = True,
        one_step_grad = True
    ):

        if detach_hiddens:
            hiddens = tree_map_tensor(hiddens, lambda t: t.detach())

        # seq to input tokens

        tokens = self.to_input_embed(seq)

        # handle hiddens

        if not exists(hiddens):
            hiddens = torch.zeros_like(tokens)
            hiddens = repeat(hiddens, '... -> num_networks ...', num_networks = self.num_networks)

        assert len(hiddens) == self.num_networks

        # network as they proposed - following figure 4

        def evaluate_network_(
            network: Module,
            network_index
        ):
            all_hiddens = (tokens, *hiddens)
            network_input = all_hiddens[network_index:-1]

            # combine with mean pool for now

            combined_input = stack(network_input).mean(dim = 0)

            # forward

            next_hidden = network(combined_input)

            # store hiddens at appropriate hierarchy, low to highest

            hiddens[network_index] = next_hidden

        # maybe 1-step

        context = torch.no_grad if one_step_grad else nullcontext

        with context():
            for index in range(self.reasoning_steps * self.lowest_steps_per_reasoning_step - 1):
                iteration = index + 1

                for network_index, (network, evaluate_network_at) in enumerate(zip(self.networks, self.evaluate_networks_at)):

                    if not divisible_by(iteration, evaluate_network_at):
                        continue

                    evaluate_network_(network, network_index)

        # 1-step gradient learning

        for network_index, network in enumerate(self.networks):

            evaluate_network_(network, network_index)

        # to output prediction, using the hiddens from the highest hierarchy

        highest_hidden = last(hiddens)

        pred = self.to_pred(highest_hidden)

        # if labels passed in, cross entropy loss

        if not exists(labels):
            return pred, hiddens

        loss = F.cross_entropy(
            rearrange(pred, 'b n l -> b l n'),
            labels,
            ignore_index = self.ignore_index
        )

        return loss, (pred, hiddens)
