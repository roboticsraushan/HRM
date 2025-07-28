from __future__ import annotations
from contextlib import nullcontext

import torch
import torch.nn.functional as F
from torch import Tensor, tensor, is_tensor, cat, stack
from torch.nn import Embedding, Linear, Sequential, Module, ModuleList
from torch.utils._pytree import tree_map

from einops import rearrange, repeat
from einops.layers.torch import Rearrange, Reduce

from x_transformers import Encoder, RMSNorm

from adam_atan2_pytorch import AdamAtan2

# helper functions

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

def last(arr):
    return arr[-1]

def divisible_by(num, den):
    return (num % den) == 0

def tree_map_tensor(sample, fn):
    return tree_map(lambda t: t if not is_tensor(t) else fn(t), sample)

# combining hiddens across hierarchies

class CombineHiddens(Module):
    def __init__(
        self,
        dim,
        num_hiddens_to_concat
    ):
        super().__init__()
        self.num_hiddens_to_concat = num_hiddens_to_concat

        self.norms = ModuleList([RMSNorm(dim) for _ in range(num_hiddens_to_concat)])

        self.to_combined = Linear(dim * self.num_hiddens_to_concat, dim, bias = False)

    def forward(
        self,
        hiddens: list[Tensor],
        hierarchy_index
    ):
        hiddens_to_concat = hiddens[hierarchy_index:]

        assert len(hiddens_to_concat) == self.num_hiddens_to_concat

        normed = tuple(norm(t) for norm, t in zip(self.norms, hiddens))

        concatted = cat(normed, dim = -1)

        return self.to_combined(concatted)

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
        min_reasoning_steps = 1,
        max_reasoning_steps = 10,
        act_binary_ce_loss_weight = 1.,
        ignore_index = -1
    ):
        super().__init__()

        # input

        self.to_input_embed = Embedding(num_tokens, dim)

        # allow for any number of hierarchical modules

        # order in hierarchy should be from low to high

        self.networks = ModuleList()

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

        # combining hiddens

        self.hidden_combiners = ModuleList([CombineHiddens(dim, self.num_networks + 1 - network_index) for network_index in range(self.num_networks)])

        # output

        self.to_pred = Linear(dim, num_tokens, bias = False)

        # Q(continue|halt) for their adaptive computation time setup

        self.act_binary_ce_loss_weight = act_binary_ce_loss_weight

        self.min_reasoning_steps = min_reasoning_steps
        self.max_reasoning_steps = max_reasoning_steps

        self.to_q_continue_halt = Sequential(
            Reduce('b n d -> b d', 'mean'),
            RMSNorm(dim),
            Linear(dim, 2, bias = False),
            Rearrange('... continue_halt -> continue_halt ...')
        )

        # loss related

        self.ignore_index = ignore_index

    def forward(
        self,
        seq,
        hiddens: tuple[Tensor, ...] | None = None,
        *,
        labels = None,
        detach_hiddens = True,
        one_step_grad = True,
        reasoning_steps = None
    ):

        reasoning_steps = default(reasoning_steps, self.reasoning_steps)

        if detach_hiddens:
            hiddens = tree_map_tensor(hiddens, lambda t: t.detach())

        # seq to input tokens

        tokens = self.to_input_embed(seq)

        # handle hiddens

        if not exists(hiddens):
            hiddens = torch.zeros_like(tokens)
            hiddens = repeat(hiddens, '... -> num_networks ...', num_networks = self.num_networks)

        assert len(hiddens) == self.num_networks

        # hiddens to a dictionary, avoid some inplace error when updating hidden
 
        hiddens = {index: hidden for index, hidden in enumerate(hiddens)}

        # network as they proposed - following figure 4

        def evaluate_network_(
            network: Module,
            hidden_combine: Module,
            network_index
        ):

            all_hiddens = (
                tokens,
                *[hiddens[i] for i in range(self.num_networks)]
            )

            # combine with mean pool for now

            combined_input = hidden_combine(all_hiddens, network_index)

            # forward

            next_hidden = network(combined_input)

            # store hiddens at appropriate hierarchy, low to highest

            hiddens[network_index] = next_hidden

        # maybe 1-step

        context = torch.no_grad if one_step_grad else nullcontext

        with context():
            for index in range(reasoning_steps * self.lowest_steps_per_reasoning_step - 1):

                iteration = index + 1

                for network_index, (network, hidden_combine, evaluate_network_at) in enumerate(zip(self.networks, self.hidden_combiners, self.evaluate_networks_at)):

                    if not divisible_by(iteration, evaluate_network_at):
                        continue

                    evaluate_network_(network, hidden_combine, network_index)

                # adaptive computation time

                is_reasoning_step_boundary = divisible_by(index, reasoning_steps)
                num_reasoning_steps = index // reasoning_steps

                if is_reasoning_step_boundary and num_reasoning_steps > self.min_reasoning_steps:

                    highest_hidden = hiddens[self.num_networks - 1]

                    q_continue, q_halt = self.to_q_continue_halt(highest_hidden).sigmoid()

                    should_continue = q_halt > q_continue

        # 1-step gradient learning

        for network_index, (network, hidden_combine) in enumerate(zip(self.networks, self.hidden_combiners)):

            evaluate_network_(network, hidden_combine, network_index)

        # to output prediction, using the hiddens from the highest hierarchy

        highest_hidden = hiddens[self.num_networks - 1]

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
