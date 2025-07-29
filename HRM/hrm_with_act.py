from __future__ import annotations

from collections import namedtuple
from contextlib import nullcontext
from random import randrange, random

import torch
import torch.nn.functional as F
from torch import Tensor, tensor, is_tensor, cat, stack, maximum
from torch.nn import Embedding, Linear, Sequential, Module, ModuleList

import einx
from einops import rearrange, repeat
from einops.layers.torch import Rearrange, Reduce

from x_transformers import Encoder, RMSNorm

# constants

Losses = namedtuple('Losses', ['main', 'act'])

# helper functions

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

def last(arr):
    return arr[-1]

def satisfy_prob(prob):
    return random() < prob

def divisible_by(num, den):
    return (num % den) == 0

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
        min_reasoning_steps_epsilon_prob = 0.5,            # they stochastically choose the minimum segment from 2 .. max with this probability, and 1 step the rest of the time
        max_reasoning_steps = 10,
        act_loss_weight = 1.,
        discount_factor = 1.,
        ignore_index = -1,
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

        self.to_logits = Linear(dim, num_tokens, bias = False)

        # Q(continue|halt) for their adaptive computation time setup

        self.discount_factor = discount_factor

        self.act_loss_weight = act_loss_weight

        self.min_reasoning_steps_epsilon_prob = min_reasoning_steps_epsilon_prob
        self.max_reasoning_steps = max_reasoning_steps

        self.to_q_halt_continue = Sequential(
            Reduce('b n d -> b d', 'mean'),
            RMSNorm(dim),
            Linear(dim, 2, bias = False),
            Rearrange('... halt_continue -> halt_continue ...')
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
        max_reasoning_steps = None,
    ):

        return_loss = exists(labels)

        max_reasoning_steps = default(max_reasoning_steps, self.max_reasoning_steps)

        if exists(hiddens) and detach_hiddens:
            hiddens = tuple(h.detach() for h in hiddens)

        # seq to input tokens

        tokens = self.to_input_embed(seq)

        # handle hiddens

        if not exists(hiddens):
            hiddens = torch.zeros_like(tokens)
            hiddens = repeat(hiddens, '... -> num_networks ...', num_networks = self.num_networks)

        assert len(hiddens) == self.num_networks

        # hiddens to a dictionary, avoid some inplace error when updating hidden
 
        hiddens = {index: hidden for index, hidden in enumerate(hiddens)}

        # maybe 1-step

        min_reasoning_steps = self.max_reasoning_steps

        if self.training:
            min_reasoning_steps = randrange(2, max_reasoning_steps + 1) if satisfy_prob(self.min_reasoning_steps_epsilon_prob) else 1

        # variables for storing the predicted q_halt_continue and hiddens for learning

        highest_hiddens = []
        pred_q_halt_continues = []

        # going through the networks

        total_steps = max_reasoning_steps * self.lowest_steps_per_reasoning_step

        for index in range(total_steps):

            iteration = index + 1

            is_reasoning_step_boundary = divisible_by(index, self.lowest_steps_per_reasoning_step)
            num_reasoning_steps = index // self.lowest_steps_per_reasoning_step

            # evaluate all networks depending on their period

            is_last_step = index == (total_steps - 1)
            context = torch.no_grad if one_step_grad and not is_last_step else nullcontext

            with context():
                for network_index, (network, hidden_combine, evaluate_network_at) in enumerate(zip(self.networks, self.hidden_combiners, self.evaluate_networks_at)):

                    if not divisible_by(iteration, evaluate_network_at):
                        continue

                    all_hiddens = (
                        tokens,
                        *hiddens.values()
                    )

                    # combine with mean pool for now

                    combined_input = hidden_combine(all_hiddens, network_index)

                    # forward

                    next_hidden = network(combined_input)

                    # store hiddens at appropriate hierarchy, low to highest

                    hiddens[network_index] = next_hidden

            # adaptive computation time

            if not is_reasoning_step_boundary:
                continue

            highest_hidden = hiddens[self.num_networks - 1]

            q_halt_continue = self.to_q_halt_continue(highest_hidden).sigmoid()

            # not training, assume batch of 1

            q_halt, q_continue = q_halt_continue

            should_halt = q_halt > q_continue

            if not return_loss and should_halt.all():
                break

            if return_loss:

                highest_hidden = hiddens[self.num_networks - 1]

                highest_hiddens.append(highest_hidden)

                pred_q_halt_continues.append(q_halt_continue)

        # to output prediction, using the hiddens from the highest hierarchy

        highest_hidden = hiddens[self.num_networks - 1]

        logits = self.to_logits(highest_hidden)

        # if labels passed in, cross entropy loss

        hiddens = hiddens.values()

        if not return_loss:
            return logits, hiddens

        # get main loss

        main_pred_loss = F.cross_entropy(
            rearrange(logits, 'b n c -> b c n'),
            labels,
            ignore_index = self.ignore_index
        )

        # compute the act loss

        q_halts, q_continues = rearrange(pred_q_halt_continues, 'l halt_continue b -> halt_continue l b')

        highest_hiddens = stack(highest_hiddens) # (l b n d)

        # q halt loss is simply on whether the prediction is correct or not

        with torch.no_grad():
            all_logits = self.to_logits(highest_hiddens)
            is_correct = (all_logits.argmax(dim = -1) == labels).all(dim = -1)

        q_halt_losses = F.binary_cross_entropy(
            q_halts,
            is_correct.float(),
            reduction = 'none'
        )

        # q continue is learned using bellman's on max(q_halt, q_continue) of next reasoning step

        q_max_halt_continue = maximum(q_halts, q_continues)

        q_continue_losses = F.binary_cross_entropy(
            q_continues[:-1],
            q_max_halt_continue[1:] * self.discount_factor, # they use a discount factor of 1., don't understand why yet
            reduction = 'none'
        )

        # average loss for learning the q values

        act_loss = cat((q_halt_losses, q_continue_losses), dim = 0).mean()

        # total loss + loss breakdown

        total_loss = (
            main_pred_loss +
            act_loss * self.act_loss_weight
        )

        loss_breakdown = Losses(main_pred_loss, act_loss)

        return total_loss, hiddens, (logits, loss_breakdown)
