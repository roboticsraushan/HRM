import torch
from torch.nn import Module, ModuleList

from einops import rearrange

# helper functions

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

# modules

class Input(Module):
    def __init__(self):
        super().__init__()

class SlowHighLevelRecurrent(Module):
    def __init__(self):
        super().__init__()

class FastLowLevelRecurrent(Module):
    def __init__(self):
        super().__init__()

class Output(Module):
    def __init__(self):
        super().__init__()
