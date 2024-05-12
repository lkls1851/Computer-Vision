import torch
import torch.nn as nn
import numpy
from d2l import torch as d2l


net=nn.Sequential(nn.LazyLinear(8), nn.ReLU(), nn.LazyLinear(1))
print(net[2].state_dict())
print("Datatype of bias: ", type(net[2].bias))
print("Data value of bias: ", net[2].weight.data)


def initialise(module):
    if type(module)==nn.LazyLinear:
        nn.init.normal_(module.weight)
        nn.init.zeros_(module.bias)


net.apply(initialise)
