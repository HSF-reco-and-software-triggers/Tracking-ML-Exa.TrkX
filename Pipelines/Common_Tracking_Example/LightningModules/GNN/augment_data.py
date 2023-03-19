from torch.nn.modules import Module
from torch import Tensor
import torch

def transform(input: Tensor):
    phi = input[:, range(1, input.shape[0], 3)]
    phi = phi + torch.rand(1) * 2 - 1
    phi[phi>1] = phi[phi>1] - 2
    input[:, range(1, input.shape[0], 3)] = phi
    return input

class PhiAugmentation(Module):
    def __init__(self) -> None:
        super().__init__()
    
    def forward(self, input):
        if self.training:
            return transform(input)
        else:
            return input