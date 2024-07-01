# attack.py
#
# The main attack class that all attacks implement.

from torch import Tensor
from torch.nn import Module

from abc import ABC

class Attack(ABC):
    def __init__(self, model: Module, device, normalize):
        self.model = model
        self.device = device
        self.normalize = normalize

    def generate(self, xs: Tensor, ys: Tensor) -> Tensor:
        """
        Generate adversarial examples.

        xs: input data
        ys: input labels

        return: adversarial data.
        """
        return xs
    
