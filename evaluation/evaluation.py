from abc import ABC, abstractmethod
from typing import Any

from torch import Tensor
from torch.nn import Module


class Evaluation(ABC):
    def __init__(self, model: Module):
        self.model = model

    @abstractmethod
    def evaluate(
        self, nat_xs: Tensor, nat_ys: Tensor, adv_xs: Tensor, adv_out: Tensor
    ) -> dict[str, Any]:
        """
        nat_xs: clean inputs
        nat_ys: clean labels
        adv_xs: adversarial inputs
        adv_ys: adversarial logits (after model)
        """
        raise NotImplementedError


class ComparativeEvaluation(Evaluation):
    def __init__(self, model, model_aug):
        """
        A comparative evaluation requires two models.
        """
        super().__init__(model)
        self.model_aug = model_aug
