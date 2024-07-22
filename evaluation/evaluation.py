from abc import ABC, abstractmethod

from torch import Tensor
from torch.nn import Module

import torch
import torch.nn.functional as F

def nan_or_zero(tensor: Tensor) -> float:
    return tensor.item() if not tensor.isnan() else 0

class Evaluation(ABC):
    def __init__(self, model: Module):
        self.model = model

    @abstractmethod
    def evaluate(
        self, nat_xs: Tensor, nat_ys: Tensor, adv_xs: Tensor, adv_out: Tensor, /
    ) -> dict[str, float]:
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

class CleanAccuracy(Evaluation):
    """
    Clean accuracy. The model's ability to successfully classify the test set.
    """
    @torch.inference_mode()
    def evaluate(self, nat_xs: Tensor, nat_ys: Tensor, *_):
        self.model.eval()
        pred = self.model(nat_xs).argmax(1)
        return {"Clean accuracy": (pred == nat_ys).sum().item() / len(nat_xs)}


class AdvAccuracy(Evaluation):
    def evaluate(self, nat_xs, nat_ys, adv_xs, adv_out):
        self.model.eval()
        pred = adv_out.argmax(1)
        return {
            "Adversarial accuracy": (pred == nat_ys).sum().item() / len(adv_out)
        }


class ACAC(Evaluation):
    def evaluate(self, nat_xs, nat_ys, adv_xs, adv_out):
        prob, pred = adv_out.softmax(1).max(1)
        cac = prob[pred != nat_ys]
        return {
            "Average Confidence of Adversarial Class": cac.mean().item(),
        }


class ACTC(Evaluation):
    def evaluate(self, nat_xs, nat_ys, adv_xs, adv_out):
        prob = adv_out.softmax(1)
        pred = adv_out.argmax(1)
        tc = prob[:, nat_ys][pred != nat_ys]
        return {
            "Average Confidence of True Class": nan_or_zero(tc),
        }


class NTE(Evaluation):
    def evaluate(self, nat_xs, nat_ys, adv_xs, adv_out):
        prob, pred = adv_out.softmax(1).topk(2, sorted=True)
        diff = prob.diff()[pred[:,0] != nat_ys]
        return {"Noise tolerance estimation": nan_or_zero(diff.mean())}


class CCV(ComparativeEvaluation):
    def evaluate(self, nat_xs, nat_ys, adv_xs, adv_out):
        prob_nat, pred_nat = self.model(nat_xs).softmax(1).max(1)
        prob_aug, pred_aug = self.model_aug(nat_xs).softmax(1).max(1)
        diff = (prob_aug - prob_nat).abs_()[(nat_ys == pred_nat) & (nat_ys == pred_aug)]
        return {
            "Classification Confidence Variance": nan_or_zero(diff.mean()),
        }

class COS(ComparativeEvaluation):
    def evaluate(self, nat_xs, nat_ys, adv_xs, adv_out):
        prob_nat, pred_nat = self.model(nat_xs).softmax(1).max(1)
        prob_aug, pred_aug = self.model_aug(nat_xs).softmax(1).max(1)
        div = correct = 0
        for i in range(len(nat_ys)):
            if nat_ys[i] == pred_nat[i] == pred_aug[i]:
                correct += 1
                m = ((prob_nat[i]+prob_aug[i])/2).log_()
                div += F.kl_div(m, prob_nat[i], reduction="batchmean") + F.kl_div(m,
                    prob_aug[i], reduction="batchmean")
        div /= 2 * correct
        return {
            "Classification Output Stability": div.item(),
        }
