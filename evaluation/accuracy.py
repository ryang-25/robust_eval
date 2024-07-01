from evaluation.evaluation import Evaluation

import torch


class CleanAccuracy(Evaluation):
    """
    Clean accuracy. The model's ability to successfully classify the test set.
    """

    @torch.no_grad
    def evaluate(self, nat_xs, nat_ys, *_):
        self.model.eval()
        pred = self.model(nat_xs).argmax(1)
        return {"Clean accuracy": (pred == nat_ys).sum().item() / len(nat_xs)}


class AdvAccuracy(Evaluation):
    @torch.no_grad
    def evaluate(self, _nat_xs, nat_ys, _adv_xs, adv_out):
        self.model.eval()
        pred = adv_out.argmax(1)
        return {
            "Adversarial accuracy": (pred == nat_ys).sum().item() / len(adv_out)
        }
