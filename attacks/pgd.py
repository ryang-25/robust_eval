# pgd.py
#
# Implements PGD, ODI-PGD
# Madry et al. https://arxiv.org/abs/1706.06083

from attacks.attack import Attack

import torch
import torch.nn.functional as F

class PGD(Attack):
    def __init__(self, model, device, normalize):
        super().__init__(model, device, normalize)
        # hyperparameters from the paper
        self.iters = 20
        self.epsilon = 8/255
        self.step_size = 2/255
        self.odi = False

    def generate(self, xs_nat: torch.Tensor, ys):
        self.model.eval()
        xs_nat = xs_nat.detach()
        if self.odi:
            w = torch.empty_like(xs_nat).uniform_(-1., 1.)
        # Slight modification: the eta rather than the input.
        eta = torch.empty_like(xs_nat).uniform_(-self.epsilon, self.epsilon)
        xs = (xs_nat + eta).clamp(0., 1.)

        for _ in range(self.iters):
            xs = xs.detach().requires_grad_()
            xs.retain_grad()
            with torch.enable_grad():
                output = self.model(self.normalize(xs)) # we need to calculate loss wrt normalized input
                if self.odi:
                    loss = (w*output).sum() # pyright: ignore reportPossiblyUnboundVariable
                else:
                    loss = F.cross_entropy(output, ys)
                output = self.model(self.normalize(xs)) # we need to calculate loss wrt normalized input
                loss = F.cross_entropy(output, ys)
            loss.backward()
            
            xs = xs + self.step_size * xs.grad.sign() # pyright: ignore reportOptionalMemberAccess
            xs = xs.clamp(xs_nat - self.epsilon, xs_nat + self.epsilon)
            xs = xs.clamp(0., 1.)
        return xs
