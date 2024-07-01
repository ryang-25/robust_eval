# apgd.py
#
# Croce and Hein. https://arxiv.org/abs/2003.01690

from attacks.attack import Attack

import torch
import torch.nn.functional as F
import torchvision.transforms.v2.functional as T

class APGD(Attack):
    def __init__(self, model, device):
        self.lower = -1.
        self.upper = 1.

    def dlr_loss(self, x, y):
        x_sorted, ind_sorted = x.sort(dim=1)
        ind = (ind_sorted[:, -1] == y).float()
        u = torch.arange(x.size(0))

        return -(x[u, y] - x_sorted[:, -2] * ind - x_sorted[:, -1] * (
            1. - ind)) / (x_sorted[:, -1] - x_sorted[:, -3] + 1e-12)



    def generate(self, xs_nat, ys):
        self.model.eval()
        xs_nat = xs_nat.detach()
        # scale to -1, 1
        t = 2 * torch.rand_like(xs_nat) - 1
        # concise normalization
        t /= t.abs().flatten(1).amax(1).view(-1, *([1] * (t.dim() - 1)))
        t *= self.epsilon # micro-op
        xs_adv = xs_nat + t.expand(-1, *xs_nat.size()[1:])
        xs_adv.clamp_(self.lower, self.upper)
        xs_adv.requires_grad_()

        grad = torch.zeros_like(xs)
        for _ in range(self.eot_iters):
            with torch.enable_grad():
                output = self.model(xs_adv)
                loss = self.dlr_loss(output, ys)
            grad += torch.autograd.grad(loss, xs_adv)[0]
        grad /= float(self.eot_iters)


        for i in range(self.iters):
            







        eta = torch.empty_like(xs_nat).uniform_(-self.epsilon, self.epsilon)




    def __init__(self, model, device):
        super().__init__(model, device)
        # hyperparameters from the paper
        self.iters = 20
        self.epsilon = 8/255
        self.step_size = 2/255
        # Took a while to realize but normalize does not in fact put out values in [0,1)
        self.lower = -1.
        self.upper = 1.

    def generate(self, xs_nat: torch.Tensor, ys):
        self.model.eval()
        xs_nat = xs_nat.detach()
        # Slight modification: the eta rather than the input.
        eta = torch.empty_like(xs_nat).uniform_(-self.epsilon, self.epsilon)
        xs = xs_nat + eta

        for _ in range(self.iters):
            xs.detach_()
            xs.requires_grad_()
            with torch.enable_grad():
                output = self.model(xs)
                loss = F.cross_entropy(output, ys)
            grad = torch.autograd.grad(loss, xs)[0].detach_()
            # in-place operations for better efficiency.
            xs = xs.detach_() + self.step_size * grad.sign_()
            xs.clamp_(xs_nat - self.epsilon, xs_nat + self.epsilon)
            xs.clamp_(self.lower, self.upper)
        return xs
