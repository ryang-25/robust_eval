# odi.py
#
# The ODI-PGD attack from https://arxiv.org/abs/2003.06878

from attacks.attack import Attack
from torch.optim import SGD

import torch
import torch.nn.functional as F

    #     opt.zero_grad()
    #     with torch.enable_grad():
    #         if i < ODI_num_steps:
    #             loss = (model(X_pgd) * randVector_).sum()
    #         elif args.lossFunc == 'xent':
    #             loss = nn.CrossEntropyLoss()(model(X_pgd), y)
    #         else:
    #             loss = margin_loss(model(X_pgd),y)
    #     loss.backward()
    #     if i < ODI_num_steps: 
    #         eta = ODI_step_size * X_pgd.grad.data.sign()
    #     else:
    #         eta = step_size * X_pgd.grad.data.sign()
    #     X_pgd = Variable(X_pgd.data + eta, requires_grad=True)
    #     eta = torch.clamp(X_pgd.data - X.data, -epsilon, epsilon)
    #     X_pgd = Variable(X.data + eta, requires_grad=True)
    #     X_pgd = Variable(torch.clamp(X_pgd, 0, 1.0), requires_grad=True)

    # acc_each = (model(X_pgd).data.max(1)[1] == y.data).detach().cpu().numpy() 
    # acc_pgd = (model(X_pgd).data.max(1)[1] == y.data).float().sum()
    # return acc_clean, acc_pgd, acc_each

class ODI(Attack):
    def __init__(self, model, device):
        super().__init__(model, device)
        self.iters = 100
        self.epsilon = 8
        self.step_size = 2.5 * self.epsilon/100


    def generate(self, xs_nat, ys_nat):
        rand = torch.empty(xs_nat.size(), dtype=torch.float32,
                           device=self.device).uniform_(-1., 1.)



        # # our random tensor
        # xs_pgd = xs
        # rand = torch.empty(xs.size(), dtype=torch.float, device=self.device,
        #                    requires_grad=True).uniform_(-1,1)
        # for i in range(self.iters):
        #     opt = SGD(xs_pgd, lr=1e-3)
        #     opt.zero_grad()
        #     with torch.enable_grad():
        #         loss = (self.model(xs_pgd) * rand).sum()
        #     loss.backward()
        #     eta = xs_pgd.grad.data.sign()
        #     xs_pgd = xs + eta
        #     eta = (xs_pgd - xs).clamp(-epsilon, epsilon)
        #     xs_pgd = xs + eta
        #     xs_pgd = xs_pgd.clamp(0, 1)
