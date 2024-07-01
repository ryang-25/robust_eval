# cw2.py
#
# L2 variant of CW attack from https://arxiv.org/abs/1608.04644

from attacks import Attack

from torch.optim import Adam

import torch

class CW2(Attack):
    def __init__(self, model, device):
        super().__init__(model, device)
        

    def generate(self, xs, ys):
        batch_size = xs.size(0)
        device = self.device
    
        imgs = torch.empty()

        upper = 1
        lower = -1

        box_mul = (upper - lower)/2
        box_plus = (upper + lower)/2

        # Convert to tanh space
        xs.sub_(box_plus).div_(box_mul * 0.999999).tanh_()





        lower_bound = torch.zeros(batch_size, device=device)
        upper_bound = torch.full(batch_size, self.upper_bound, device=device)
        init_consts = torch.full(batch_size, self.init_const, device=device)
                                    
        o_bestl2 = torch.full(batch_size, float("Inf"), device=device)
        o_bestattack = torch.zeros_like(xs)

        modifier = torch.empty_like(xs)

        optimizer = Adam

        for i in range(self.steps):
            print(o_bestl2)
            best_l2 = torch.full(batch_size, float("Inf"), device=device)
            best_score = torch.full(batch_size, -1., device=device)

            for j in range(self.iters):
                
                
                loss = _

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()



