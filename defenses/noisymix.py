from torch.distributions.beta import Beta

import torch

def noise(x, add_level, multi_level, sparse_level):
    add_noise = 0.
    mult_noise = 1.
    dist = Beta(2., 5.)
    if add_level > 0.:
        # var = x.var().sqrt_()
        add_noise = add_level * dist.sample() * torch.randn_like(x)
        sparse = torch.rand_like(x)
        add_noise[sparse<sparse_level] = 0.
    if multi_level > 0.:
        multi_noise = multi_level * dist.sample() * (2*torch.randn_like(x)-1) + 1
        sparse = torch.rand_like(x)
        mult_noise[sparse<sparse_level] = 1.
    return multi_noise * x + add_noise


def do_noisy_mixup(x, y, jsd=0, alpha=0.0, add_noise_level=0.0, mult_noise_level=0.0, sparse_level=0.0):
    lam = np.random.beta(alpha, alpha) if alpha > 0.0 else 1.0
    
    if jsd==0:
        index = torch.randperm(x.size()[0]).cuda()
        x = lam * x + (1 - lam) * x[index]
        x = _noise(x, add_noise_level=add_noise_level, mult_noise_level=mult_noise_level, sparse_level=sparse_level)
    else:
        kk = 0
        q = np.int(x.shape[0]/3)
        index = torch.randperm(q).cuda()
    
        for i in range(1,4):
            x[kk:kk+q] = lam * x[kk:kk+q] + (1 - lam) * x[kk:kk+q][index]
            x[kk:kk+q] = _noise(x[kk:kk+q], add_noise_level=add_noise_level*i, mult_noise_level=mult_noise_level, sparse_level=sparse_level)
            kk += q
     
    y_a, y_b = y, y[index]
    
    return x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)