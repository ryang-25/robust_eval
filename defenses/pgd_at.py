# pgd_at.py
#
# An implementation PGD-AT proposed in Madry et al.

from attacks import PGD
from defenses import Clean
from evaluation import CleanAccuracy

import time
import torch.nn.functional as F

class PgdAt(Clean):
    def train(self, train_loader):
        """
        Training for one epoch.
        """
        self.model.train()
        loss_ema = torch.zeros(1, device=self.device)
        total_acc = 0.
        pgd = PGD(self.model, self.device, self.normalize)
        pgd.iters = 7
        pgd.step_size = 2/255
        n_images = len(train_loader)

        for images, labels in train_loader:
            images, labels = images.to(self.device), labels.to(self.device)
            adv = pgd.generate(images, labels)
            adv = self.normalize(adv)
            
            self.model.train()
            self.optimizer.zero_grad()
            output = self.model(adv)
            loss = F.cross_entropy(output, labels)

            loss.backward()
            self.optimizer.step()
            loss_ema = loss_ema * 0.9 + loss * 0.1
            total_acc += CleanAccuracy(self.model).evaluate(adv, labels).popitem()[1]
        self.scheduler.step()
        total_acc /= n_images
        return total_acc, loss_ema.item()

    def generate(self, train_loader, test_loader, start_epoch, epochs):
        adv_acc = 0.
        for epoch in range(start_epoch, epochs):
            begin_time = time.time()
            if self.is_ddp:
                train_loader.sampler.set_epoch(epoch)
            adv_acc, train_loss = self.train(train_loader)
            if self.is_main:
                test_loss, test_acc = self.test(test_loader)
                self.checkpoint()
                print(f"Epoch {epoch} took {time.time()-begin_time:.2f}s. training "\
                    f"loss: {train_loss:>4f}, test loss: {test_loss:>4f}, adversarial accuracy: {adv_acc:>4f} "\
                    f"test accuracy: {test_acc:>4f}")
        return self.state_dict(), adv_acc
