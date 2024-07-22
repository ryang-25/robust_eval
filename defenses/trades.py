# trades.py
#
# An implementation of TRADES.

from defenses import Clean

import time
import torch
import torch.nn.functional as F

class TRADES(Clean):
    def __init__(self, model, device, dataset: str, checkpoint_path: str, normalize):
        super().__init__(model, device, dataset, checkpoint_path, normalize)
        self.steps = 10
        self.step_size = 0.003
        self.epsilon = 0.031
        self.beta = 6

    def train(self, train_loader):
        model = self.model
        loss = torch.zeros(1, device=self.device)
        for images, labels in train_loader:
            self.optimizer.zero_grad()
            model.eval()
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)
            images_norm = self.normalize(images)
            images_adv = images + 0.001 * torch.randn_like(images) # leaf
            for _ in range(self.steps):
                images_adv = images_adv.detach().requires_grad_()
                with torch.enable_grad():
                    output, output_adv = model(images_norm), model(self.normalize(images_adv))
                    loss_kl = F.kl_div(F.log_softmax(output_adv, dim=1),
                                    F.softmax(output, dim=1), reduction="sum")
                loss_kl.backward()
                images_adv = images_adv + self.step_size * images_adv.grad.sign() # pyright: ignore
                images_adv = images_adv.clamp(images-self.epsilon, images+self.epsilon).clamp(0., 1.)        

            model.train()
            self.optimizer.zero_grad()
            output = model(images_norm)
            nat_loss = F.cross_entropy(output, labels)
            robust_loss = F.kl_div(F.log_softmax(model(self.normalize(images_adv)), dim=1),
                                                       F.softmax(output, dim=1),
                                                       reduction="batchmean")
            batch_loss = nat_loss + self.beta * robust_loss
            batch_loss.backward()
            self.optimizer.step()
            loss = loss.detach() + batch_loss
        self.scheduler.step()
        return loss.item() / len(train_loader)


    def generate(self, train_loader, test_loader, start_epoch, epochs):
        test_acc = 0.
        for epoch in range(start_epoch, epochs):
            begin_time = time.time()
            if self.is_ddp:
                train_loader.sampler.set_epoch(epoch)
            train_loss = self.train(train_loader)
            if self.is_main:
                test_loss, test_acc = self.test(test_loader)
                self.checkpoint()
                print(f"Epoch {epoch} took {time.time() - begin_time:3f}s. training "\
                        f"loss: {train_loss:>4f}, test loss: {test_loss:>4f}, test accuracy: {test_acc}")
        return self.state_dict(), test_acc
