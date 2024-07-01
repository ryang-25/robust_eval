# clean.py
#
# No defense, just training.

from defenses.defense import Defense
from utils import DDP

from torch.utils.data import DataLoader

import time
import torch
import torch.nn.functional as F

class Clean(Defense):
    def train(self, train_loader: DataLoader):
        self.model.train()
        device_type = "cuda" #self.device.type
        scaler = torch.GradScaler(device_type)

        loss_ema = 0.
        for images, labels in train_loader:
            self.optimizer.zero_grad()
            images, labels = images.to(self.device), labels.to(self.device)
            images = self.normalize(images)
            # attempt to use some AMP.
            with torch.autocast(device_type, dtype=torch.float16):
                output = self.model(images)
                loss = F.cross_entropy(output, labels)
            scaler.scale(loss).backward()
            scaler.step(self.optimizer)
            scaler.update()
            loss_ema = loss_ema * 0.9 + loss.item() * 0.1
        self.scheduler.step()
        return loss_ema

    @torch.no_grad
    def test(self, test_loader: DataLoader):
        self.model.eval()
        loss = 0.
        correct = 0
        for images, labels in test_loader:
            images, labels = images.to(self.device), labels.to(self.device)
            output = self.model(self.normalize(images))
            loss += F.cross_entropy(output, labels).item()
            pred = output.argmax(1)
            correct += (pred == labels).sum().item()
        total_len = len(test_loader.dataset)
        return loss / total_len, correct / total_len

    def generate(self, train_loader, test_loader, start_epoch, epochs):
        best_acc = 0.
        best_weights = None
        for epoch in range(start_epoch, epochs):
            begin_time = time.time()
            if DDP:
                train_loader.sampler.set_epoch(epoch)
            train_loss = self.train(train_loader)
            if not DDP or self.device == 0:
                test_loss, test_acc = self.test(test_loader)
                if test_acc > best_acc:
                    best_weights = self.model.module.state_dict() if DDP else self.model.state_dict()
                    best_acc = test_acc
                self.checkpoint() # checkpoint
                print(f"Epoch {epoch} took {time.time() - begin_time:.2f}s. training",
                    f"loss: {train_loss:>4f}, test loss: {test_loss:>4f}, test accuracy: {test_acc:>4f}")
        return best_weights, best_acc
