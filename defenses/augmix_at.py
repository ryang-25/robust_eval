# Copyright 2019 Google LLC
# Copyright 2024 Roland Yang
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from attacks.pgd import PGD
from defenses.clean import Clean

from torchvision.transforms import v2

import time
import torch
import torch.nn.functional as F

class AugMixAT(Clean):
    def train(self, train_loader):
        """
        We may have to reduce learning rate if gradients explode.
        """
        self.model.train()

        transforms = v2.Compose([
            v2.ToDtype(torch.uint8, scale=True),
            v2.AugMix(interpolation=v2.InterpolationMode.BILINEAR, all_ops=False),
            v2.ToDtype(torch.float32, scale=True),
        ])

        pgd = PGD(self.model, self.device, self.normalize)
        pgd.iters = 7
        pgd.step_size = 2/255

        loss_ema = torch.zeros(1, device=self.device)
        for images, labels in train_loader:
            self.optimizer.zero_grad()
            images, labels = images.to(self.device), labels.to(self.device)
            images = transforms(images)
            images_adv = pgd.generate(images, labels)
            images_adv = self.normalize(images_adv)
            output = self.model(images_adv)
            loss = F.cross_entropy(output, labels)
            loss.backward()
            self.optimizer.step()
            loss_ema = loss_ema * 0.9 + loss.detach() * 0.1
        self.scheduler.step() # we choose to adjust per epoch
        return loss_ema
    
    def generate(self, train_loader, test_loader, start_epoch, epochs):
        best_acc = 0.
        best_weights = None
        for epoch in range(start_epoch, epochs):
            begin_time = time.time()
            if self.is_ddp:
                train_loader.sampler.set_epoch(epoch)
            train_loss = self.train(train_loader)
            if self.is_main:
                test_loss, test_acc = self.test(test_loader)
                if test_acc > best_acc:
                    best_weights = self.state_dict()
                    best_acc = test_acc
                self.checkpoint() # checkpoint
                print(f"Epoch {epoch} took {time.time() - begin_time:.2f}s. training",
                    f"loss: {train_loss:>4f}, test loss: {test_loss:>4f}, test accuracy: {test_acc:>4f}")
        return best_weights, best_acc
