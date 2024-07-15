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

from defenses.clean import Clean

from torchvision.transforms import v2

import time
import torch
import torch.nn.functional as F

class AugMix(Clean):
    def train(self, train_loader):
        """
        We may have to reduce learning rate if gradients explode.
        """
        self.model.train()

        # terribly hacky way to do things but whatever, we want to avoid passing in or constant mean and std :(
        transforms = v2.Compose([
            v2.ToDtype(torch.uint8, scale=True),
            v2.AugMix(interpolation=v2.InterpolationMode.BILINEAR, all_ops=False),
            v2.ToDtype(torch.float32, scale=True),
        ])

        loss_ema = torch.zeros(1, device=self.device)
        for images, labels in train_loader:
            self.optimizer.zero_grad()
            images, labels = images.to(self.device), labels.to(self.device)
            
            # JSD calculation here
            images_aug1, images_aug2 = transforms(images), transforms(images)
            images_all = torch.cat((images, images_aug1, images_aug2), dim=0)
            images_all = self.normalize(images_all)

            output_all = self.model(images_all)
            nat_out = output_all.split(len(images))[0]

            # Calculate cross entropy on natural
            loss = F.cross_entropy(nat_out, labels)

            p_nat, p_aug1, p_aug2 = output_all.softmax(1).split(len(images))

            # trick from https://github.com/google-research/augmix/blob/master/cifar.py#L232
            m = ((p_nat + p_aug1 + p_aug2)/3.).clamp(1e-7, 1).log()
            loss +=  12/3 * (F.kl_div(m, p_nat, reduction="batchmean") +
                F.kl_div(m, p_aug1, reduction="batchmean") +
                F.kl_div(m, p_aug2, reduction="batchmean"))

            loss.backward()
            self.optimizer.step()
            loss_ema = loss_ema * 0.9 + loss.detach() * 0.1
        self.scheduler.step() # we choose to adjust per epoch
        return loss_ema.item()
    
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
