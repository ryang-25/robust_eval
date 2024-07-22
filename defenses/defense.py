from abc import ABC, abstractmethod

from torch import device
from torch.nn import Module
from torch.optim import SGD
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
from torch.utils.data import DataLoader
from typing import Callable, Dict

import torch

class Defense(ABC):
    def __init__(self, model: Module, device: device, dataset: str,
                 checkpoint_path: str,
                 normalize: Callable):
        """
        dataset: the name of the dataset.
        checkpoint_path: THe path where checkpoints are saved.
        """
        self.model = model
        self.device = device
        self.checkpoint_path = checkpoint_path
        self.normalize = normalize
        model = getattr(model, "_orig_mod", model)
        self.is_ddp = hasattr(model, "module")
        self.is_main = not self.is_ddp or self.device.index == 0
        match dataset:
            case "CIFAR-10" | "CIFAR-100":
                self.optimizer = SGD(model.parameters(), lr=0.01, momentum=0.9,
                                     weight_decay=5e-4, nesterov=True)
                self.scheduler = CosineAnnealingLR(self.optimizer, T_max=200, eta_min=1e-5)
            case "ImageNet":
                self.scheduler = StepLR(self.optimizer, 30, 0.1)

    def state_dict(self):
        model = getattr(self.model, "_orig_mod", self.model) # unwrap the compile
        model = model.module if self.is_ddp else model # unwrap ddp
        return model.state_dict()


    def checkpoint(self):
        """
        Make a checkpoint so we don't lose our work.
        """
        model = self.state_dict()
        state = {
            "model_state": model,
            "optimizer_state": self.optimizer.state_dict(),
            "scheduler_state": self.scheduler.state_dict(), 
        }
        torch.save(state, self.checkpoint_path)


    @abstractmethod
    def generate(self, train_loader: DataLoader, test_loader: DataLoader,
                 start_epoch: int,
                 epochs: int) -> tuple[Dict[str, type], float]:
        raise NotImplementedError