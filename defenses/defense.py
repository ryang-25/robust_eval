from abc import ABC, abstractmethod

from utils import DDP

from torch import device
from torch.nn import Module
from torch.optim import SGD
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
from torch.utils.data import DataLoader
from typing import Dict

import torch

class Defense(ABC):
    def __init__(self, model: Module, device: device, dataset: str,
                 checkpoint_path: str,
                 normalize: callable):
        """
        dataset: the name of the dataset.
        checkpoint_path: THe path where checkpoints are saved.
        """
        self.model = model
        self.device = device
        self.checkpoint_path = checkpoint_path
        self.normalize = normalize
        match dataset:
            case "CIFAR-10" | "CIFAR-100":
                self.optimizer = SGD(model.parameters(), lr=0.01, momentum=0.9,
                                     weight_decay=5e-4, nesterov=True)
                self.scheduler = CosineAnnealingLR(self.optimizer, T_max=200, eta_min=1e-5)
            case "ImageNet":
                self.scheduler = StepLR(self.optimizer, 30, 0.1)


    def checkpoint(self):
        """
        Make a checkpoint so we don't lose our work.
        """
        model = self.model.module.state_dict() if DDP else self.model.state_dict()
        state = {
            "model_state": model,
            "optimizer_state": self.optimizer.state_dict(),
            "scheduler_state": self.scheduler.state_dict(), 
        }
        torch.save(state, self.checkpoint_path)


    @abstractmethod
    def generate(self, train_loader: DataLoader, test_loader: DataLoader,
                 start_epoch: int,
                 epochs: int) -> tuple[Dict[str, any], float]:
        raise NotImplementedError