# utils.py
#
# Associated utilities

from wide_resnet import *
from preactresnet import *
from torch.nn import Module

import os
import sys
import torch
import torch.distributed as dist
import torchvision.datasets as DS
import torchvision.transforms.v2 as v2

MODELS_DIR = "./models/"
WEIGHTS_DIR = "./models/weights/"

TRAIN_RESULTS = "TRAIN_RESULTS"
EVAL_RESULTS = "EVAL_RESULTS"

def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def load_test_set(set: str) -> DS.ImageFolder | DS.VisionDataset:
    """
    Loads the test dataset
    """
    match set:
        case "CIFAR-10":
            return DS.CIFAR10(MODELS_DIR, train=False, download=True,
                              transform=v2.Compose([v2.ToImage(),
                                                    v2.ToDtype(torch.float32, scale=True)]))
        case "CIFAR-100":
            return DS.CIFAR100(MODELS_DIR, train=False, download=True,
                              transform=v2.Compose([v2.ToImage(),
                                                    v2.ToDtype(torch.float32, scale=True)]))


def create_model(model: str) -> Module:
    try:
        return eval(model)()
    except Exception as e:
        sys.exit(f"Could not find model {model}!\n\nDiagnostics:\n{e}")

def normalize(set: str) -> callable:
    match set:
        case "CIFAR-10" | "CIFAR-100" | "CIFAR-10-C" | "CIFAR-10-P":
            return v2.Normalize([0.5] * 3, [0.5] * 3)


def write_train_results(method, accuracy, epochs, path):
    out = f"{method} was trained for {epochs} epochs and achieved a best accuracy of {accuracy}.\n" + f"The weights have been saved to {path}.\n\n"
    print(out)
    with open(TRAIN_RESULTS, "a+") as file:
        file.write(out)

def write_evaluation(model, weights, method, metrics):
    metrics = "\n".join((f"{k}: {v}" for k, v in metrics.items()))
    out = f"{model} with weights {weights} was evaluated using {method}.\n" + f"Results:\n{metrics}\n"
    print(out)
    with open(EVAL_RESULTS, "a+") as file:
        file.write(out)
