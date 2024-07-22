# ce.py
#
# Implementing https://arxiv.org/pdf/1903.12261 precisely.

from datasets import CIFAR10C, CIFAR100C
from utils import MODELS_DIR
from evaluation import CleanAccuracy
from preactresnet import *
from wide_resnet import *

from torch.utils.data import DataLoader
from torchvision.transforms import v2

from argparse import ArgumentParser

import torch
import os

corruptions = [
    "brightness",
    "contrast",
    "defocus_blur",
    "elastic_transform",
    "fog",
    "frost",
    "gaussian_blur",
    "gaussian_noise",
    "glass_blur",
    "impulse_noise",
    "jpeg_compression",
    "motion_blur",
    "pixelate",
    "saturate",
    "shot_noise",
    "snow",
    "spatter",
    "speckle_noise",
    "zoom_blur"
]

SEVERITY_LEVELS = 5+1

@torch.inference_mode()
def uce_c(model, device, args):
    """
    Turns out we don't actually have the mCE data for CIFAR-10-C.
    """

    cuda_available = torch.cuda.is_available()
    normalize = v2.Normalize([0.5] * 3, [0.5] * 3)
    preprocess = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True), normalize ])
    total_acc = 0
    for corruption in corruptions:
        test_set = CIFAR10C(MODELS_DIR, corruption, preprocess)
        test_loader = DataLoader(
            test_set,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=cuda_available,
            persistent_workers=True if args.num_workers > 0 else False,
        )
        c_acc = 0
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            c_acc += CleanAccuracy(model).evaluate(images, labels).popitem()[1]
        print("Current CE:", c_acc /len(test_loader))
        total_acc += c_acc / len(test_loader)
    return 1 - total_acc / len(corruptions)


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = eval(args.model)()
    model = model.to(device)
    model.load_state_dict(torch.load(args.weights, map_location=device))
    print("uce_c: ", uce_c(model, device, args))


if __name__ == "__main__":
    parser = ArgumentParser("ce.py")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--weights")
    parser.add_argument("--model")
    main(parser.parse_args())
