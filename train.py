# train.py
#
# Adversarial training.

from defenses import *
from utils import *

from argparse import ArgumentParser, Namespace
from torch.distributed import barrier, destroy_process_group
from torch.nn import Module
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision.transforms import v2

import os
import sys
import torch
import torch.multiprocessing as mp
import torchvision.datasets as DS


def load_train_set(set: str) -> DS.ImageFolder | DS.VisionDataset:
    """
    Function to load the dataset with the preprocessing 
    """
    cifar_transforms = transform=v2.Compose([v2.RandomHorizontalFlip(),
                                             v2.RandomCrop(32, padding=4),
                                             v2.ToImage(),
                                             v2.ToDtype(torch.float32, scale=True)])
    match set:
        case "CIFAR-10":
            return DS.CIFAR10(MODELS_DIR, train=True, download=True, transform=cifar_transforms)
        case "CIFAR-100":
            return DS.CIFAR100(MODELS_DIR, train=True, download=True, transform=cifar_transforms)
        case "ImageNet":
            transform = v2.Compose([])
            return DS.ImageNet(MODELS_DIR, split="train", transform=transform)
        case "TinyImageNet":
            raise NotImplementedError
        case _:
            raise NotImplementedError


def create_defense(method: str, model: Module, device: torch.device,
                   dataset: str, checkpoint_path, normalize) -> Defense:
    """
    creates a defense for the model
    """
    try:
        return eval(method)(model, device, dataset, checkpoint_path, normalize)
    except Exception as e:
        sys.exit(f"Could not find defense {method}!\n\nDiagnostics:\n{e}")


def main(device: torch.device, args: Namespace, world_size=1):
    is_cuda = device.type == "cuda"
    if is_cuda:
        if args.ddp:
            setup(device, world_size)
            args.batch_size *= world_size # assume homogenous GPUs
        torch.backends.cudnn.benchmark = args.no_benchmark
        torch.set_float32_matmul_precision("high") # Use TensorFloat32 computation
        print("Using TensorFloat32 computation...")
    model = create_model(args.model).to(device) # preemptive move to GPU
    if args.ddp:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = DDP(model, device_ids=[device])
    # https://github.com/pytorch/pytorch/issues/125093
    # Only Linux supports GPU compile with Triton.
    if args.compile and sys.platform.startswith("linux"):
        model.compile(mode="reduce-overhead")
        print("Model compile started.")
    if not os.path.exists(MODELS_DIR):
        os.makedirs(MODELS_DIR)
    train_set = load_train_set(args.dataset)
    train_loader = DataLoader(train_set, batch_size=args.batch_size,
                              num_workers=args.num_workers,
                              pin_memory=is_cuda,
                              persistent_workers=True if args.num_workers > 0 else False,
                              sampler=DistributedSampler(train_set) if args.ddp else None)
    test_set = load_test_set(args.dataset)
    test_loader = DataLoader(test_set, batch_size=args.batch_size,
                              num_workers=args.num_workers,
                              pin_memory=is_cuda,
                              persistent_workers=True if args.num_workers > 0 else False)

    normal = normalize(args.dataset)
    defense = create_defense(args.method, model, device, args.dataset, args.checkpoint_path, normal)
    is_main = not args.ddp or device.index == 0

    # Restore weights if interrupted
    start_epoch = 0
    if args.resume:
        if is_main:
            load = torch.load(args.checkpoint_path, map_location=device)
            defense.model.load_state_dict(load["model_state"])
            defense.optimizer.load_state_dict(load["optimizer_state"])
            defense.scheduler.load_state_dict(load["scheduler_state"])
            start_epoch = defense.scheduler.last_epoch
        if args.ddp:
            barrier()

    weights, accuracy = defense.generate(train_loader, test_loader, start_epoch, args.epochs)
    if is_main:
        if not os.path.exists(args.output):
            os.makedirs(args.output)
        path = os.path.join(args.output, f"{args.dataset}-{args.model}-{args.method}-{args.epochs}.pt")
        torch.save(weights, path)
        print("Weights have been saved! Best accuracy:", accuracy)
    if args.ddp:
        barrier()
        destroy_process_group()


if __name__ == "__main__":
    parser = ArgumentParser("robust_eval evaluation framework")
    parser.add_argument("--batch-size", type=int, default=128,
                           help="The batch size of the data.")
    parser.add_argument("--dataset",
                           choices=["CIFAR-10", "CIFAR-100", "ImageNet", "TinyImageNet"],
                           required=True,
                           help="The dataset to train on.")
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--model", default="WideResNet168")
    parser.add_argument("--output", default=WEIGHTS_DIR,
                           help="the root directory of the model's state_dict that you want to save to.")
    parser.add_argument("--method", default="Clean", help="Defense method you want to train.")
    parser.add_argument("--num-workers", type=int, default=4,
                        help='Number of pre-fetching threads.')
    parser.add_argument("--resume", action="store_true", help="Resume training from the checkpoint")
    parser.add_argument("--checkpoint-path", required=True, help="Path for checkpoints")
    parser.add_argument("--no-benchmark", action="store_false", help="Disable cuDNN autotuner")
    parser.add_argument("--ddp", action="store_true", help="Enables training with DistributedDataParallel.")
    parser.add_argument("--compile", action="store_true", help="Compile the model.")
    parser.add_argument("--id", type=int, default=0, help="Choose which CUDA device to train on. A no-op if DDP is enabled.")
    args = parser.parse_args()
    if args.ddp:
        world_size = torch.cuda.device_count()
        assert world_size > 0
        wrapper = lambda rank : main(torch.device("cuda", rank), args, world_size) # this could potentially be a problem?
        mp.spawn(wrapper, nprocs=world_size)
    else:
        if torch.cuda.is_available():
            device = torch.device("cuda", args.id)
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
        main(device, args)

