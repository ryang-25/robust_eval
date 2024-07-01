# train.py
#
# Adversarial training.

from defenses import *
from utils import *

from argparse import ArgumentParser, Namespace
from torch.nn import Module
from torch.utils.data import DataLoader
from torchvision.transforms import v2
from typing import Optional

import os
import sys
import torch
import torchvision.datasets as DS

if DDP:
    from torch.distributed import barrier, destroy_process_group
    from torch.nn.parallel import DistributedDataParallel as DDPar
    from torch.utils.data.distributed import DistributedSampler

    import torch.multiprocessing as mp


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


def create_defense(method: Optional[str], model: Module, device: torch.device,
                   dataset: str, checkpoint_path, normalize) -> Defense:
    """
    creates a defense for the model
    """
    try:
        return eval(method)(model, device, dataset, checkpoint_path, normalize)
    except Exception as e:
        sys.exit(f"Could not find defense {method}!\n\nDiagnostics:\n{e}")


# def main(device, args: Namespace, world_size):
def main(args: Namespace):
    cuda_available = torch.cuda.is_available()
    device = torch.device("cuda" if cuda_available else "cpu")
    if DDP:
        setup(device, world_size)
        args.batch_size *= world_size

    torch.backends.cudnn.benchmark = args.no_benchmark
    torch.set_float32_matmul_precision("high")

    model = create_model(args.model)
    model = model.to(device) #model.to(rank)
    if DDP:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = DDPar(model, device_ids=[device])
    # https://github.com/pytorch/pytorch/issues/125093
    # Currently only Linux supports GPU compile with triton.
    if False:
        if not DDP and (not cuda_available or sys.platform.startswith("linux")):
            model = torch.compile(model, mode="reduce-overhead")

    if not os.path.exists(MODELS_DIR):
        os.makedirs(MODELS_DIR)
    train_set = load_train_set(args.dataset)
    train_loader = DataLoader(train_set, batch_size=args.batch_size,
                              num_workers=args.num_workers,
                              pin_memory=cuda_available,
                              persistent_workers=True if args.num_workers > 0 else False,
                              # sampler=DistributedSampler(train_set)
                              )
    test_set = load_test_set(args.dataset)
    test_loader = DataLoader(test_set, batch_size=args.batch_size,
                              num_workers=args.num_workers,
                              pin_memory=cuda_available,
                              persistent_workers=True if args.num_workers > 0 else False)
    normal = normalize(args.dataset)
    defense = create_defense(args.method, model, device, args.dataset, args.checkpoint_path, normal)
    # Restore weights if interrupted
    start_epoch = 0
    if args.resume:
        if not DDP or device == 0:
            load = torch.load(args.checkpoint_path, map_location=device)
            defense.model.load_state_dict(load["model_state"])
            defense.optimizer.load_state_dict(load["optimizer_state"])
            defense.scheduler.load_state_dict(load["scheduler_state"])
            start_epoch = defense.scheduler.last_epoch
        if DDP:
            barrier()

    weights, accuracy = defense.generate(train_loader, test_loader, start_epoch, args.epochs)

    if not DDP or device == 0:
        print("Best accuracy:", accuracy)

        # save weights to path
        if not os.path.exists(args.output):
            os.makedirs(args.output)
        path = os.path.join(args.output, f"{args.dataset}-{args.model}-{args.method}.pt")
        torch.save(weights, path)
        write_train_results(args.method, accuracy, args.epochs, path)
    if DDP:
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
    main(parser.parse_args())
    if DDP:
        world_size = torch.cuda.device_count()
        mp.spawn(main, args=(parser.parse_args(), world_size), nprocs=world_size)

