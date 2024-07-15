# evaluate.py
#
# Attack evaluation.

from attacks import *
from evaluation import *
from utils import create_model, load_test_set, normalize, write_evaluation

from argparse import ArgumentParser, Namespace
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from typing import Optional

import sys
import torch

def create_attack(method: Optional[str], model, device, normal) -> Attack:
    if method is None:
        method = "Attack"
    try:
        return eval(method)(model, device, normal)
    except:
        sys.exit("Invalid attack!")


def create_evaluation(method: str, model, model_aug) -> Evaluation:
    try:
        evaluation = eval(method)
        if model_aug is not None:
            return evaluation(model, model_aug)
        return evaluation(model)
    except TypeError:
        sys.exit("A comparative evaluation was chosen, but no weights-aug was passed!")
    except:
        sys.exit("Invalid evaluation!")


@torch.no_grad
def evaluate(attack: Attack, method: Evaluation, test_loader: DataLoader, device, normalize):
    final_metric = {}
    for images, targets in test_loader:
        images, targets = images.to(device), targets.to(device)
        adv_xs = attack.generate(images, targets)  # adverarial samples
        adv_xs = normalize(adv_xs) # normalize here!
        images = normalize(images)
        # could be slightly wrong, drop the last sample if you care
        # or set the batch_size to a divisor of the dataset
        metrics = method.evaluate(images, targets, adv_xs, method.model(adv_xs))
        for k in metrics:
            # update in place
            final_metric[k] = (
                final_metric[k] + metrics[k] if k in final_metric else metrics[k]
            )
            print(f"{k}: {metrics[k]}")
    for k in final_metric:
        final_metric[k] /= len(test_loader)  # Average metrics across batch #
    return final_metric

def main(args: Namespace):
    cuda_available = torch.cuda.is_available()
    should_compile = args.compile and (sys.platform.startswith("linux") or not cuda_available)
    device = torch.device(f"cuda:{args.id}" if cuda_available else "cpu")
    if cuda_available:
        torch.backends.cudnn.benchmark = args.no_benchmark
        torch.set_float32_matmul_precision("high")
    model = create_model(args.model).to(device) # preemptive move to GPU
    if should_compile:
        model = torch.compile(model)
        print("Model compile finished.")

    load = torch.load(args.weights, map_location=device)
    load = load.get("model_state", load) # if we're using a resume checkpoint
    model.load_state_dict(load)

    model_aug = None
    if args.weights_aug is not None:
        model_aug = create_model(args.model).to(device)
        if should_compile:
            model_aug = torch.compile(model_aug)
        load = torch.load(args.weights_aug, map_location=device)
        load = load.get("model_state", load) # if we're using a resume checkpoint
        model_aug.load_state_dict(load)

    normal = normalize(args.dataset)
    attack_model = model if args.black_box or model_aug is None else model_aug
    attack = create_attack(args.attack, attack_model, device, normal)
    method = create_evaluation(args.evaluation, model, model_aug)
    test_set = load_test_set(args.dataset)
    test_loader = DataLoader(
        test_set,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=cuda_available,
        persistent_workers=True if args.num_workers > 0 else False,
    )
    metrics = evaluate(attack, method, test_loader, device, normal)
    write_evaluation(args.model, args.weights, args.attack, metrics)


if __name__ == "__main__":
    parser = ArgumentParser("robust_eval evaluation framework")
    parser.add_argument("--model", help="The model to train on.")
    parser.add_argument("--weights", required=True, help="Model weights to load.")
    parser.add_argument(
        "--weights-aug",
        help="Augmented (defensive) model weights. Required if you're considering a comparative attack.",
    )
    parser.add_argument(
        "--dataset",
        choices=["CIFAR-10", "CIFAR-100", "CIFAR-10-C"],
        required=True,
        help="The dataset to test on.",
    )
    parser.add_argument("--attack", help="The type of attack to perform.")
    parser.add_argument(
        "--black-box",
        help="Whether to perform a black box attack. In a black box attack, attacks are generated against the original "
        "model rather than the defensive one.",
        action="store_true",
    )
    parser.add_argument("--evaluation", default="CleanAccuracy", help="The evaluation method to use.")
    parser.add_argument(
        "--batch-size", type=int, default=128, help="The batch size of the data."
    )
    parser.add_argument(
        "--num-workers", type=int, default=4, help="Number of pre-fetching threads."
    )
    parser.add_argument("--no-benchmark", action="store_false", help="Disable cuDNN autotuner")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--id", type=int, default=0)
    main(parser.parse_args())

