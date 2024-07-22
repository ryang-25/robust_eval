# evaluate.py
#
# Attack evaluation.

from attacks import *
from evaluation import *
from utils import create_model, load_test_set, normalize, write_evaluation

from argparse import ArgumentParser, Namespace
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
def evaluate(attack: Attack, methods: tuple[Evaluation, ...], test_loader: DataLoader, device, normalize):
    final_metrics = [{} for _ in range(len(methods))]
    for images, labels in test_loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        adv_xs = attack.generate(images, labels) # adversarial examples
        with torch.inference_mode():
            images, adv_xs = normalize(images), normalize(adv_xs) # normalize here!
            # could be slightly wrong, drop the last sample if you care
            # or set the batch_size to a divisor of the dataset
            metrics = tuple(m.evaluate(images, labels, adv_xs, m.model(adv_xs)) for m in methods)
        for i in range(len(metrics)):
            metric = metrics[i]
            final_metric = final_metrics[i]
            for k in metrics[i]:
                # update in place
                final_metric[k] = final_metric[k] + metric[k] if k in final_metric else metric[k]
            print(f"{k}: {metric[k]}") # pyright: ignore
    for i in range(len(final_metrics)):
        for k in final_metrics[i]:
            final_metrics[i][k] /= len(test_loader) # Average metrics across batch #
    return final_metrics

def main(args: Namespace):
    is_cuda = torch.cuda.is_available()
    should_compile = args.compile and sys.platform.startswith("linux")
    if is_cuda:
        device = torch.device(f"cuda:{args.id}")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    if is_cuda:
        torch.backends.cudnn.benchmark = args.no_benchmark
        torch.set_float32_matmul_precision("high")
    model = create_model(args.model).to(device) # preemptive move to GPU
    if should_compile:
        model.compile()
        print("Model compile finished.")

    load = torch.load(args.weights, map_location=device)
    load = load.get("model_state", load) # if we're using a resume checkpoint
    model.load_state_dict(load)

    model_aug = None
    if args.weights_aug is not None:
        model_aug = create_model(args.model).to(device)
        if should_compile:
            model_aug.compile()
        load = torch.load(args.weights_aug, map_location=device)
        load = load.get("model_state", load) # if we're using a resume checkpoint
        model_aug.load_state_dict(load)

    normal = normalize(args.dataset)
    attack_model = model if args.black_box or model_aug is None else model_aug
    attack = create_attack(args.attack, attack_model, device, normal)
    methods = tuple(create_evaluation(e, model, model_aug) for e in args.evaluation)
    test_set = load_test_set(args.dataset)
    test_loader = DataLoader(
        test_set,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=is_cuda,
        persistent_workers=True if args.num_workers > 0 else False,
    )
    metrics = evaluate(attack, methods, test_loader, device, normal)
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
    parser.add_argument("--evaluation", nargs="+", default="CleanAccuracy", help="The evaluation method to use.")
    parser.add_argument(
        "--batch-size", type=int, default=128, help="The batch size of the data."
    )
    parser.add_argument(
        "--num-workers", type=int, default=4, help="Number of pre-fetching threads."
    )
    parser.add_argument("--no-benchmark", action="store_false", help="Disable cuDNN autotuner")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--id", type=int, default=0, help="GPU id to select.")
    main(parser.parse_args())

