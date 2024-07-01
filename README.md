# robust_eval

A minimal testing framework for adversarial and corruption robustness.
To get started, run

```
python3 ./train.py
```

to train a model.

We currently only support projected gradient descent (PGD) (also called $FGSM^k$) to generate small $\ell_\infty$ perturbations.

## Models

*   PreActResNet
*   WideResNet

## Musings

*   The approach in PyTorch is to normalize the data in the dataloader, but we opt to normalize afterwards and in calls to `model(...)`
*   `torch.DistributedDataParallel` is not usable and also does not work with `torch.compile`
*   We were concerned about https://github.com/ryderling/DEEPSEC/issues/3, but it seems that PyTorch now accepts soft labels (https://github.com/pytorch/pytorch/issues/11959)


## Evaluation Criteria

### Corruption Error (CE)

We're hesitant to call the mean of the corruption error the *mCE* since that is the mean of

$$CE = \frac{\sum E_s^f}{\sum E_s^{Alexnet}}$$

across all corruptions, but since we use CIFAR-10-C, we calculate

$$CE = \sum E_s^f$$

across all corruptions.

## Defense Methods

*   PGD-AT (2 steps, step size of 3e-3, and $\varepsilon = 0.031$)
*   TRADES
*   AugMix (with JSD loss)


## Performance

On Linux:

```
pip install torchtriton --extra-index-url "https://download.pytorch.org/whl/cu121"
```
