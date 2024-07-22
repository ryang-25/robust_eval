# robust_eval

A minimal testing framework for adversarial and corruption robustness.
To get started, run

```
python3 ./train.py
```

to train a model.

We currently only support projected gradient descent (PGD) (also called $FGSM^k$) to generate small $\ell_\infty$ perturbations.
