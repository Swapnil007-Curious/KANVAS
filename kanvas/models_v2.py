"""v2 model fitting: KAAM, MLP, logistic regression, LightGBM and EBM on one split.

Every ``fit_*`` function takes scaled float32 arrays (training and validation),
a seed and one hyperparameter setting, and returns a ``Fitted`` with a
``predict`` function (scaled features -> failure probability) and training
facts (epochs, early stopping, seconds). Selection over the validation grid and
the five seeds is done by ``scripts/run_experiments.py``.

KAAM training. ``phi_i(x) = w_i SiLU(x) + sum_j c_ij B_j(x)`` with fixed knots,
so each feature's basis values B_j(x_i) and SiLU(x_i) depend only on the input.
``kaam_design`` evaluates them once per split with each edge's own ``_basis``
(the same Cox-de Boor code the model runs in ``forward``), and training fits
the 151 parameters on that design with minibatch Adam. The fitted values are
then written into the edges of a standard ``KAAM``; ``tests/test_v2_models.py``
checks that ``KAAM.forward`` reproduces the training-time logit.
"""

from __future__ import annotations

import copy
import os
import time
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn.functional as F

from .baselines import MLP, count_parameters
from .data import FEATURE_NAMES
from .metrics import roc_auc
from .model import KAAM
from .utils import set_seed

UNIT_RANGES = {name: (0.0, 1.0) for name in FEATURE_NAMES}
NUM_KNOTS, SPLINE_ORDER = 10, 3
BATCH_SIZE = 2048
# Early stopping monitors validation ROC-AUC (PROTOCOL.md, deviation 1): an epoch counts as an improvement
# when it raises the best validation AUC by more than MIN_DELTA; training stops after PATIENCE epochs
# without one and the best epoch is restored. Validation log loss was used first and was replaced before
# any test access: under the time-ordered split the validation failure rate (68%) is three times the
# training rate (21%), so an untrained network scoring about 0.5 had the lowest validation log loss and
# training stopped at epoch 1.
MAX_EPOCHS = 500
PATIENCE = 10
MIN_DELTA = 1e-4
KAAM_LR = 0.01
N_THREADS = int(os.environ.get("KANVAS_THREADS", "2"))

KAAM_LAMBDAS = [0.0, 1e-4, 1e-3, 1e-2]
MLP_GRID = [{"hidden": (8, 6), "lr": 0.003}, {"hidden": (8, 6), "lr": 0.01},
            {"hidden": (12,), "lr": 0.003}, {"hidden": (12,), "lr": 0.01}]
LR_C_GRID = [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0]
LGBM_GRID = [{"num_leaves": 31, "min_child_samples": 20}, {"num_leaves": 31, "min_child_samples": 200},
             {"num_leaves": 127, "min_child_samples": 20}, {"num_leaves": 127, "min_child_samples": 200}]


@dataclass
class Fitted:
    predict: object
    info: dict = field(default_factory=dict)
    model: object = None


# ----------------------------------------------------------------------------- KAAM
def kaam_design(model, x, chunk=65536):
    """Basis values (n, n_features, n_basis) and SiLU values (n, n_features) for scaled inputs ``x``."""
    x = torch.as_tensor(np.asarray(x, dtype=np.float32))
    names = model.feature_names
    edges = [model.edges[name] for name in names]
    n_basis = edges[0].n_basis
    basis = torch.empty((len(x), len(names), n_basis), dtype=torch.float32)
    silu = torch.empty((len(x), len(names)), dtype=torch.float32)
    with torch.no_grad():
        for start in range(0, len(x), chunk):
            xb = x[start:start + chunk]
            for i, edge in enumerate(edges):
                xc = torch.clamp(xb[:, i], min=edge.knots[0].item(), max=edge.knots[-1].item())
                basis[start:start + chunk, i] = edge._basis(xc)
                silu[start:start + chunk, i] = F.silu(xc)
    return basis, silu


def _second_difference_penalty(coef):
    d2 = coef[:, 2:] - 2 * coef[:, 1:-1] + coef[:, :-2]
    return (d2 ** 2).sum()


def _train_loop(params, logit_fn, n_train, y_train, n_val, y_val, penalty_fn, lr, seed):
    """Minibatch Adam with early stopping on validation ROC-AUC; restores the best epoch.

    The validation log loss is recorded each epoch as well, for the training-curve figure.
    """
    gen = torch.Generator().manual_seed(seed)
    opt = torch.optim.Adam(params, lr=lr)
    yt = torch.as_tensor(np.asarray(y_train, dtype=np.float32))
    yv = torch.as_tensor(np.asarray(y_val, dtype=np.float32))
    yv_np = np.asarray(y_val)
    best, best_epoch, best_state, since = -np.inf, 0, [p.detach().clone() for p in params], 0
    history, loss_history = [], []
    for epoch in range(1, MAX_EPOCHS + 1):
        perm = torch.randperm(n_train, generator=gen)
        for start in range(0, n_train, BATCH_SIZE):
            idx = perm[start:start + BATCH_SIZE]
            opt.zero_grad()
            loss = F.binary_cross_entropy_with_logits(logit_fn("train", idx), yt[idx])
            if penalty_fn is not None:
                loss = loss + penalty_fn()
            loss.backward()
            opt.step()
        with torch.no_grad():
            val_logit = logit_fn("val", None)
            val_loss = F.binary_cross_entropy_with_logits(val_logit, yv).item()
        val_auc = roc_auc(yv_np, val_logit.numpy())
        history.append(val_auc)
        loss_history.append(val_loss)
        if val_auc > best + MIN_DELTA:
            best, best_epoch, since = val_auc, epoch, 0
            best_state = [p.detach().clone() for p in params]
        else:
            since += 1
        if since >= PATIENCE:
            break
    with torch.no_grad():
        for p, s in zip(params, best_state):
            p.copy_(s)
    return {"best_epoch": best_epoch, "epochs_run": epoch, "early_stopped": since >= PATIENCE,
            "best_val_auc": float(best), "val_auc_history": history, "val_loss_history": loss_history}


def fit_kaam(design_train, y_train, design_val, y_val, seed, lam):
    """Fit KAAM on precomputed designs; returns a Fitted whose ``model`` is a standard KAAM."""
    set_seed(seed)
    model = KAAM(FEATURE_NAMES, UNIT_RANGES, num_knots=NUM_KNOTS, spline_order=SPLINE_ORDER)
    coef = torch.nn.Parameter(torch.stack([model.edges[n].spline_weight.detach() for n in FEATURE_NAMES]).clone())
    base = torch.nn.Parameter(torch.stack([model.edges[n].base_weight.detach() for n in FEATURE_NAMES]).clone())
    bias = torch.nn.Parameter(model.bias.detach().clone())
    designs = {"train": design_train, "val": design_val}

    def logit_fn(part, idx):
        basis, silu = designs[part]
        if idx is not None:
            basis, silu = basis[idx], silu[idx]
        return torch.einsum("nfk,fk->n", basis, coef) + (silu * base).sum(1) + bias

    penalty = (lambda: lam * _second_difference_penalty(coef)) if lam > 0 else None
    start = time.perf_counter()
    info = _train_loop([coef, base, bias], logit_fn, len(design_train[0]), y_train, len(design_val[0]), y_val,
                       penalty, KAAM_LR, seed)
    info["seconds"] = time.perf_counter() - start
    with torch.no_grad():
        for i, n in enumerate(FEATURE_NAMES):
            model.edges[n].spline_weight.copy_(coef[i])
            model.edges[n].base_weight.copy_(base[i])
        model.bias.copy_(bias)
    info["roughness"] = float(_second_difference_penalty(coef.detach()).item())
    info["parameters"] = count_parameters(model)

    def predict(design):
        basis, silu = design
        with torch.no_grad():
            return torch.sigmoid(torch.einsum("nfk,fk->n", basis, coef) + (silu * base).sum(1) + bias).numpy()

    return Fitted(predict, info, model)


# ----------------------------------------------------------------------------- MLP
def fit_mlp(x_train, y_train, x_val, y_val, seed, hidden, lr):
    set_seed(seed)
    model = MLP(x_train.shape[1], hidden)
    xs = {"train": torch.as_tensor(x_train), "val": torch.as_tensor(x_val)}

    def logit_fn(part, idx):
        x = xs[part] if idx is None else xs[part][idx]
        return model.net(x).squeeze(-1)

    start = time.perf_counter()
    info = _train_loop(list(model.parameters()), logit_fn, len(x_train), y_train, len(x_val), y_val, None, lr, seed)
    info["seconds"] = time.perf_counter() - start
    info["parameters"] = count_parameters(model)

    def predict(x):
        with torch.no_grad():
            return model(torch.as_tensor(np.asarray(x, dtype=np.float32))).numpy()

    return Fitted(predict, info, model)


# ----------------------------------------------------------------------------- logistic regression
def fit_lr(x_train, y_train, c):
    from sklearn.linear_model import LogisticRegression

    start = time.perf_counter()
    model = LogisticRegression(C=c, max_iter=10_000).fit(x_train, y_train)
    info = {"seconds": time.perf_counter() - start, "parameters": int(model.coef_.size + model.intercept_.size),
            "n_iter": int(model.n_iter_[0])}
    return Fitted(lambda x: model.predict_proba(x)[:, 1], info, model)


# ----------------------------------------------------------------------------- LightGBM
def fit_lgbm(x_train, y_train, x_val, y_val, seed, num_leaves, min_child_samples):
    import lightgbm as lgb

    # early stopping on validation AUC, as for the networks (PROTOCOL.md, deviation 1)
    params = {"objective": "binary", "metric": "auc", "learning_rate": 0.1, "num_leaves": num_leaves,
              "min_child_samples": min_child_samples, "bagging_fraction": 0.8, "bagging_freq": 1,
              "feature_fraction": 0.8, "seed": seed, "verbose": -1, "num_threads": N_THREADS}
    start = time.perf_counter()
    dtrain = lgb.Dataset(x_train, label=y_train, free_raw_data=True)
    dval = lgb.Dataset(x_val, label=y_val, reference=dtrain)
    booster = lgb.train(params, dtrain, num_boost_round=2000, valid_sets=[dval],
                        callbacks=[lgb.early_stopping(50, verbose=False)])
    info = {"seconds": time.perf_counter() - start, "best_iteration": int(booster.best_iteration),
            "early_stopped": bool(booster.current_iteration() < 2000)}
    return Fitted(lambda x: booster.predict(x, num_iteration=booster.best_iteration), info, booster)


# ----------------------------------------------------------------------------- EBM
def fit_ebm(x_train, y_train, seed, interactions):
    from interpret.glassbox import ExplainableBoostingClassifier

    kwargs = {"random_state": seed, "n_jobs": N_THREADS, "feature_names": list(FEATURE_NAMES)}
    if interactions is not None:
        kwargs["interactions"] = interactions
    start = time.perf_counter()
    model = ExplainableBoostingClassifier(**kwargs).fit(x_train, y_train)
    info = {"seconds": time.perf_counter() - start, "terms": len(model.term_names_),
            "interaction_terms": [t for t in model.term_names_ if " & " in t]}
    return Fitted(lambda x: model.predict_proba(x)[:, 1], info, model)


def kaam_state(model):
    """A compact, picklable copy of a KAAM's parameters."""
    return {k: v.detach().clone() for k, v in copy.deepcopy(model.state_dict()).items()}


def load_kaam(state):
    model = KAAM(FEATURE_NAMES, UNIT_RANGES, num_knots=NUM_KNOTS, spline_order=SPLINE_ORDER)
    model.load_state_dict(state)
    model.eval()
    return model
