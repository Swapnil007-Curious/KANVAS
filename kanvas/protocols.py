"""The four v2 evaluation protocols (paper/PROTOCOL.md, section 2), as index arrays.

Every function returns positions into the dataset arrays, as a dict with keys
``train``, ``val`` and ``test`` (or a list of such dicts). Nothing here reads a
model or a test score.
"""

from __future__ import annotations

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold, train_test_split

RANDOM_SPLIT_SEEDS = [42, 43, 44, 45, 46]
CANONICAL_SPLIT_SEED = 42
N_FOLDS = 5
GROUP_SEED = 0
FRACTIONS = (0.70, 0.15, 0.15)


def random_split(y, seed, positions=None):
    """Stratified 70/15/15 split, test first, then validation from the rest (the v1 rule).

    With ``positions``, only those rows are split and positions into the full arrays are returned.
    """
    y = np.asarray(y)
    positions = np.arange(len(y)) if positions is None else np.asarray(positions)
    labels = y[positions]
    n = len(positions)
    n_test, n_val = int(np.ceil(FRACTIONS[2] * n)), int(np.ceil(FRACTIONS[1] * n))
    rest, test = train_test_split(positions, test_size=n_test, random_state=seed, stratify=labels)
    train, val = train_test_split(rest, test_size=n_val, random_state=seed, stratify=y[rest])
    return {"train": np.sort(train), "val": np.sort(val), "test": np.sort(test)}


def time_split(start_time):
    """Earliest 70% train, next 15% validation, last 15% test, by start time (stable for ties)."""
    order = np.argsort(np.asarray(start_time, dtype=float), kind="mergesort")
    n = len(order)
    n_train, n_val = int(np.floor(FRACTIONS[0] * n)), int(np.floor(FRACTIONS[1] * n))
    return {"train": np.sort(order[:n_train]), "val": np.sort(order[n_train:n_train + n_val]),
            "test": np.sort(order[n_train + n_val:])}


def known_user_mask(user, split):
    """For each test task, True when its user has at least one training task."""
    user = np.asarray(user)
    return np.isin(user[split["test"]], np.unique(user[split["train"]]))


def grouped_folds(y, groups, n_folds=N_FOLDS, seed=GROUP_SEED):
    """User-grouped folds: outer StratifiedGroupKFold for test, inner one (seed = fold) for validation."""
    y, groups = np.asarray(y), np.asarray(groups)
    outer = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    folds = []
    for k, (rest, test) in enumerate(outer.split(np.zeros(len(y)), y, groups)):
        inner = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=k)
        tr, va = next(inner.split(np.zeros(len(rest)), y[rest], groups[rest]))
        folds.append({"train": np.sort(rest[tr]), "val": np.sort(rest[va]), "test": np.sort(test)})
    return folds


def leave_users_out(y, user, drop_users, seed=CANONICAL_SPLIT_SEED):
    """Remove every task of ``drop_users``, then the random split of protocol (a) with ``seed``."""
    keep = np.flatnonzero(~np.isin(np.asarray(user), list(drop_users)))
    return random_split(y, seed, positions=keep)


def check_split(split, n_total=None, groups=None):
    """Raise if the parts overlap (or, with ``groups``, if a group spans two parts)."""
    parts = [np.asarray(split[k]) for k in ("train", "val", "test")]
    allpos = np.concatenate(parts)
    if len(np.unique(allpos)) != len(allpos):
        raise AssertionError("a row appears in two parts")
    if n_total is not None and len(allpos) != n_total:
        raise AssertionError("not every row is assigned")
    if groups is not None:
        g = [set(np.asarray(groups)[p].tolist()) for p in parts]
        if g[0] & g[1] or g[0] & g[2] or g[1] & g[2]:
            raise AssertionError("a group appears in two parts")
