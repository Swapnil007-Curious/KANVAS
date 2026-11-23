"""Per-split preprocessing: clipping and min-max scaling fitted on the training split only.

v1 clipped every feature at the 1st and 99th percentiles of *all* rows inside
``data.build_kanvas_dataset``, before any split existed, so the clip bounds
carried a little information from the validation and test tasks into training.
v2 builds the dataset unclipped and fits both steps here, on the training split
of each protocol, then applies the fitted constants unchanged to the validation
and test splits:

1. clip each feature to the training split's ``[q_lower, q_upper]`` quantiles;
2. min-max scale each clipped feature with the training split's (min, max), so
   training values land in [0, 1] and other splits may fall slightly outside.

Every model in v2 (KAAM, logistic regression, MLP, LightGBM, EBM) receives the
same transformed inputs. Tree models are invariant to monotone rescaling; the
clipping still applies to them, which keeps the inputs identical across models.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .data import FEATURE_NAMES

CLIP_QUANTILES = (0.01, 0.99)


@dataclass(frozen=True)
class Preprocessor:
    """Constants fitted on one training split."""

    features: tuple
    clip_bounds: dict  # feature -> (lower, upper) training quantiles, raw units
    ranges: dict  # feature -> (min, max) of the clipped training values, raw units

    def clip(self, X: pd.DataFrame) -> pd.DataFrame:
        """Clip ``X`` (raw units) to the training bounds; columns must match ``features`` in order."""
        _check(X, self.features)
        out = X.copy()
        for c in self.features:
            lo, hi = self.clip_bounds[c]
            out[c] = out[c].clip(lo, hi)
        return out

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Clip, then min-max scale with the training constants."""
        clipped = self.clip(X)
        lo = pd.Series({c: self.ranges[c][0] for c in self.features})
        hi = pd.Series({c: self.ranges[c][1] for c in self.features})
        return (clipped - lo) / (hi - lo)

    def to_raw(self, feature: str, scaled):
        """Map scaled values of one feature back to raw units (for plotting curves)."""
        lo, hi = self.ranges[feature]
        return lo + np.asarray(scaled, dtype=float) * (hi - lo)

    def to_scaled(self, feature: str, raw):
        lo, hi = self.ranges[feature]
        return (np.asarray(raw, dtype=float) - lo) / (hi - lo)


def fit_preprocessor(X_train: pd.DataFrame, quantiles=CLIP_QUANTILES, features=None) -> Preprocessor:
    """Fit clip bounds and scaling ranges on the training split alone.

    A feature that is constant after clipping (possible in a small or one-user
    training split) gets the range ``(lo, lo + 1)`` so scaling stays defined;
    its scaled value is then 0 for every training task.
    """
    features = tuple(FEATURE_NAMES if features is None else features)
    _check(X_train, features)
    q_lo, q_hi = quantiles
    bounds, ranges = {}, {}
    for c in features:
        lo, hi = (float(v) for v in X_train[c].quantile([q_lo, q_hi]))
        bounds[c] = (lo, hi)
        clipped = X_train[c].clip(lo, hi)
        mn, mx = float(clipped.min()), float(clipped.max())
        ranges[c] = (mn, mx if mx > mn else mn + 1.0)
    return Preprocessor(features, bounds, ranges)


def _check(X, features):
    if tuple(X.columns) != tuple(features):
        raise ValueError(f"columns {list(X.columns)} must equal {list(features)} in order")
