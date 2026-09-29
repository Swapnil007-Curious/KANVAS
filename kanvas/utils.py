"""Helper utilities for KANVAS: knot-vector construction and reproducibility.

These helpers are deliberately kept free of any cross-feature logic — they
only ever operate on a single scalar feature's range, which is what allows
`BSplineEdge` to enforce the "one function per feature, no interaction"
architectural constraint.
"""

import random

import numpy as np
import torch


def build_clamped_knot_vector(x_min, x_max, num_knots, spline_order, grid_extend=0.1):
    """Build a uniform, clamped B-spline knot vector for one scalar feature.

    A clamped B-spline of order ``k`` (degree ``k``, e.g. k=3 is cubic) needs
    its first and last knots repeated ``k + 1`` times so that the resulting
    curve actually touches its first and last control points at the domain
    boundaries, instead of trailing off as an open uniform spline would.

    The domain is extended slightly beyond the feature's observed range
    ``[x_min, x_max]`` by ``grid_extend`` on each side, so that inputs near
    (but outside) the training range still land inside a well-defined region
    of the spline rather than exactly on its boundary.

    Layout of the returned knot vector::

        [low]*(k+1)  +  num_knots interior knots (uniform)  +  [high]*(k+1)

    where ``low = x_min - grid_extend`` and ``high = x_max + grid_extend``.

    Args:
        x_min: Minimum expected value of the feature.
        x_max: Maximum expected value of the feature.
        num_knots: Number of *interior* knots (strictly between low and high).
        spline_order: Spline degree ``k`` (3 = cubic).
        grid_extend: Extra margin added beyond [x_min, x_max] on each side.

    Returns:
        1D ``torch.FloatTensor`` of length ``num_knots + 2 * (spline_order + 1)``.
        The number of B-spline basis functions (and thus learnable
        coefficients) this knot vector supports is
        ``num_knots + spline_order + 1``.
    """
    if x_max <= x_min:
        raise ValueError(f"x_max ({x_max}) must be greater than x_min ({x_min})")

    low = x_min - grid_extend
    high = x_max + grid_extend
    k = spline_order

    # num_knots interior points, strictly inside (low, high).
    interior = torch.linspace(low, high, num_knots + 2)[1:-1]

    knots = torch.cat(
        [
            torch.full((k + 1,), low, dtype=torch.float32),
            interior.to(torch.float32),
            torch.full((k + 1,), high, dtype=torch.float32),
        ]
    )
    return knots


def set_seed(seed=42):
    """Seed python, numpy, and torch RNGs for reproducible experiments.

    Args:
        seed: Integer seed shared across all three RNG sources.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
