"""BSplineEdge: one learnable univariate function for one scalar feature.

This module is the core interpretability engine of KANVAS. Every feature in
a KAAM model gets exactly one ``BSplineEdge`` instance, and that instance is
never shown any other feature's value. The function it represents,
``phi(x)``, is a smooth, arbitrary-shape curve learned entirely from data,
which is what lets a clinician later ask "what does this model believe about
creatinine, holding everything else aside?" and get a real answer.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import build_clamped_knot_vector


class BSplineEdge(nn.Module):
    """A single learnable edge function phi(x) = base_weight * silu(x) + B(x).

    This is the "base activation + spline residual" parameterization from
    the original KAN paper (Liu et al., 2024): the base term (a scaled SiLU)
    gives the function a sensible, well-behaved starting shape and easy
    gradient flow, while the B-spline term ``B(x)`` — a linear combination of
    B-spline basis functions evaluated via the Cox-de Boor recursion — lets
    the function bend into an arbitrary smooth shape as training progresses.

    Cox-de Boor recursion, briefly:
        Degree-0 basis functions are indicator functions of the knot span
        ``[t_i, t_{i+1})``. Higher-degree basis functions are built
        recursively as a convex combination of two lower-degree neighbors:

            N_{i,0}(x) = 1 if t_i <= x < t_{i+1} else 0

            N_{i,p}(x) = (x - t_i) / (t_{i+p} - t_i)         * N_{i,p-1}(x)
                       + (t_{i+p+1} - x) / (t_{i+p+1} - t_{i+1}) * N_{i+1,p-1}(x)

        B(x) is then ``sum_i coeff_i * N_{i,k}(x)`` for spline order ``k``.
        Because the knot vector is clamped (its first and last knots are
        repeated ``k + 1`` times), some of these denominators are exactly
        zero at the boundary; those terms are defined to contribute zero,
        which is the standard convention and is handled explicitly below.

    The whole computation is vectorized over the batch dimension with plain
    tensor ops (no Python loop over batch elements), so it scales to large
    datasets.
    """

    def __init__(self, num_knots=10, spline_order=3, x_min=0.0, x_max=1.0, grid_extend=0.1):
        """Build the edge function's fixed knot grid and learnable parameters.

        Args:
            num_knots: Number of interior knots used to place the spline grid.
            spline_order: Spline degree ``k`` (3 = cubic, the KAN default).
            x_min: Expected minimum value of this feature, used to place the grid.
            x_max: Expected maximum value of this feature, used to place the grid.
            grid_extend: Extra margin beyond [x_min, x_max] so that mild
                out-of-range inputs still fall inside a well-defined region
                of the grid instead of sitting exactly on its edge.
        """
        super().__init__()
        self.num_knots = num_knots
        self.spline_order = spline_order
        self.x_min = float(x_min)
        self.x_max = float(x_max)
        self.grid_extend = float(grid_extend)

        knots = build_clamped_knot_vector(x_min, x_max, num_knots, spline_order, grid_extend)
        self.register_buffer("knots", knots)

        # Number of B-spline basis functions for a clamped knot vector of
        # this length and order: len(knots) - spline_order - 1.
        self.n_basis = knots.shape[0] - spline_order - 1

        # Small random init keeps the learned function close to flat/linear
        # at the start of training, so early gradients stay well-behaved.
        self.spline_weight = nn.Parameter(torch.randn(self.n_basis) * 0.1)
        self.base_weight = nn.Parameter(torch.randn(()) * 0.1)

    def _basis(self, x):
        """Evaluate all B-spline basis functions at ``x`` via Cox-de Boor.

        Args:
            x: Tensor of shape (batch,), already clamped into the grid's
                [low, high] range.

        Returns:
            Tensor of shape (batch, n_basis) with basis function values.
        """
        t = self.knots  # (m,)
        xu = x.unsqueeze(-1)  # (batch, 1)

        # Degree-0 basis: indicator of the half-open knot span [t_i, t_{i+1}).
        bases = ((xu >= t[:-1]) & (xu < t[1:])).to(x.dtype)  # (batch, m-1)

        # x exactly equal to the top of the grid falls outside every
        # half-open span above, so patch it into the final span explicitly.
        at_high = x == t[-1]
        if at_high.any():
            bases = bases.clone()
            bases[at_high, -1] = 1.0

        for p in range(1, self.spline_order + 1):
            left_num = xu - t[: -(p + 1)]
            left_den = t[p:-1] - t[: -(p + 1)]
            right_num = t[p + 1 :] - xu
            right_den = t[p + 1 :] - t[1:-p]

            # Repeated (clamped) knots make some denominators exactly zero;
            # by convention those terms contribute zero to the recursion.
            left_coeff = torch.where(left_den == 0, torch.zeros_like(left_den), 1.0 / left_den)
            right_coeff = torch.where(right_den == 0, torch.zeros_like(right_den), 1.0 / right_den)

            bases = left_num * left_coeff.unsqueeze(0) * bases[:, :-1] + right_num * right_coeff.unsqueeze(
                0
            ) * bases[:, 1:]

        return bases

    def forward(self, x):
        """Evaluate phi(x) = base_weight * silu(x) + B(x) on a batch.

        Args:
            x: Tensor of shape (batch_size,) — this edge's single feature,
                for every sample in the batch.

        Returns:
            Tensor of shape (batch_size,) with this feature's contribution
            to the model's pre-sigmoid logit, for each sample.
        """
        x_clamped = torch.clamp(x, min=self.knots[0].item(), max=self.knots[-1].item())
        basis = self._basis(x_clamped)  # (batch, n_basis)
        spline_out = basis @ self.spline_weight  # (batch,)
        base_out = self.base_weight * F.silu(x_clamped)  # (batch,)
        return spline_out + base_out

    def get_curve(self, num_points=200):
        """Evaluate phi(x) across this feature's full range, for plotting.

        Args:
            num_points: Number of evenly spaced points to sample between
                x_min and x_max.

        Returns:
            Tuple ``(x_values, y_values)`` of 1D numpy arrays, each of
            length ``num_points``.
        """
        x = torch.linspace(self.x_min, self.x_max, num_points)
        with torch.no_grad():
            y = self.forward(x)
        return x.numpy(), y.numpy()
