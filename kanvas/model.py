"""KAAM: Kolmogorov-Arnold Additive Model.

KAAM predicts a binary risk score as

    risk = Sigmoid( sum_i phi_i(x_i) + bias )

where each ``phi_i`` is an independent, learned nonlinear function of a
single feature (a ``BSplineEdge``). No feature's function is ever given
access to another feature's value — this is enforced structurally by
storing each edge in its own slot of an ``nn.ModuleDict`` and slicing out
exactly one input column per edge in the forward pass, not merely assumed
by convention. This strict decoupling is what makes the model's prediction
decomposable, feature by feature, into clinically inspectable contributions.
"""

import numpy as np
import torch
import torch.nn as nn

from .bspline import BSplineEdge


class KAAM(nn.Module):
    """Additive model: risk = Sigmoid(sum of independent per-feature functions + bias)."""

    def __init__(self, feature_names, feature_ranges, num_knots=10, spline_order=3):
        """Build one independent BSplineEdge per feature.

        Args:
            feature_names: List of feature name strings, in the same order
                the model expects columns to appear in its input tensor.
            feature_ranges: Dict mapping each feature name to an
                ``(x_min, x_max)`` tuple describing its expected range.
            num_knots: Number of interior spline knots, shared by every edge.
            spline_order: Spline degree (3 = cubic), shared by every edge.
        """
        super().__init__()
        missing = [name for name in feature_names if name not in feature_ranges]
        if missing:
            raise KeyError(f"feature_ranges is missing entries for: {missing}")

        self.feature_names = list(feature_names)

        # One independent edge per feature, keyed by name. Because forward()
        # below slices exactly one column per edge, no edge can ever see
        # another feature's value.
        self.edges = nn.ModuleDict(
            {
                name: BSplineEdge(
                    num_knots=num_knots,
                    spline_order=spline_order,
                    x_min=feature_ranges[name][0],
                    x_max=feature_ranges[name][1],
                )
                for name in self.feature_names
            }
        )
        self.bias = nn.Parameter(torch.zeros(()))

    def forward(self, x):
        """Compute risk scores for a batch of patients.

        Args:
            x: Tensor of shape (batch_size, num_features), with columns in
                the same order as ``self.feature_names``.

        Returns:
            Tensor of shape (batch_size,) with risk scores in (0, 1).
        """
        contributions = [self.edges[name](x[:, i]) for i, name in enumerate(self.feature_names)]
        logit = torch.stack(contributions, dim=0).sum(dim=0) + self.bias
        return torch.sigmoid(logit)

    def explain(self, x):
        """Decompose one patient's prediction into per-feature contributions.

        Args:
            x: 1D tensor of shape (num_features,) — a single patient's
                feature vector, in the same order as ``self.feature_names``.

        Returns:
            Dict mapping each feature name to its scalar contribution
            ``phi_i(x_i)`` (a python float), *before* the sigmoid and bias
            are applied. Summing these values and adding ``self.bias`` then
            passing through sigmoid reproduces this patient's risk score.
        """
        if x.dim() != 1:
            raise ValueError(f"explain() expects a 1D feature vector, got shape {tuple(x.shape)}")

        contributions = {}
        with torch.no_grad():
            for i, name in enumerate(self.feature_names):
                xi = x[i].reshape(1)
                contributions[name] = self.edges[name](xi).item()
        return contributions

    def plot_all_curves(self, num_points=200):
        """Plot every feature's learned phi_i curve in a grid of subplots.

        Args:
            num_points: Number of points to sample per curve.

        Returns:
            The matplotlib Figure containing the grid of subplots.
        """
        import matplotlib.pyplot as plt

        n = len(self.feature_names)
        ncols = min(3, n)
        nrows = int(np.ceil(n / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows), squeeze=False)
        axes = axes.reshape(-1)

        for ax, name in zip(axes, self.feature_names):
            x_vals, y_vals = self.edges[name].get_curve(num_points)
            ax.plot(x_vals, y_vals)
            ax.axhline(0, color="gray", linewidth=0.5)
            ax.set_title(name)
            ax.set_xlabel(name)
            ax.set_ylabel(f"phi({name})")
            ax.grid(True, alpha=0.3)

        for ax in axes[n:]:
            ax.axis("off")

        fig.tight_layout()
        return fig
