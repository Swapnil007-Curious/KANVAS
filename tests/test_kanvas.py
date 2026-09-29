"""Unit tests for the KANVAS architecture: BSplineEdge and KAAM."""

import torch

from kanvas.bspline import BSplineEdge
from kanvas.model import KAAM


def test_bspline_edge_output_shape_matches_batch():
    """BSplineEdge output shape must match the input batch shape."""
    edge = BSplineEdge(num_knots=8, spline_order=3, x_min=-2.0, x_max=2.0)
    for batch_size in (1, 5, 64):
        x = torch.linspace(-2.0, 2.0, batch_size)
        y = edge(x)
        assert y.shape == (batch_size,)


def test_bspline_edge_handles_out_of_range_without_nan():
    """Out-of-range inputs must be handled gracefully via clamping, never NaN/inf."""
    edge = BSplineEdge(num_knots=8, spline_order=3, x_min=0.0, x_max=10.0)
    x = torch.tensor([-1000.0, -50.0, -0.1, 0.0, 10.0, 10.1, 50.0, 1000.0])
    y = edge(x)
    assert torch.isfinite(y).all()


def test_kaam_forward_output_between_zero_and_one():
    """KAAM forward pass output must always be a valid probability in (0, 1)."""
    feature_names = ["age", "creatinine", "glucose"]
    feature_ranges = {"age": (0, 100), "creatinine": (0.1, 15), "glucose": (40, 500)}
    model = KAAM(feature_names, feature_ranges)

    x = torch.tensor(
        [
            [30.0, 1.0, 90.0],
            [-500.0, 999.0, -999.0],  # deliberately out of range
        ]
    )
    y = model(x)
    assert y.shape == (2,)
    assert torch.all((y > 0) & (y < 1))


def test_kaam_explain_returns_one_entry_per_feature():
    """explain() must return exactly one contribution per feature, no more, no less."""
    feature_names = ["age", "creatinine", "glucose"]
    feature_ranges = {"age": (0, 100), "creatinine": (0.1, 15), "glucose": (40, 500)}
    model = KAAM(feature_names, feature_ranges)

    x = torch.tensor([45.0, 2.0, 110.0])
    contributions = model.explain(x)

    assert isinstance(contributions, dict)
    assert set(contributions.keys()) == set(feature_names)
    assert len(contributions) == len(feature_names)


def test_kaam_features_are_decoupled():
    """Changing feature_1's value must not change feature_2's contribution.

    This is the test that verifies the architectural constraint — no feature
    interaction — is actually enforced by the code, not just assumed.
    """
    feature_names = ["feature_1", "feature_2", "feature_3"]
    feature_ranges = {"feature_1": (-3, 3), "feature_2": (0, 10), "feature_3": (-5, 5)}
    model = KAAM(feature_names, feature_ranges)

    x_a = torch.tensor([-2.5, 5.0, 1.0])
    x_b = torch.tensor([2.9, 5.0, 1.0])  # only feature_1 changes

    contrib_a = model.explain(x_a)
    contrib_b = model.explain(x_b)

    assert contrib_a["feature_1"] != contrib_b["feature_1"]
    assert contrib_a["feature_2"] == contrib_b["feature_2"]
    assert contrib_a["feature_3"] == contrib_b["feature_3"]
