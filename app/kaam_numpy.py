"""KAAM in plain numpy: the canonical model's exact arithmetic, without torch.

``export_demo.py`` checks this module against the checkpoint's own test AUC and
the published phi curves, so the demo runs the same model the paper reports.
"""

import collections
import json
import pickle
import zipfile

import numpy as np

# ---------------------------------------------------------------- reading a .pt without torch
_DTYPES = {"FloatStorage": np.float32, "DoubleStorage": np.float64, "LongStorage": np.int64,
           "IntStorage": np.int32, "BoolStorage": np.bool_}


class _StorageType:
    def __init__(self, name):
        self.name = name


def _rebuild_tensor(storage, offset, size, stride, requires_grad=False, hooks=None, metadata=None):
    size, stride = tuple(size), tuple(stride)
    if not size:
        return np.array(storage[offset])
    view = np.lib.stride_tricks.as_strided(storage[offset:], shape=size,
                                           strides=[s * storage.itemsize for s in stride])
    return np.array(view)


def load_torch_checkpoint(path):
    """Read a ``torch.save`` zip checkpoint into numpy arrays and plain Python objects.

    Only the handful of classes a state_dict checkpoint needs are allowed through
    the unpickler; anything else raises instead of executing.
    """
    zf = zipfile.ZipFile(path)
    root = zf.namelist()[0].split("/")[0]
    if zf.read(f"{root}/byteorder").decode().strip() != "little":
        raise ValueError("big-endian checkpoint not supported")

    class Unpickler(pickle.Unpickler):
        def find_class(self, module, name):
            if (module, name) in {("torch._utils", "_rebuild_tensor_v2"), ("torch._utils", "_rebuild_tensor")}:
                return _rebuild_tensor
            if module == "torch" and name in _DTYPES:
                return _StorageType(name)
            if (module, name) == ("collections", "OrderedDict"):
                return collections.OrderedDict
            raise pickle.UnpicklingError(f"refusing to load {module}.{name}")

        def persistent_load(self, pid):
            _, storage_type, key, _, _ = pid
            return np.frombuffer(zf.read(f"{root}/data/{key}"), dtype=_DTYPES[storage_type.name]).copy()

    with zf.open(f"{root}/data.pkl") as fh:
        return Unpickler(fh).load()


# ---------------------------------------------------------------- the model
def _silu(x):
    return x / (1.0 + np.exp(-x))


def bspline_basis(x, knots, order):
    """Cox-de Boor, line for line as ``kanvas.bspline.BSplineEdge._basis``."""
    t = knots
    xu = x[:, None]
    bases = ((xu >= t[:-1]) & (xu < t[1:])).astype(x.dtype)
    bases[x == t[-1], -1] = 1.0
    with np.errstate(divide="ignore"):
        for p in range(1, order + 1):
            left_den = t[p:-1] - t[: -(p + 1)]
            right_den = t[p + 1:] - t[1:-p]
            left = np.where(left_den == 0, 0.0, 1.0 / left_den).astype(x.dtype)
            right = np.where(right_den == 0, 0.0, 1.0 / right_den).astype(x.dtype)
            bases = (xu - t[: -(p + 1)]) * left * bases[:, :-1] + (t[p + 1:] - xu) * right * bases[:, 1:]
    return bases


class KAAMNumpy:
    """risk = sigmoid(bias + sum_i phi_i(x_i)), inputs in raw trace units."""

    def __init__(self, params, dtype=np.float64):
        self.dtype = dtype
        self.names = list(params["feature_names"])
        self.ranges = {n: tuple(params["ranges"][n]) for n in self.names}
        self.order = int(params["spline_order"])
        self.bias = dtype(params["bias"])
        self.edges = {n: {"knots": np.asarray(e["knots"], dtype), "w": np.asarray(e["spline_weight"], dtype),
                          "base": dtype(e["base_weight"])} for n, e in params["edges"].items()}
        self.phi_mean = {n: float(v) for n, v in params.get("phi_mean", {}).items()}

    def scale(self, name, raw):
        lo, hi = self.ranges[name]
        return ((np.asarray(raw, np.float64) - lo) / (hi - lo)).astype(self.dtype)

    def phi(self, name, raw):
        """phi_i at raw-unit values (1D array) -> log-odds, uncentred, exactly as explain() returns it."""
        e = self.edges[name]
        x = np.clip(np.atleast_1d(self.scale(name, raw)), e["knots"][0], e["knots"][-1])
        return bspline_basis(x, e["knots"], self.order) @ e["w"] + e["base"] * _silu(x)

    def logit(self, X):
        """X: (n, 10) raw units in self.names order."""
        X = np.asarray(X, np.float64)
        return sum(self.phi(n, X[:, i]) for i, n in enumerate(self.names)) + self.bias

    def predict(self, X):
        return 1.0 / (1.0 + np.exp(-self.logit(X)))

    @property
    def baseline_logit(self):
        """The average training task's logit: bias plus each phi's training mean."""
        return float(self.bias) + sum(self.phi_mean.values())

    def explain(self, row):
        """One task -> {feature: contribution vs the average training task} (exact SHAP for an additive model)."""
        return {n: float(self.phi(n, [row[i]])[0]) - self.phi_mean[n] for i, n in enumerate(self.names)}


def load_demo(path):
    with open(path) as fh:
        bundle = json.load(fh)
    return KAAMNumpy(bundle["model"]), bundle
