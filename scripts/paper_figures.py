"""Draw every data figure of the paper as a vector PDF from the result files.

    python scripts/paper_figures.py

Writes paper/figures/*.pdf (the pipeline diagram is TikZ, paper/figures/pipeline.tex). Colours are the
Okabe-Ito set, checked for colour-vision deficiency with the dataviz validator; every model also has its
own marker, so no figure relies on colour alone. The result root can be changed with KANVAS_RESULTS.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
RES = Path(os.environ.get("KANVAS_RESULTS", ROOT / "results"))
OUT = ROOT / "paper" / "figures"

MODELS = ["kaam", "lr", "mlp", "lgbm", "ebm", "ebm_int"]
NAME = {"kaam": "KAAM", "lr": "Logistic reg.", "mlp": "MLP", "lgbm": "LightGBM", "ebm": "EBM, additive",
        "ebm_int": "EBM, pairwise"}
COLOR = {"kaam": "#0072B2", "mlp": "#E69F00", "ebm": "#009E73", "lgbm": "#D55E00", "ebm_int": "#CC79A7",
         "lr": "#56B4E9"}
MARK = {"kaam": "o", "lr": "X", "mlp": "^", "lgbm": "D", "ebm": "s", "ebm_int": "P"}
RAISE, LOWER, NEUTRAL, INK, MUTED, GRID = "#D55E00", "#0072B2", "#8C8C8C", "#222222", "#666666", "#E6E6E6"
NL = chr(10)
FAM = [("random", "all", "Random"), ("time", "all", f"Time-ordered{NL}all users"),
       ("time", "known users", f"Time-ordered{NL}known users"), ("time", "new users", f"Time-ordered{NL}new users"),
       ("grouped", "all", "User-grouped"), ("louo-d1", "all", f"Without{NL}User 1"),
       ("louo-d14", "all", f"Without{NL}Users 1, 4")]
FEATURES = ["plan_cpu", "plan_mem", "plan_gpu", "inst_num", "avg_cpu_usr", "avg_gpu_util", "avg_cpu_kernel",
            "avg_load_1", "cap_gpu", "avg_net_receive"]
UNITS = {"plan_cpu": "100 = 1 core", "plan_mem": "GB", "plan_gpu": "100 = 1 GPU", "inst_num": "instances",
         "avg_cpu_usr": "%", "avg_gpu_util": "%", "avg_cpu_kernel": "%", "avg_load_1": "load",
         "cap_gpu": "GPUs", "avg_net_receive": "rate"}
COL1, COL2 = 3.45, 7.1  # IEEE column and page widths, inches

plt.rcParams.update({
    "font.size": 7.5, "axes.titlesize": 7.5, "axes.labelsize": 7.5, "xtick.labelsize": 6.5, "ytick.labelsize": 6.5,
    "legend.fontsize": 6.5, "font.family": "DejaVu Sans", "pdf.fonttype": 42, "ps.fonttype": 42,
    "axes.edgecolor": "#888888", "axes.linewidth": 0.6, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "xtick.major.width": 0.5, "ytick.major.width": 0.5,
    "xtick.major.size": 2.5, "ytick.major.size": 2.5, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.5,
    "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False, "lines.linewidth": 1.4,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.02, "figure.dpi": 150,
    "pdf.compression": 9,
})


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}.pdf", metadata={"Creator": None, "Producer": None, "CreationDate": None})
    if os.environ.get("KANVAS_FIG_PNG"):  # raster copies for checking the layout by eye
        fig.savefig(Path(os.environ["KANVAS_FIG_PNG"]) / f"{name}.png", dpi=200)
    plt.close(fig)
    print("wrote", name)


def read(path):
    p = RES / path
    return pd.read_csv(p) if p.exists() else None


# ----------------------------------------------------------------------------- data figures
def attrition():
    p = RES / "data" / "attrition_full_1h.csv"
    if not p.exists():
        p = ROOT / "results" / "data" / "attrition_full_1h.csv"
    att = pd.read_csv(p)
    names = ["Task table", "Labelled", "First instance found", "Telemetry in window", "Machine spec found",
             "No missing feature"]
    fig, ax = plt.subplots(figsize=(COL1, 1.55))
    y = np.arange(len(att))[::-1]
    ax.barh(y, att["tasks"] / 1e6, color="#0072B2", height=0.62, edgecolor="white", linewidth=1)
    for yi, (_, r) in zip(y, att.iterrows()):
        rate = "" if pd.isna(r["failure_rate"]) else f"  failure rate {100 * r['failure_rate']:.1f}%"
        ax.text(r["tasks"] / 1e6 + 0.02, yi, f"{int(r['tasks']):,}{rate}", va="center", fontsize=6.3, color=INK)
    ax.set_yticks(y, names[:len(att)])
    ax.set_xlabel("Tasks (millions)")
    ax.set_xlim(0, 2.25)
    ax.grid(axis="y", visible=False)
    ax.tick_params(axis="y", length=0)
    save(fig, "attrition")


def concentration():
    uc = read("interpret/user_concentration.csv")
    if uc is None:
        return
    fig, ax = plt.subplots(figsize=(COL1, 1.9))
    for ds, ls, lab in [("full", "-", "full trace"), ("original", "--", "original sample")]:
        g = uc[uc.dataset == ds]
        ax.plot(g["k"], 100 * g["failure_share"], ls, color=RAISE, label=f"failures, {lab}")
        ax.plot(g["k"], 100 * g["task_share"], ls, color=LOWER, label=f"tasks, {lab}")
    ax.set_xscale("log")
    ax.set_xlabel("Top $k$ users (ranked by failures or by tasks)")
    ax.set_ylabel("Share held (%)")
    ax.set_ylim(0, 100)
    ax.legend(loc="lower right", ncol=1, handlelength=2.2)
    save(fig, "concentration")


# ----------------------------------------------------------------------------- benchmark
def benchmark(datasets=("full", "original"), name="benchmark", width=COL2):
    cis = {ds: read(f"analysis/{ds}/model_ci.csv") for ds in datasets}
    if cis[datasets[0]] is None:
        return
    fig, axes = plt.subplots(1, len(FAM), figsize=(width, 2.05), sharey=True)
    ypos = {m: i for i, m in enumerate(MODELS[::-1])}
    for ax, (proto, sub, title) in zip(axes, FAM):
        for j, ds in enumerate(datasets):
            ci = cis[ds]
            if ci is None:
                continue
            off = 0.0 if len(datasets) == 1 else (0.17 if j == 0 else -0.17)
            for m in MODELS:
                r = ci[(ci.protocol == proto) & (ci.subset == sub) & (ci.model == m)]
                if not len(r):
                    continue
                r = r.iloc[0]
                yy = ypos[m] + off
                ax.plot([r["ci_low"], r["ci_high"]], [yy, yy], color=COLOR[m], lw=1.1, solid_capstyle="round")
                ax.plot(r["auc"], yy, MARK[m], color=COLOR[m], ms=4.2,
                        mfc=COLOR[m] if j == 0 else "white", mew=0.9)
        lo_, hi_ = [], []
        for ds in datasets:
            ci = cis[ds]
            if ci is not None:
                sel = ci[(ci.protocol == proto) & (ci.subset == sub)]
                lo_.append(sel["ci_low"].min())
                hi_.append(sel["ci_high"].max())
        lo, hi = min(lo_), max(hi_)
        pad = 0.08 * (hi - lo) + 0.005
        ax.set_xlim(lo - pad, hi + pad)
        if lo - pad < 0.5 < hi + pad:
            ax.axvline(0.5, color=NEUTRAL, lw=0.7, ls=(0, (3, 2)))
        ax.set_title(title, fontsize=6.3)
        ax.grid(axis="y", visible=False)
        ax.tick_params(axis="x", labelsize=5.6)
        ax.locator_params(axis="x", nbins=3)
    axes[0].set_yticks(list(ypos.values()), [NAME[m] for m in ypos])
    axes[0].tick_params(axis="y", length=0)
    fig.supxlabel("Test ROC-AUC (95% bootstrap interval)", fontsize=7, y=-0.02)
    if len(datasets) > 1:
        from matplotlib.lines import Line2D
        h = [Line2D([], [], marker="o", color=INK, ls="", ms=4, label="full trace"),
             Line2D([], [], marker="o", color=INK, mfc="white", ls="", ms=4, label="original sample")]
        fig.legend(handles=h, loc="upper center", ncol=2, bbox_to_anchor=(0.55, 1.12))
    fig.subplots_adjust(wspace=0.22)
    save(fig, name)


def reliability(name="reliability", protos=(("random", "Random"), ("time", "Time-ordered"), ("grouped", "User-grouped")),
                width=COL2, ds="full"):
    rel = read(f"analysis/{ds}/reliability.csv")
    if rel is None:
        return
    fig, axes = plt.subplots(1, len(protos), figsize=(width, 2.2 if width > 4 else 1.7), sharey=True)
    for ax, (proto, title) in zip(np.atleast_1d(axes), protos):
        ax.plot([0, 1], [0, 1], color=NEUTRAL, lw=0.7, ls=(0, (3, 2)))
        for m in MODELS:
            g = rel[(rel.protocol == proto) & (rel.model == m) & (rel["count"] >= 100)]
            if not len(g):
                continue
            ax.plot(g["mean_p"], g["rate"], "-", marker=MARK[m], color=COLOR[m], ms=2.8, lw=1.0, label=NAME[m])
        ax.set_title(title)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect("equal")
        ax.set_xlabel("Mean predicted risk")
    np.atleast_1d(axes)[0].set_ylabel("Observed failure rate")
    h, lab = np.atleast_1d(axes)[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="upper center", ncol=6, bbox_to_anchor=(0.5, 1.07))
    save(fig, name)


# ----------------------------------------------------------------------------- interpretability
DISCRETE = ["plan_cpu", "plan_mem", "plan_gpu", "inst_num", "cap_gpu"]
_TRAIN = {}


def train_values(dataset="full"):
    """Raw training-split features of the canonical random split (for the data ticks under the curves)."""
    if dataset not in _TRAIN:
        try:
            sys.path.insert(0, str(ROOT / "scripts"))
            from run_experiments import RUNS, load_dataset
            d = load_dataset(dataset)
            tr = np.load(RUNS / dataset / "random" / "s42" / "split.npz")["train"]
            _TRAIN[dataset] = d["X"].iloc[tr]
        except Exception:  # noqa: BLE001  (the build is local; without it the figure has no ticks)
            _TRAIN[dataset] = None
    return _TRAIN[dataset]


def data_ticks(ax, feature, x_lo, x_hi, dataset="full", min_share=0.005):
    """Ticks at the values of a discrete feature that hold at least min_share of training tasks; height ~ share."""
    X = train_values(dataset)
    if X is None or feature not in DISCRETE:
        return
    v = X[feature]
    v = v[(v >= x_lo) & (v <= x_hi)]
    share = v.value_counts(normalize=True)
    share = share[share >= min_share]
    if not len(share):
        return
    import matplotlib.transforms as mtransforms
    trans = mtransforms.blended_transform_factory(ax.transData, ax.transAxes)
    hmax = share.max()
    ax.vlines(share.index.to_numpy(dtype=float), 0, 0.03 + 0.12 * share.to_numpy() / hmax, transform=trans,
              color=MUTED, lw=0.8, alpha=0.9)
def shapes_kaam_ebm(name="shapes_kaam_ebm", features=FEATURES, ncols=5, width=COL2):
    sh = read("interpret/shapes_kaam_ebm.csv")
    corr = read("interpret/ebm_shape_corr.csv")
    if sh is None:
        return
    src = RES / "interpret" / "ebm_shape_source.json"
    shape_ds = json.loads(src.read_text(encoding="utf-8"))["dataset"] if src.exists() else "full"
    nrows = int(np.ceil(len(features) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(width, 1.35 * nrows + 0.2))
    for ax, f in zip(np.ravel(axes), features):
        g = sh[sh.feature == f]
        ax.plot(g["x"], g["kaam"], color=COLOR["kaam"], label="KAAM")
        ax.plot(g["x"], g["ebm"], color=COLOR["ebm"], ls=(0, (4, 1.5)), label="EBM, additive")
        r = corr.set_index("feature").loc[f, "pearson_r"]
        rd = read("interpret/ebm_shape_corr_data.csv")
        extra = "" if rd is None else f", {rd.set_index('feature').loc[f, 'pearson_r_data']:.2f} data"
        ax.set_title(f"{f}{NL}r {r:.2f} grid{extra}", fontsize=6.0)
        ax.set_xlabel(UNITS[f], fontsize=6, labelpad=1)
        ax.axhline(0, color=NEUTRAL, lw=0.5)
        ax.tick_params(labelsize=5.5)
        data_ticks(ax, f, g["x"].min(), g["x"].max(), dataset=shape_ds)
        ax.locator_params(axis="x", nbins=4 if ncols > 3 else 3)
    for ax in np.ravel(axes)[len(features):]:
        ax.axis("off")
    np.ravel(axes)[0].set_ylabel("log-odds")
    if nrows > 1:
        np.ravel(axes)[ncols].set_ylabel("log-odds")
    h, lab = np.ravel(axes)[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.04))
    fig.tight_layout(h_pad=0.6, w_pad=0.5)
    save(fig, name)


def curves_all():
    cu = read("interpret/curves_canonical.csv")
    if cu is None:
        return
    fig, axes = plt.subplots(2, 5, figsize=(COL2, 3.0))
    for ax, f in zip(np.ravel(axes), FEATURES):
        g = cu[cu.feature == f]
        for mname, gg in g.groupby("model"):
            if mname.startswith("full"):
                ax.plot(gg["x"], gg["phi"], color=COLOR["kaam"], lw=0.9, alpha=0.75,
                        label="full trace, 5 seeds" if mname.endswith("0") else None)
            else:
                ax.plot(gg["x"], gg["phi"], color=COLOR["mlp"], lw=1.1, ls=(0, (4, 1.5)), label="original sample")
        ax.axhline(0, color=NEUTRAL, lw=0.5)
        ax.set_title(f, fontsize=6.5)
        ax.set_xlabel(UNITS[f], fontsize=6, labelpad=1)
        ax.tick_params(labelsize=5.5)
        data_ticks(ax, f, g["x"].min(), g["x"].max(), dataset="full")
    axes[0, 0].set_ylabel(r"$\phi$ (log-odds)")
    axes[1, 0].set_ylabel(r"$\phi$ (log-odds)")
    h, lab = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, lab, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.03))
    fig.tight_layout(h_pad=0.6, w_pad=0.5)
    save(fig, "curves_all")


def plan_cpu_protocols(width=COL1, name="plan_cpu_protocols"):
    pc = read("interpret/plan_cpu_protocols.csv")
    if pc is None:
        return
    fig, ax = plt.subplots(figsize=(width, 1.95))
    style = {"random": (COLOR["kaam"], "-", 1.8, "random split 42"), "time": (COLOR["mlp"], "-", 1.5, "time-ordered"),
             "grouped": (COLOR["ebm"], "-", 0.9, "user-grouped folds"), "louo": (COLOR["ebm_int"], "--", 1.2,
                                                                              "without User 1 / Users 1, 4")}
    seen = set()
    for run, g in pc.groupby("run", sort=False):
        kind = run.split("/")[0]
        c, ls, lw, lab = style[kind]
        if kind == "louo" and run.endswith("d14"):
            ls = ":"
        ax.plot(g["x"] / 100, g["phi"], color=c, ls=ls, lw=lw, label=None if kind in seen else lab)
        seen.add(kind)
    ax.axhline(0, color=NEUTRAL, lw=0.5)
    X = train_values("full")
    if X is not None:
        import matplotlib.transforms as mtransforms
        v = X["plan_cpu"]
        share = v[(v >= pc["x"].min()) & (v <= pc["x"].max())].value_counts(normalize=True)
        share = share[share >= 0.005]
        trans = mtransforms.blended_transform_factory(ax.transData, ax.transAxes)
        ax.vlines(share.index.to_numpy(dtype=float) / 100, 0, 0.03 + 0.12 * share.to_numpy() / share.max(),
                  transform=trans, color=MUTED, lw=0.8)
    ax.set_xlabel("Requested CPU (cores)")
    ax.set_ylabel(r"$\phi_{\mathrm{plan\_cpu}}$ (log-odds)")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, handlelength=2.0, columnspacing=1.0)
    save(fig, name)


def case_waterfall(name="case_waterfall", cases=("True positive", "False positive", "False negative", "True negative"),
                   width=COL2):
    """Waterfall from the mean training task's logit to the model's logit, one bar per feature."""
    p = RES / "interpret" / "cases.json"
    if not p.exists():
        return
    cj = json.loads(p.read_text(encoding="utf-8"))
    base = cj["baseline_logit"]
    sel = [c for c in cj["cases"] if c["case"] in cases]
    sel.sort(key=lambda c: cases.index(c["case"]))
    ncols = 2 if len(sel) > 1 else 1
    nrows = int(np.ceil(len(sel) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(width, 2.05 * nrows), squeeze=False)
    for ax, c in zip(np.ravel(axes), sel):
        items = sorted(c["contribution"].items(), key=lambda kv: -abs(kv[1]))
        n = len(items)
        cum = base
        ax.plot([base, base], [n + 0.6, n - 0.4], color=INK, lw=1.4)
        for i, (f, v) in enumerate(items):
            y = n - 1 - i
            ax.barh(y, v, left=cum, color=RAISE if v > 0 else LOWER, height=0.62, edgecolor="white", lw=0.6)
            if i < n - 1:
                ax.plot([cum + v, cum + v], [y - 0.31, y - 0.69], color=MUTED, lw=0.5)
            ax.text(max(cum, cum + v) + 0.04, y, f"{v:+.2f}", va="center", ha="left", fontsize=5.5, color=INK)
            cum += v
        ax.axvline(0, color=NEUTRAL, lw=0.7, ls=(0, (3, 2)))
        ax.plot(cum, -0.9, "v", color=INK, ms=4)
        ax.text(cum, -1.45, f"logit {cum:.3f}" if abs(cum) < 0.05 else f"logit {cum:.2f}", ha="center", va="top",
                fontsize=5.6, color=INK)
        path = np.cumsum([base] + [v for _, v in items])
        lo, hi = min(path.min(), 0.0), max(path.max(), 0.0)
        ax.set_xlim(lo - 0.08 * (hi - lo) - 0.1, hi + 0.22 * (hi - lo) + 0.25)
        labels = [f"{f} = {c['raw'][f]:,.4g} (p{c['percentile'][f]:.0f})" for f, _ in items]
        ax.set_yticks([n] + list(range(n - 1, -1, -1)), [f"mean training task ({base:.2f})"] + labels)
        ax.set_ylim(-2.6, n + 0.8)
        ax.grid(axis="y", visible=False)
        ax.tick_params(axis="y", length=0, labelsize=5.6)
        ax.set_title(f"{c['case']}: risk {c['risk']:.3f}, outcome {'failed' if c['label'] else 'finished'}",
                     fontsize=6.8)
        ax.set_xlabel("failure log-odds (dashed: risk 0.5)", fontsize=6)
    for ax in np.ravel(axes)[len(sel):]:
        ax.axis("off")
    fig.tight_layout(w_pad=1.0, h_pad=0.8)
    save(fig, name)


def grouped_folds(ds="full", name="grouped_folds", width=COL1):
    fa = read(f"analysis/{ds}/fold_auc.csv")
    ch = read(f"analysis/{ds}/chance_tests.csv")
    if fa is None:
        return
    fig, ax = plt.subplots(figsize=(width, 2.0))
    folds = sorted(fa["fold"].unique())
    present = [m for m in MODELS if (fa.model == m).any()]
    for i, m in enumerate(present):
        g = fa[fa.model == m].set_index("fold").loc[folds]
        xs = i + np.linspace(-0.26, 0.26, len(folds))
        for x, (_, r) in zip(xs, g.iterrows()):
            ax.plot([x, x], [r["ci_low"], r["ci_high"]], color=COLOR[m], lw=0.7, alpha=0.8)
        ax.plot(xs, g["auc"], MARK[m], color=COLOR[m], ms=3.2, mfc="white", mew=0.8, ls="")
        if ch is not None and m in set(ch["model"]):
            r = ch.set_index("model").loc[m]
            ax.plot([i - 0.33, i + 0.33], [r["auc"], r["auc"]], color=COLOR[m], lw=2.0, solid_capstyle="butt")
    ax.axhline(0.5, color=NEUTRAL, lw=0.7, ls=(0, (3, 2)))
    ax.set_xticks(range(len(present)), [NAME[m].replace(", ", "\n") for m in present], fontsize=5.8)
    ax.set_ylabel("Test ROC-AUC")
    ax.grid(axis="x", visible=False)
    ax.tick_params(axis="x", length=0)
    save(fig, name)


def lookback():
    lb = read("lookback/lookback.csv")
    cu = read("lookback/curves.csv")
    if lb is None:
        return
    fig, (a, b) = plt.subplots(1, 2, figsize=(COL2 * 0.72, 1.85), gridspec_kw={"width_ratios": [1, 1.25]})
    o = lb[lb.width_h != 1]
    a.axhline(0, color=NEUTRAL, lw=0.7, ls=(0, (3, 2)))
    a.errorbar(o["width_h"], o["diff"], yerr=[o["diff"] - o["ci_low"], o["ci_high"] - o["diff"]], fmt="o",
               color=COLOR["kaam"], ms=3.5, lw=1.0, capsize=0)
    a.set_xscale("log")
    a.set_xticks([0.5, 1, 2, 4, 6], ["0.5", "1", "2", "4", "6"])
    a.minorticks_off()
    a.set_xlabel("Lookback window (h)")
    a.set_ylabel(r"$\Delta$AUC vs. 1 h (shared tasks)")
    a.set_title("(a)", loc="left")
    if cu is not None:
        g = cu[cu.feature == "avg_cpu_kernel"]
        shades = ["#9ecae1", "#6baed6", "#3182bd", "#08519c", "#08306b"]
        for (h, gg), c in zip(g.groupby("width_h"), shades):
            b.plot(gg["x"], gg["phi"], color=c, lw=1.1, label=f"{h:g} h")
        b.axhline(0, color=NEUTRAL, lw=0.5)
        b.set_xlabel("avg_cpu_kernel (%)")
        b.set_ylabel(r"$\phi$ (log-odds)")
        b.legend(ncol=1, loc="upper left", fontsize=5.8)
        b.set_title("(b)", loc="left")
    fig.tight_layout(w_pad=1.2)
    save(fig, "lookback")


def main():
    attrition()
    concentration()
    benchmark()
    benchmark(datasets=("full",), name="benchmark_full", width=COL2)
    reliability()
    shapes_kaam_ebm()
    # three request features and three telemetry features with the largest KAAM swing (results/interpret/ebm_shape_corr.csv)
    sw = read("interpret/ebm_shape_corr.csv")
    req = ["plan_cpu", "plan_mem", "plan_gpu", "inst_num", "cap_gpu"]
    top = (sw[sw.feature.isin(req)].nlargest(3, "kaam_swing")["feature"].tolist() +
           sw[~sw.feature.isin(req)].nlargest(3, "kaam_swing")["feature"].tolist()) if sw is not None else FEATURES[:6]
    shapes_kaam_ebm(name="shapes_kaam_ebm_ieee", features=top, ncols=3, width=COL1)
    curves_all()
    plan_cpu_protocols()
    case_waterfall()
    case_waterfall(name="case_waterfall_ieee", cases=("True positive", "False positive"), width=COL2)
    grouped_folds()
    grouped_folds(ds="original", name="grouped_folds_orig")
    lookback()


if __name__ == "__main__":
    main()
