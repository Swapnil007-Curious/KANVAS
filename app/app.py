"""KANVAS live demo: one GPU job's failure risk, split exactly into ten readable contributions.

    cd app
    streamlit run app.py

Runs the canonical KAAM (split 42, seed 0) from kaam_demo.json, which
export_demo.py builds and checks against the checkpoint, the published curves
and the paper. No torch, no identifiers. Motion lives in motion.js.
"""

import base64
import json
import pathlib

import numpy as np
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from kaam_numpy import load_demo

HERE = pathlib.Path(__file__).resolve().parent

# Graphite layers plus one diverging pair. Hue only ever means "raises" or "lowers" failure risk, the same
# red/blue meaning as the paper's figures; every other surface and control is monochrome.
EMBER, GLACIER = "#FF6A55", "#5BA8FF"
INK0, INK1, INK2, INK3 = "#08090C", "#0D0F13", "#14171D", "#1B1F27"  # sidebar, canvas, cards, raised
PAPER, LABEL, FOG, SMOKE = "#E9EBEE", "#D3D7DD", "#A0A6B1", "#7C838F"  # text, from primary to tertiary
HAIR, GRID = "rgba(255,255,255,0.07)", "rgba(255,255,255,0.055)"
# Three voices: the paper's own serif for headlines and the big numbers, a grotesk for the interface,
# a monospace for every value the model computes.
SERIF = '"Source Serif 4", Georgia, serif'
SANS = '"Mona Sans", system-ui, sans-serif'
MONO = '"IBM Plex Mono", ui-monospace, monospace'
Y_RANGE = (-1.0, 1.75)  # one scale for every curve, so a flat curve looks flat
PLOT_CFG = {"displayModeBar": False, "scrollZoom": False}

st.set_page_config(page_title="KANVAS demo", page_icon=":material/memory:", layout="wide",
                   initial_sidebar_state="auto")


@st.cache_resource
def _load():
    return load_demo(HERE / "kaam_demo.json")


@st.cache_resource
def _figure_b64():
    return base64.b64encode((HERE / "assets" / "plan_cpu_curve_overlay_dark.png").read_bytes()).decode()


MODEL, B = _load()
NAMES = MODEL.names
D = B["train_dist"]
CASES = {c["name"]: c for c in B["cases"]}
LOOKUP = {tuple(r): (n, f) for r, n, f in zip(B["requests"]["rows"], B["requests"]["runs"], B["requests"]["failed"])}
KEY = B["requests"]["columns"]
TEMPLATE = tuple(B["template"]["values"][k] for k in KEY)


def _plural(d, word):
    return f"{d:g} {word}" + ("" if d == 1 else "s")


# label, display scale (raw / scale = what a person reads), formatter, input widget, slider step + printf
META = {
    "plan_cpu": dict(label="CPU request", scale=100, show=lambda d: _plural(d, "core"), kind="select"),
    # two decimals: at one, 29.296875 and 29.31640625 GB both read "29.3 GB" and the slider confuses them
    "plan_mem": dict(label="Memory request", scale=1, show=lambda d: f"{d:.2f} GB", kind="select"),
    "plan_gpu": dict(label="GPU request", scale=100, show=lambda d: f"{d:g} GPU" + ("s" if d > 1 else ""),
                     kind="select"),
    "inst_num": dict(label="Instances", scale=1, show=lambda d: _plural(d, "instance"), kind="select"),
    "cap_gpu": dict(label="Machine size", scale=1, show=lambda d: f"{d:g}-GPU machine", kind="radio"),
    "avg_cpu_usr": dict(label="Machine user CPU", scale=1, show=lambda d: f"{d:.1f}%", kind="slider",
                        step=0.1, fmt="%.1f%%"),
    "avg_gpu_util": dict(label="Machine GPU utilisation", scale=1, show=lambda d: f"{d:.0f}%", kind="slider",
                         step=1.0, fmt="%.0f%%", help="Summed over the machine's GPUs, so above 100% is normal."),
    "avg_cpu_kernel": dict(label="Machine kernel CPU", scale=1, show=lambda d: f"{d:.1f}%", kind="slider",
                           step=0.1, fmt="%.1f%%"),
    "avg_load_1": dict(label="Machine load average", scale=1, show=lambda d: f"{d:.1f}", kind="slider",
                       step=0.5, fmt="%.1f"),
    "avg_net_receive": dict(label="Machine network receive", scale=1e6, show=lambda d: f"{d:,.0f} MB/s",
                            kind="slider", step=1.0, fmt="%.0f MB/s"),
}
REQUEST = ["plan_cpu", "plan_mem", "plan_gpu", "inst_num"]
MACHINE = ["cap_gpu", "avg_cpu_kernel", "avg_cpu_usr", "avg_gpu_util", "avg_load_1", "avg_net_receive"]


@st.cache_resource
def _motion_js():
    labels = ["Average training task", "This task"] + [META[n]["label"] for n in NAMES]
    return (HERE / "motion.js").read_text().replace("__LABELS__", json.dumps(labels))


def show(name, raw):
    return META[name]["show"](raw / META[name]["scale"])


def signed(v, digits=2):
    return f"{v:+.{digits}f}".replace("-", "−")


def pct(p):
    return f"{100 * p:.1f}%"


def rgba(hex_color, a):
    h = hex_color.lstrip("#")
    return f"rgba({int(h[0:2], 16)},{int(h[2:4], 16)},{int(h[4:6], 16)},{a:.3f})"


def count(key, value, text, dec=0, sign=False, group=False, suffix="", reveal=False):
    """A number motion.js springs from its previous value to this one; the server text is the final value."""
    rev = ' data-reveal="1"' if reveal else ""  # no backslashes inside f-string braces: Python 3.11 rejects them
    return (f'<span data-kv="count" data-key="{key}" data-to="{value:.6f}" data-dec="{dec}" '
            f'data-sign="{int(sign)}" data-group="{int(group)}" data-suffix="{suffix}"{rev}>{text}</span>')


def percentile(name, raw):
    """Mid-rank percentile of a raw value among training tasks."""
    d = D[name]
    if "values" in d:
        v, c = np.asarray(d["values"]), np.asarray(d["counts"])
        return 100 * (c[v < raw].sum() + c[v == raw].sum() / 2) / c.sum()
    q = np.asarray(d["quantiles"])
    return float(np.interp(raw, q, np.linspace(0, 100, len(q))))


# ------------------------------------------------------------------ input state
# A slider can only show values on its step grid, but a real task's telemetry (22.43%) is not on it. The slider
# shows the nearest grid value, and an input nobody has touched keeps the task's exact recorded value, so a
# loaded task reproduces the paper's numbers exactly.
def _to_widget(name, raw):
    m = META[name]
    if m["kind"] != "slider":
        return raw
    return round(round(raw / m["scale"] / m["step"]) * m["step"], 6)


def _bounds(name):
    m, d = META[name], D[name]
    return (round(np.floor(d["lo"] / m["scale"] / m["step"]) * m["step"], 6),
            round(np.ceil(d["hi"] / m["scale"] / m["step"]) * m["step"], 6))


def _from_widget(name):
    v = st.session_state[f"in_{name}"]
    exact = st.session_state.get("exact")
    if exact is not None and v == _to_widget(name, exact[name]):
        return exact[name]
    return v * META[name]["scale"] if META[name]["kind"] == "slider" else v


def _set_inputs(values):
    st.session_state.exact = dict(values)
    for n in NAMES:
        st.session_state[f"in_{n}"] = _to_widget(n, values[n])


def _on_case():
    if st.session_state.case:
        _set_inputs(CASES[st.session_state.case]["values"])


def _on_edit(name):
    exact = st.session_state.get("exact")
    if exact is None or st.session_state[f"in_{name}"] != _to_widget(name, exact[name]):
        st.session_state.case = None


FIRST_RUN = "ready" not in st.session_state  # the entrance sequence plays once per page load
if FIRST_RUN:
    st.session_state.case = "True positive"
    _set_inputs(CASES["True positive"]["values"])
    st.session_state.curve = "auto"
    st.session_state.last_click = None
    st.session_state.ready = True

# ------------------------------------------------------------------ styling
st.markdown(f"""<style>
@font-face {{ font-family: "Mona Sans"; src: url("app/static/fonts/MonaSans-Variable.woff2") format("woff2");
  font-weight: 200 900; font-stretch: 75% 125%; font-display: swap; }}
.block-container {{ padding-top: 2.3rem; padding-bottom: 2rem; max-width: 1760px; }}
[data-testid="stHeader"] {{ background: transparent; }}
[data-testid="stAppViewContainer"] {{
  background: radial-gradient(1200px 520px at 78% -12%, rgba(91,168,255,0.05), transparent 70%), {INK1}; }}
/* slots that only carry styles, the motion engine or an empty placeholder take no space */
[data-testid="stElementContainer"]:has([data-testid="stHtml"]),
[data-testid="stElementContainer"]:has([data-testid="stMarkdownContainer"] > style:only-child),
[data-testid="stElementContainer"]:has(.kv-empty) {{ display: none; }}

/* ---------- type: serif voice, sans interface, mono values */
.stApp {{ letter-spacing: .004em; }}
.serif {{ font-family: {SERIF}; font-optical-sizing: auto; }}
.eyebrow {{ font: 600 .7rem/1 {SANS}; letter-spacing: .17em; text-transform: uppercase; color: {FOG}; }}
h1.title {{ font: 560 clamp(2rem, 1.2rem + 2.4vw, 3.05rem)/1.02 {SERIF}; font-optical-sizing: auto;
  letter-spacing: -.022em; color: {PAPER}; margin: .7rem 0 .55rem; padding: 0; }}
h1.title em {{ font-style: italic; font-weight: 400; color: {LABEL}; }}
h2.title {{ font: 560 1.95rem/1.1 {SERIF}; letter-spacing: -.018em; color: {PAPER}; margin: .7rem 0 .5rem; padding: 0; }}
h2.title em {{ font-style: italic; font-weight: 400; color: {LABEL}; }}
.sub {{ color: {FOG}; font-size: 1rem; line-height: 1.55; max-width: 112ch; margin-bottom: .2rem; }}
.mono {{ font-family: {MONO}; font-variant-numeric: tabular-nums; }}
.src {{ color: {SMOKE}; font-size: .78rem; letter-spacing: .014em; }}
.h3 {{ font: 600 1.32rem/1.2 {SERIF}; color: {PAPER}; letter-spacing: -.01em; }}
.h3 span {{ font: 450 .9rem/1.2 {SANS}; color: {FOG}; letter-spacing: .004em; margin-left: .45rem; }}
[data-testid="stCaptionContainer"], .wordmark-sub {{ letter-spacing: .014em; color: {SMOKE}; }}

/* ---------- status strip */
.strip {{ display: flex; flex-wrap: wrap; align-items: center; gap: .55rem .9rem; font: 500 .74rem/1 {SANS};
  letter-spacing: .06em; color: {FOG}; }}
.strip .live {{ display: inline-flex; align-items: center; gap: .5rem; padding: .36rem .65rem .36rem .55rem;
  border-radius: 999px; background: rgba(255,255,255,0.04); border: 1px solid {HAIR}; color: {PAPER};
  text-transform: uppercase; letter-spacing: .16em; font-weight: 650; font-size: .66rem; }}
.strip .live i {{ position: relative; width: 7px; height: 7px; border-radius: 50%; background: {PAPER}; }}
.strip .live i::after {{ content: ""; position: absolute; inset: -4px; border-radius: 50%;
  border: 1px solid {PAPER}; opacity: 0; animation: kvPulse 2.4s cubic-bezier(.2,.7,.2,1) infinite; }}
.strip b {{ font: 500 .78rem/1 {MONO}; color: {PAPER}; letter-spacing: 0; }}
.strip .sep {{ width: 3px; height: 3px; border-radius: 50%; background: {SMOKE}; opacity: .7; }}
@keyframes kvPulse {{ 0% {{ transform: scale(.6); opacity: .7; }} 70%, 100% {{ transform: scale(1.9); opacity: 0; }} }}
@media (max-width: 640px) {{ .strip {{ gap: .5rem .8rem; }} .strip .sep {{ display: none; }} }}

/* ---------- surfaces */
.card {{ position: relative; background: {INK2}; border: 1px solid {HAIR}; border-radius: 16px;
  box-shadow: inset 0 1px 0 rgba(255,255,255,0.05), 0 18px 40px -26px rgba(0,0,0,0.9);
  padding: 1.05rem 1.25rem 1.15rem; height: 100%; overflow: hidden; }}
.hero {{ min-height: 13.2rem; }}
.st-key-panel_wf, .st-key-panel_curve {{ background: {INK2}; border: 1px solid {HAIR}; border-radius: 16px;
  box-shadow: inset 0 1px 0 rgba(255,255,255,0.05), 0 18px 40px -26px rgba(0,0,0,0.9);
  padding: 1rem 1.15rem .55rem; gap: .35rem; }}

/* ---------- the readout: the glow is the prediction (hue = side of the threshold, strength = distance) */
.readout .glow {{ position: absolute; inset: 0; pointer-events: none; }}
.readout .glow.e {{ background: radial-gradient(130% 150% at 0% 0%, rgba(255,106,85,1), transparent 62%); }}
.readout .glow.g {{ background: radial-gradient(130% 150% at 0% 0%, rgba(91,168,255,1), transparent 62%); }}
.readout .inner {{ position: relative; }}
.risk {{ display: flex; align-items: baseline; gap: .2rem; margin: .45rem 0 .6rem; color: var(--tone); }}
.risk .n {{ font: 300 5rem/0.9 {SERIF}; font-optical-sizing: auto; letter-spacing: -.035em;
  font-variant-numeric: lining-nums tabular-nums; }}
.risk .u {{ font: 300 2rem/1 {SERIF}; opacity: .75; }}
.rail {{ position: relative; height: 6px; border-radius: 3px; background: rgba(255,255,255,0.08); }}
.rail .fill {{ position: absolute; inset: 0 auto 0 0; border-radius: 3px; background: var(--tone); opacity: .9;
  box-shadow: 0 0 14px var(--edge); }}
.rail .notch {{ position: absolute; left: 50%; top: -5px; bottom: -5px; width: 2px; margin-left: -1px;
  background: {PAPER}; opacity: .55; border-radius: 1px; }}
.scale {{ display: flex; justify-content: space-between; margin-top: .45rem; font: 500 .7rem/1 {MONO}; color: {SMOKE}; }}
.verdict {{ margin-top: .85rem; }}
.pill {{ font: 650 .7rem/1 {SANS}; letter-spacing: .14em; text-transform: uppercase; padding: .4rem .65rem;
  border-radius: 999px; color: var(--tone); background: var(--chip); border: 1px solid var(--edge); }}

/* ---------- the ledger: the project's claim, set as a sum */
.ledger .row, .ledger .total {{ display: flex; justify-content: space-between; align-items: baseline; gap: 1rem; }}
.ledger .row {{ padding: .42rem 0; }}
.ledger .lbl {{ color: {FOG}; font-size: .93rem; }}
.ledger .val {{ font: 500 1.08rem/1 {MONO}; color: {PAPER}; font-variant-numeric: tabular-nums; }}
.ledger .total {{ border-top: 1px solid rgba(255,255,255,0.16); margin-top: .35rem; padding-top: .7rem; }}
.ledger .total .lbl {{ font: 600 1.06rem/1.2 {SERIF}; color: {PAPER}; }}
.ledger .total .val {{ font-size: 1.36rem; font-weight: 600; }}
.ledger .note {{ margin-top: .75rem; color: {SMOKE}; font-size: .82rem; }}
.ledger .note b {{ color: {FOG}; font-weight: 550; }}

/* ---------- facts */
.fact {{ display: flex; align-items: baseline; gap: .7rem; padding: .3rem 0; color: {FOG}; font-size: .92rem; }}
.fact .big {{ font: 500 1.45rem/1 {MONO}; color: {PAPER}; min-width: 5.2ch; font-variant-numeric: tabular-nums; }}
.divider {{ height: 1px; background: {HAIR}; margin: .7rem 0 .75rem; }}
.outcome {{ font-size: .95rem; color: {LABEL}; }}
.outcome b {{ font-weight: 600; }}

/* ---------- the one warning carries the "raises risk" hue, nothing louder */
.banner {{ display: flex; gap: .8rem; align-items: center; margin: .25rem 0 .2rem; padding: .7rem 1rem;
  border-radius: 12px; background: linear-gradient(90deg, rgba(255,106,85,0.11), rgba(255,106,85,0.02) 60%), {INK2};
  border: 1px solid rgba(255,106,85,0.22); color: {LABEL}; font-size: .93rem;
  animation: kvDrop .5s cubic-bezier(.2,.85,.25,1) both; }}
.banner .dot {{ width: 8px; height: 8px; border-radius: 50%; background: {EMBER}; flex: none;
  box-shadow: 0 0 0 4px rgba(255,106,85,0.16); }}
@keyframes kvDrop {{ from {{ opacity: 0; transform: translateY(-6px); }} }}

/* ---------- audit */
.stat .n {{ font: 300 3.1rem/1 {SERIF}; font-optical-sizing: auto; letter-spacing: -.03em; color: {PAPER};
  font-variant-numeric: lining-nums tabular-nums; }}
.stat .what {{ color: {LABEL}; font-size: .95rem; margin: .6rem 0 .7rem; min-height: 2.6em; }}
.figure {{ padding: .6rem; }}
.figure img {{ display: block; width: 100%; border-radius: 10px; }}
/* audit charts build when the tab opens (a CSS animation restarts whenever its element becomes visible) */
.st-key-auc_drop .barlayer .point path {{ transform-box: fill-box; transform-origin: 0 50%;
  animation: kvGrowX .9s cubic-bezier(.2,.85,.25,1) both; }}
.st-key-auc_drop .barlayer .point:nth-child(2) path {{ animation-delay: .12s; }}
.st-key-auc_drop .barlayer .point:nth-child(3) path {{ animation-delay: .24s; }}
.st-key-auc_drop .barlayer .point:nth-child(4) path {{ animation-delay: .5s; animation-duration: 1.2s; }}
.st-key-one_core .barlayer .point path {{ transform-box: fill-box; transform-origin: 50% 100%;
  animation: kvGrowY .9s cubic-bezier(.2,.85,.25,1) both; }}
.st-key-one_core .barlayer .point:nth-child(2) path {{ animation-delay: .14s; }}
.st-key-one_core .barlayer .point:nth-child(3) path {{ animation-delay: .28s; }}
.st-key-auc_drop .annotation, .st-key-auc_drop .errorbar, .st-key-one_core .bartext {{
  animation: kvFade .5s .75s ease-out both; }}
@keyframes kvGrowX {{ from {{ transform: scaleX(0); }} }}
@keyframes kvGrowY {{ from {{ transform: scaleY(0); }} }}
@keyframes kvFade {{ from {{ opacity: 0; }} }}

/* ---------- chrome */
section[data-testid="stSidebar"] .eyebrow {{ color: {SMOKE}; margin: 1.1rem 0 .25rem; }}
.wordmark {{ font: 700 1.05rem/1 {SANS}; font-stretch: 125%; letter-spacing: .2em; color: {PAPER}; }}
.wordmark-sub {{ font: italic 400 .92rem/1.35 {SERIF}; color: {FOG}; margin-top: .5rem; }}
[data-testid="stTabs"] button p {{ font-size: .95rem; font-weight: 550; }}
[data-testid="stExpander"] details {{ background: {INK2}; border-color: {HAIR} !important; border-radius: 16px; }}
[data-testid="stButtonGroup"] button {{ transition: transform 90ms ease-out, background-color 140ms ease-out; }}
[data-testid="stButtonGroup"] button:active {{ transform: scale(.97); }}

@media (prefers-reduced-motion: reduce) {{
  .strip .live i::after, .banner, [class*="st-key-"] .barlayer .point path, .st-key-auc_drop .annotation,
  .st-key-auc_drop .errorbar, .st-key-one_core .bartext {{ animation: none !important; }}
  [data-testid="stButtonGroup"] button {{ transition: none; }}
  [data-testid="stButtonGroup"] button:active {{ transform: none; }}
}}
@media (prefers-reduced-transparency: reduce) {{ .readout .glow {{ display: none; }} }}
</style>""", unsafe_allow_html=True)

# Entrance: reading order, once per page load. The element is emitted on every run (empty after the first),
# because Streamlit tracks elements by position: dropping it would shift and remount everything below it.
st.markdown("""<style>
@keyframes kvRise { from { opacity: 0; transform: translateY(12px); } }
.kv-in { animation: kvRise .75s cubic-bezier(.2,.85,.25,1) both; animation-delay: var(--d, 0ms); }
[data-testid="stTabs"] [role="tablist"] { animation: kvRise .75s .24s cubic-bezier(.2,.85,.25,1) both; }
[data-testid="stButtonGroup"] { animation: kvRise .75s .3s cubic-bezier(.2,.85,.25,1) both; }
.st-key-panel_wf { animation: kvRise .8s .52s cubic-bezier(.2,.85,.25,1) both; }
.st-key-panel_curve { animation: kvRise .8s .6s cubic-bezier(.2,.85,.25,1) both; }
section[data-testid="stSidebar"] [data-testid="stVerticalBlock"] > div { animation: kvRise .6s .1s ease-out both; }
@media (prefers-reduced-motion: reduce) {
  .kv-in, [data-testid="stTabs"] [role="tablist"], [data-testid="stButtonGroup"], .st-key-panel_wf,
  .st-key-panel_curve, section[data-testid="stSidebar"] [data-testid="stVerticalBlock"] > div {
    animation-name: kvFadeOnly !important; }
  @keyframes kvFadeOnly { from { opacity: 0; } }
}
</style>""" if FIRST_RUN else "<style></style>", unsafe_allow_html=True)

st.html(f"<script>{_motion_js()}</script>", unsafe_allow_javascript=True)

# ------------------------------------------------------------------ sidebar: the ten inputs
with st.sidebar:
    st.markdown('<div class="wordmark">KANVAS</div><div class="wordmark-sub">Ten inputs. Each one has its own '
                'curve.</div>', unsafe_allow_html=True)
    st.markdown('<div class="eyebrow">Job request</div>', unsafe_allow_html=True)
    st.caption("What the task asked the scheduler for.")
    for n in REQUEST:
        st.select_slider(META[n]["label"], options=D[n]["choices"], key=f"in_{n}", on_change=_on_edit, args=(n,),
                         format_func=lambda raw, n=n: show(n, raw))
    st.markdown('<div class="eyebrow">Machine it landed on</div>', unsafe_allow_html=True)
    st.caption("Other jobs' telemetry on that machine in the hour before the task started.")
    for n in MACHINE:
        m = META[n]
        if m["kind"] == "radio":
            st.radio(m["label"], D[n]["choices"], key=f"in_{n}", on_change=_on_edit, args=(n,), horizontal=True,
                     format_func=lambda raw, n=n: show(n, raw))
        else:
            lo, hi = _bounds(n)
            st.slider(m["label"], min_value=lo, max_value=hi, step=m["step"], format=m["fmt"], key=f"in_{n}",
                      on_change=_on_edit, args=(n,), help=m.get("help"))

# ------------------------------------------------------------------ the model, live
raw = np.array([_from_widget(n) for n in NAMES], dtype=np.float64)
contrib = MODEL.explain(raw)
base = MODEL.baseline_logit
logit = float(MODEL.logit(raw[None])[0])
risk = 1.0 / (1.0 + np.exp(-logit))
flagged = risk >= B["threshold"]
request_key = tuple(float(raw[NAMES.index(k)]) for k in KEY)
runs, failed = LOOKUP.get(request_key, (0, 0))
case = next((c for c in B["cases"]
             if all(abs(raw[i] - c["values"][n]) <= 1e-9 * max(1.0, abs(c["values"][n])) for i, n in enumerate(NAMES))),
            None)
order = sorted(NAMES, key=lambda n: -abs(contrib[n]))


# ------------------------------------------------------------------ charts
def _layout(fig, height, **kw):
    fig.update_layout(template="none", height=height, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      font=dict(family=SANS, size=13, color=FOG), showlegend=False,
                      hoverlabel=dict(bgcolor=INK3, bordercolor="rgba(255,255,255,0.12)",
                                      font=dict(family=SANS, size=13, color=PAPER)), **kw)
    fig.update_xaxes(automargin=True, linecolor=HAIR, tickcolor=HAIR)
    fig.update_yaxes(automargin=True, linecolor=HAIR, tickcolor=HAIR)
    return fig


def _num_axis(**kw):
    return dict(tickfont=dict(family=MONO, size=11.5, color=SMOKE), gridcolor=GRID, zeroline=False, **kw)


def waterfall_fig():
    vals = np.array([contrib[n] for n in order])
    ends = base + np.cumsum(vals)
    starts = ends - vals
    rows = [f"{META[n]['label']}  <span style='color:{SMOKE}'>{show(n, raw[NAMES.index(n)])}</span>" for n in order]
    cats = ["Average training task"] + rows + ["This task"]
    fig = go.Figure()
    fig.add_bar(y=rows, x=vals, base=starts, orientation="h", width=0.62, customdata=order,
                marker=dict(color=[EMBER if v > 0 else GLACIER for v in vals], line=dict(width=0), cornerradius=3),
                text=[signed(v) for v in vals], textposition="outside", cliponaxis=False,
                textfont=dict(family=MONO, size=12.5, color=FOG),
                hovertext=[f"{META[n]['label']}: {show(n, raw[NAMES.index(n)])} (training percentile "
                           f"{percentile(n, raw[NAMES.index(n)]):.0f})<br>{signed(contrib[n], 3)} log-odds vs the "
                           f"average training task<br>click to see this feature's curve" for n in order],
                hoverinfo="text", selected=dict(marker=dict(opacity=1)), unselected=dict(marker=dict(opacity=0.3)))
    fig.add_scatter(y=[cats[0], cats[-1]], x=[base, logit], mode="markers+text", hoverinfo="skip",
                    marker=dict(size=12, color=PAPER, line=dict(color=INK2, width=3)),
                    text=[f"  {signed(base)}  <span style='color:{SMOKE}'>risk {pct(1 / (1 + np.exp(-base)))}</span>",
                          f"  {signed(logit)}  <span style='color:{SMOKE}'>risk {pct(risk)}</span>"],
                    textposition="middle right", textfont=dict(family=MONO, size=13, color=PAPER), cliponaxis=False)
    # connectors: each bar starts where the one above it ended
    for k, x in enumerate([base] + list(ends)):
        fig.add_shape(type="line", x0=x, x1=x, y0=k + 0.31, y1=k + 0.69, line=dict(color="rgba(255,255,255,0.22)",
                                                                                     width=1))
    fig.add_vline(x=0, line=dict(color="rgba(255,255,255,0.32)", width=1, dash="dot"))
    fig.add_annotation(x=0, y=-1.05, yref="y", text="flag threshold, risk 50%", showarrow=False,
                       font=dict(family=SANS, size=11.5, color=FOG), yanchor="middle", bgcolor=INK2, borderpad=3)
    lo = min(starts.min(), ends.min(), base, 0.0)
    hi = max(starts.max(), ends.max(), base, 0.0)
    span = hi - lo
    _layout(fig, 478, margin=dict(l=70, r=12, t=4, b=46), bargap=0.3, clickmode="event+select", meta={"kind": "wf"},
            xaxis=_num_axis(range=[lo - 0.08 * span, hi + 0.45 * span], showgrid=True,
                            title=dict(text="log-odds of failure  ·  each bar: change vs the average training task",
                                       font=dict(size=12, color=SMOKE))),
            yaxis=dict(categoryorder="array", categoryarray=cats, range=[11.6, -1.45], automargin=True, showgrid=False,
                       tickfont=dict(family=SANS, size=13.5, color=LABEL), ticksuffix="   "))
    return fig


def _hist(name):
    d, s = D[name], META[name]["scale"]
    if "values" in d:
        x = np.asarray(d["values"]) / s
        return x, np.asarray(d["counts"]), (D[name]["hi"] - D[name]["lo"]) / s * 0.018
    e = np.asarray(d["hist"]["edges"]) / s
    return (e[:-1] + e[1:]) / 2, np.asarray(d["hist"]["counts"]), (e[1] - e[0]) * 0.82


def curve_fig(name, height=500):
    s, d = META[name]["scale"], D[name]
    g = np.linspace(d["lo"], d["hi"], 400)
    y = MODEL.phi(name, g) - MODEL.phi_mean[name]
    xv, c = raw[NAMES.index(name)] / s, contrib[name]
    tone = EMBER if c > 0 else GLACIER
    sparse = "values" in d and len(d["values"]) <= 3
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    hx, hc, hw = _hist(name)
    fig.add_bar(x=hx, y=hc, width=hw, marker=dict(color="rgba(160,166,177,0.30)", line=dict(width=0), cornerradius=1.5),
                secondary_y=True, hovertemplate="%{y:,} training tasks<extra></extra>")
    for x0, x1 in ((d["lo"], d["p5"]), (d["p95"], d["hi"])):
        if x1 > x0:
            fig.add_vrect(x0=x0 / s, x1=x1 / s, fillcolor="rgba(255,255,255,0.028)", line_width=0, layer="below")
    fig.add_hline(y=0, line=dict(color="rgba(255,255,255,0.16)", width=1))
    if sparse:  # two real values: draw the points, and only hint at the spline between them
        vx = np.asarray(d["values"])
        fig.add_scatter(x=g / s, y=y, mode="lines", hoverinfo="skip", secondary_y=False,
                        line=dict(color=SMOKE, width=1.3, dash="dot"))
        fig.add_scatter(x=vx / s, y=MODEL.phi(name, vx) - MODEL.phi_mean[name], mode="markers", hoverinfo="skip",
                        marker=dict(size=10, color=LABEL), secondary_y=False)
    else:  # a soft halo under the line, so the learned curve reads as the one luminous thing in the panel
        fig.add_scatter(x=g / s, y=y, mode="lines", hoverinfo="skip", secondary_y=False,
                        line=dict(color="rgba(233,235,238,0.07)", width=11))
        fig.add_scatter(x=g / s, y=y, mode="lines", hoverinfo="skip", secondary_y=False,
                        line=dict(color=PAPER, width=2.4))
    fig.add_scatter(x=[xv], y=[c], mode="markers", hoverinfo="skip", secondary_y=False,
                    marker=dict(size=38, color=rgba(tone, 0.20), line=dict(width=0)))
    fig.add_scatter(x=[xv], y=[c], mode="markers+text", secondary_y=False,
                    marker=dict(size=15, color=tone, line=dict(color=INK2, width=2.5)),
                    text=[f"    {signed(c)}"], textposition="middle right",
                    textfont=dict(family=MONO, size=15, color=PAPER), cliponaxis=False,
                    hovertemplate=f"{show(name, raw[NAMES.index(name)])}: {signed(c, 3)} log-odds<extra></extra>")
    _layout(fig, height, margin=dict(l=8, r=8, t=8, b=46), meta={"kind": "curve", "feature": name})
    fig.update_xaxes(**_num_axis(showgrid=False), title=dict(text=META[name]["label"], font=dict(size=12, color=SMOKE)),
                     ticksuffix={"plan_cpu": " cores", "plan_gpu": " GPU", "plan_mem": " GB",
                                 "avg_net_receive": " MB/s"}.get(name, ""))
    fig.update_yaxes(**_num_axis(range=Y_RANGE), secondary_y=False,
                     title=dict(text="log-odds vs average task", font=dict(size=12, color=SMOKE)))
    fig.update_yaxes(range=[0, max(hc) * 4.6], visible=False, secondary_y=True)
    return fig


def minis_fig():
    """All ten curves as one figure (one chart to redraw per rerun instead of ten), on one shared scale."""
    fig = make_subplots(rows=2, cols=5, subplot_titles=[META[n]["label"] for n in NAMES],
                        horizontal_spacing=0.035, vertical_spacing=0.17)
    for i, n in enumerate(NAMES):
        r, c = i // 5 + 1, i % 5 + 1
        s, d = META[n]["scale"], D[n]
        g = np.linspace(d["lo"], d["hi"], 200)
        y = MODEL.phi(n, g) - MODEL.phi_mean[n]
        cv = contrib[n]
        fig.add_hline(y=0, line=dict(color="rgba(255,255,255,0.12)", width=1), row=r, col=c)
        if "values" in d and len(d["values"]) <= 3:
            vx = np.asarray(d["values"])
            fig.add_scatter(x=g / s, y=y, mode="lines", line=dict(color=SMOKE, width=1.2, dash="dot"),
                            hoverinfo="skip", row=r, col=c)
            fig.add_scatter(x=vx / s, y=MODEL.phi(n, vx) - MODEL.phi_mean[n], mode="markers", hoverinfo="skip",
                            marker=dict(size=7, color=LABEL), row=r, col=c)
        else:
            fig.add_scatter(x=g / s, y=y, mode="lines", line=dict(color=PAPER, width=1.8), hoverinfo="skip",
                            row=r, col=c)
        fig.add_scatter(x=[raw[NAMES.index(n)] / s], y=[cv], mode="markers", row=r, col=c,
                        marker=dict(size=10, color=EMBER if cv > 0 else GLACIER, line=dict(color=INK2, width=2)),
                        hovertemplate=f"{META[n]['label']}: {show(n, raw[NAMES.index(n)])}<br>"
                                      f"{signed(cv, 3)} log-odds<extra></extra>")
    _layout(fig, 400, margin=dict(l=6, r=6, t=34, b=6), meta={"kind": "mini"})
    fig.update_xaxes(showticklabels=False, showgrid=False, showline=False, zeroline=False)
    fig.update_yaxes(range=Y_RANGE, showticklabels=False, showgrid=False, zeroline=False, showline=False)
    fig.update_annotations(font=dict(family=SANS, size=12.5, color=LABEL))
    for ann, ax in zip(fig.layout.annotations, [fig.layout[f"xaxis{'' if i == 0 else i + 1}"] for i in range(10)]):
        ann.update(x=ax.domain[0], xanchor="left")
    return fig


# ------------------------------------------------------------------ page
st.markdown(
    f'<div class="strip kv-in" style="--d:0ms"><span class="live"><i></i>Live model</span>'
    f'<span>Canonical KAAM</span><span class="sep"></span><span><b>151</b> parameters</span><span class="sep"></span>'
    f'<span>test AUC <b>{B["metrics"]["test_auc"]:.3f}</b></span><span class="sep"></span>'
    f'<span><b>{B["metrics"]["test_tasks"]:,}</b> held-out tasks</span><span class="sep"></span>'
    f'<span>Alibaba PAI GPU trace</span></div>'
    '<h1 class="title kv-in" style="--d:80ms">Will this GPU job fail, <em>and why?</em></h1>'
    f'<div class="sub kv-in" style="--d:160ms">A 151-parameter additive model trained on '
    f'{B["metrics"]["train_tasks"]:,} real tasks from Alibaba\'s PAI GPU cluster. Every prediction splits exactly '
    'into ten contributions, one per input, with no sampling or approximation.</div>', unsafe_allow_html=True)

# The limitation, stated before anything else can be read as a promise (paper, user-grouped and time-ordered results).
st.markdown('<div class="banner kv-in" role="note" style="--d:200ms"><span class="dot"></span><span><b>Limitation: '
            'do not use this model for new users.</b> It works best for users whose earlier jobs are in its training '
            'data. For users it has never seen, its risk scores are much less reliable, and for some groups of new '
            'users they were no better than a coin flip in our tests. It is also less accurate on jobs from a later '
            'period than the one it was trained on.</span></div>', unsafe_allow_html=True)

TAB_PREDICT, TAB_AUDIT = "Predict and explain", "Audit: what the explanations exposed"
tab_predict, tab_audit = st.tabs([TAB_PREDICT, TAB_AUDIT], key="tab", on_change="rerun")

with tab_predict:
    top = st.columns([2.2, 3], vertical_alignment="bottom")
    with top[0]:
        st.segmented_control("Load a real task from the held-out test set", list(CASES), key="case",
                             on_change=_on_case)
    with top[1]:
        if case:
            outcome = "failed" if case["true_label"] else "finished normally"
            st.markdown(f'<div class="src" style="font-size:.86rem;padding-bottom:.78rem">Test task picked by a fixed '
                        f'rule: <span style="color:{LABEL}">{case["rule"]}</span>. It actually '
                        f'<span style="color:{LABEL}">{outcome}</span>.</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="src" style="font-size:.86rem;padding-bottom:.78rem">Custom input. Move any slider '
                        'on the left and every number below updates.</div>', unsafe_allow_html=True)

    tone = EMBER if flagged else GLACIER
    strength = 0.10 + 0.42 * min(1.0, abs(risk - B["threshold"]) / 0.5)  # glow grows with distance from the line
    glow_e, glow_g = (strength, 0.0) if flagged else (0.0, strength)
    c1, c2, c3 = st.columns([1.05, 1.15, 1.2], gap="medium")
    with c1:
        st.markdown(
            f'<div class="card hero readout kv-in" style="--d:360ms;--tone:{tone};--chip:{rgba(tone, 0.12)};'
            f'--edge:{rgba(tone, 0.35)}">'
            f'<div class="glow e" data-kv="style" data-key="glow-e" data-prop="opacity" data-to="{glow_e:.4f}" '
            f'style="opacity:{glow_e:.4f}"></div>'
            f'<div class="glow g" data-kv="style" data-key="glow-g" data-prop="opacity" data-to="{glow_g:.4f}" '
            f'style="opacity:{glow_g:.4f}"></div>'
            f'<div class="inner"><div class="eyebrow">Predicted failure risk</div>'
            f'<div class="risk"><span class="n">{count("risk", 100 * risk, f"{100 * risk:.1f}", dec=1)}</span>'
            f'<span class="u">%</span></div>'
            f'<div class="rail"><div class="fill" data-kv="style" data-key="rail" data-prop="width" data-unit="%" '
            f'data-to="{100 * risk:.4f}" style="width:{100 * risk:.4f}%"></div><div class="notch"></div></div>'
            f'<div class="scale"><span>0</span><span>flag at 50</span><span>100</span></div>'
            f'<div class="verdict"><span class="pill">{"Flagged" if flagged else "Not flagged"}</span></div>'
            f'</div></div>', unsafe_allow_html=True)
    with c2:
        total = sum(contrib.values())
        st.markdown(
            f'<div class="card hero ledger kv-in" style="--d:430ms"><div class="eyebrow">The arithmetic</div>'
            f'<div class="row" style="margin-top:.55rem"><span class="lbl">Average training task</span>'
            f'<span class="val">{count("l-base", base, signed(base, 3), dec=3, sign=True)}</span></div>'
            f'<div class="row"><span class="lbl">Ten pushes, the bars below</span>'
            f'<span class="val">{count("l-sum", total, signed(total, 3), dec=3, sign=True)}</span></div>'
            f'<div class="total"><span class="lbl">This task, in log-odds</span>'
            f'<span class="val">{count("l-logit", logit, signed(logit, 3), dec=3, sign=True)}</span></div>'
            f'<div class="note">sigmoid({signed(logit, 3)}) = <b>{pct(risk)}</b>. The bars add up exactly.</div>'
            f'</div>', unsafe_allow_html=True)
    with c3:
        if runs:
            facts = (f'<div class="fact"><span class="big">{count("runs", runs, f"{runs:,}", group=True)}</span>'
                     f'training runs made this exact request on this machine size</div>'
                     f'<div class="fact"><span class="big">'
                     f'{count("fail", 100 * failed / runs, pct(failed / runs), dec=1, suffix="%")}</span>'
                     f'of those runs failed</div>')
        else:
            facts = ('<div class="fact" style="padding-top:.5rem">No training task made this exact request on this '
                     'machine size. The model is combining what it learned about each input separately.</div>')
        if case:
            right = (case["true_label"] == 1) == flagged
            what = {(1, True): "a caught failure", (0, False): "a correct all-clear",
                    (1, False): "a missed failure", (0, True): "a false alarm"}[(case["true_label"], bool(flagged))]
            facts += (f'<div class="divider"></div><div class="outcome">Actually '
                      f'<b>{"failed" if case["true_label"] else "succeeded"}</b>, so this is '
                      f'<b style="color:{PAPER if right else EMBER}">{what}</b>.</div>')
        st.markdown(f'<div class="card hero kv-in" style="--d:500ms"><div class="eyebrow">In the training data</div>'
                    f'<div style="margin-top:.55rem">{facts}</div></div>', unsafe_allow_html=True)

    # always one element in this slot (an empty div when there is no banner) so nothing below it remounts
    st.markdown('<div class="banner"><span class="dot"></span><span>This exact request is mostly one user\'s '
                'recurring job: 99.7% of its 1,492 training runs came from a single user. The Audit tab shows what '
                'that does to the model.</span></div>' if request_key == TEMPLATE else '<div class="kv-empty"></div>',
                unsafe_allow_html=True)

    left, right_col = st.columns([1.2, 1], gap="medium")
    with left, st.container(key="panel_wf"):
        st.markdown('<div class="h3">Why<span>each input\'s push, largest first</span></div>', unsafe_allow_html=True)
        event = st.plotly_chart(waterfall_fig(), key="wf", on_select="rerun", selection_mode="points",
                                config=PLOT_CFG, theme=None)
        points = event.selection.points if event and event.selection else []
        clicked = points[0].get("customdata") if points else None
        if isinstance(clicked, list):
            clicked = clicked[0]
        if clicked and clicked != st.session_state.last_click:
            st.session_state.curve = clicked
        st.session_state.last_click = clicked
    with right_col, st.container(key="panel_curve"):
        if st.session_state.curve not in ["auto"] + NAMES:
            st.session_state.curve = "auto"
        shown = order[0] if st.session_state.curve == "auto" else st.session_state.curve
        hdr = st.columns([1.5, 1], vertical_alignment="center")
        hdr[0].markdown(f'<div class="h3">The curve<span>behind the {META[shown]["label"]} bar</span></div>',
                        unsafe_allow_html=True)
        # labels must not change between reruns: this Streamlit keys selectbox options by their label
        hdr[1].selectbox("Curve", ["auto"] + NAMES, key="curve", label_visibility="collapsed",
                         format_func=lambda n: "Largest bar" if n == "auto" else META[n]["label"])
        st.plotly_chart(curve_fig(shown, height=418), key="curve_plot", config=PLOT_CFG, theme=None)
        st.caption("The same curve applies to every task, whatever the other nine inputs are. Grey bars: training "
                   "tasks at each value. Shaded: outside the middle 90% of the data, where the curve has least support.")

    with st.expander("All ten curves on one scale"):
        st.plotly_chart(minis_fig(), key="minis", config=PLOT_CFG, theme=None)

    with st.expander("The full breakdown, as numbers"):
        rows = [{"Input": META[n]["label"], "Value": show(n, raw[NAMES.index(n)]),
                 "Training percentile": round(percentile(n, raw[NAMES.index(n)])),
                 "φ(x)": round(float(MODEL.phi(n, [raw[NAMES.index(n)]])[0]), 4),
                 "Training mean of φ": round(MODEL.phi_mean[n], 4),
                 "Contribution vs average task": round(contrib[n], 4)} for n in order]
        st.dataframe(rows, hide_index=True)
        st.caption(f"Bias {signed(float(MODEL.bias), 4)} + training means {signed(base - float(MODEL.bias), 4)} = "
                   f"average task {signed(base, 4)}. For an additive model these centred contributions are the exact "
                   "SHAP values: no sampling, no approximation.")

with tab_audit:
    P, N6, T = B["paper"], B["notebook06"], B["template"]
    st.markdown('<h2 class="title">The explanations exposed <em>a shortcut</em></h2>', unsafe_allow_html=True)
    st.markdown(f'<div class="sub">The model\'s most confident catch in the test set and its most confident false '
                f'alarm made the same request: {show("plan_cpu", 100)}, '
                f'{show("plan_mem", T["values"]["plan_mem"])}, {show("plan_gpu", T["values"]["plan_gpu"])}, '
                f'{show("inst_num", 1)}, on an {show("cap_gpu", 8)}. Following that bar back into the data:</div>',
                unsafe_allow_html=True)
    st.write("")

    s = st.columns(4, gap="medium")
    runs_share = 100 * T["train_failed"] / T["train_runs"]
    stats = [
        (count("a-runs", T["train_runs"], f'{T["train_runs"]:,}', group=True, reveal=True),
         "training runs of that one request", "computed here from the published data"),
        (count("a-fail", runs_share, f"{runs_share:.1f}%", dec=1, suffix="%", reveal=True),
         "of them failed", "computed here from the published data"),
        (count("a-user", P["false_alarm_request_one_user_share"], f'{P["false_alarm_request_one_user_share"]}%', dec=1,
               suffix="%", reveal=True), "of those runs came from one user", "paper §6.10"),
        (count("a-share", P["top_user"]["failure_share"], f'{P["top_user"]["failure_share"]:.1f}%', dec=1, suffix="%",
               reveal=True), "of all failures came from that user",
         f'who sent {P["top_user"]["task_share"]:.1f}% of all tasks · paper §6.11'),
    ]
    for col, (num, what, src) in zip(s, stats):
        col.markdown(f'<div class="card stat"><div class="n">{num}</div><div class="what">{what}</div>'
                     f'<div class="src">{src}</div></div>', unsafe_allow_html=True)

    st.write("")
    a, b = st.columns([1.15, 1], gap="large")
    with a:
        st.markdown('<div class="h3" style="padding:.3rem 0 .2rem">Remove that user\'s tasks and the score falls.'
                    '<span>On new users it is no better than a coin flip.</span></div>', unsafe_allow_html=True)
        auc = [("All 2,834 test tasks", B["metrics"]["test_auc"], None, "computed here"),
               (f'Without the {T["test_tasks"]} test tasks making that request', T["test_auc_without"], None,
                "computed here"),
               ("Same model, that user's test tasks removed", P["auc_without_top_user"], None, "paper §6.10"),
               ("Retrained on a user-grouped split (unseen users)", P["auc"]["KAAM"][1], P["auc"]["KAAM"][2],
                "paper §6.11, 3 seeds")]
        fig = go.Figure()
        fig.add_bar(y=[r[0] for r in auc], x=[r[1] for r in auc], orientation="h", width=0.56, hoverinfo="skip",
                    error_x=dict(type="data", array=[r[2] or 0 for r in auc], visible=True, color=FOG, thickness=1.2),
                    marker=dict(color=[GLACIER, rgba(GLACIER, 0.45), rgba(GLACIER, 0.45), EMBER], line=dict(width=0),
                                cornerradius=3))
        for label, v, err, src in auc:  # labels clear of the error bar and the coin-flip line
            fig.add_annotation(y=label, x=max(v + (err or 0) + 0.015, 0.515), xanchor="left", showarrow=False,
                               text=f"{v:.3f}" + (f" ± {err:.3f}" if err else "") +
                               f"   <span style='font-family:{SANS};color:{SMOKE}'>{src}</span>",
                               font=dict(family=MONO, size=13, color=PAPER))
        fig.add_vline(x=0.5, line=dict(color="rgba(255,255,255,0.32)", dash="dot", width=1))
        fig.add_annotation(x=0.5, y=-0.78, text="coin flip, 0.5", showarrow=False, bgcolor=INK1, borderpad=3,
                           font=dict(family=SANS, size=11.5, color=FOG), yanchor="bottom")
        _layout(fig, 330, margin=dict(l=8, r=8, t=8, b=40), meta={"kind": "audit"},
                xaxis=_num_axis(range=[0, 1.34], tickvals=[0, 0.2, 0.4, 0.6, 0.8, 1.0],
                                title=dict(text="test AUC", font=dict(size=12, color=SMOKE))),
                yaxis=dict(range=[3.6, -1.0], tickfont=dict(family=SANS, size=13.5, color=LABEL), ticksuffix="   ",
                           showgrid=False))
        st.plotly_chart(fig, key="auc_drop", config=PLOT_CFG, theme=None)
    with b:
        st.markdown('<div class="h3" style="padding:.3rem 0 .2rem">In training, the one-core effect is one person'
                    '</div>', unsafe_allow_html=True)
        rates = [(f'1-core tasks,<br>that user ({N6["one_core_top_user"]["tasks"]:,})',
                  N6["one_core_top_user"]["failure_rate"], EMBER),
                 (f'1-core tasks,<br>{N6["one_core_other_users"]["users"]} other users '
                  f'({N6["one_core_other_users"]["tasks"]:,})', N6["one_core_other_users"]["failure_rate"], GLACIER),
                 (f'Everything else<br>({N6["not_one_core"]["tasks"]:,})', N6["not_one_core"]["failure_rate"],
                  "rgba(160,166,177,0.45)")]
        fig = go.Figure()
        fig.add_bar(x=[r[0] for r in rates], y=[r[1] for r in rates], width=0.5, hoverinfo="skip",
                    marker=dict(color=[r[2] for r in rates], line=dict(width=0), cornerradius=4),
                    text=[f"{r[1]:.1f}%" for r in rates], textposition="outside", cliponaxis=False,
                    textfont=dict(family=MONO, size=15, color=PAPER))
        _layout(fig, 330, margin=dict(l=8, r=8, t=8, b=8), meta={"kind": "audit"},
                yaxis=_num_axis(range=[0, 100], ticksuffix="%",
                                title=dict(text="failure rate, training tasks", font=dict(size=12, color=SMOKE))),
                xaxis=dict(tickfont=dict(family=SANS, size=12.5, color=LABEL)))
        st.plotly_chart(fig, key="one_core", config=PLOT_CFG, theme=None)
        st.markdown('<div class="src">notebook 06, training split. Other users\' 1-core tasks fail <i>less</i> often '
                    'than everything else.</div>', unsafe_allow_html=True)

    st.write("")
    a, b = st.columns([1.45, 1], gap="large", vertical_alignment="center")
    with a:
        st.markdown(f'<div class="card figure"><img src="data:image/png;base64,{_figure_b64()}" alt="Two plan_cpu curves. '
                    'The one learned on the random split swings widely across CPU requests; the one learned on the '
                    'user-grouped split stays close to zero."></div>', unsafe_allow_html=True)
    with b:
        st.markdown(f'<div class="h3" style="font-size:1.45rem;line-height:1.25">Hold the test users out of training '
                    f'and the strongest curve <em>goes flat</em>.</div><div class="sub" style="margin-top:.6rem">The '
                    f'plan_cpu swing falls from <span class="mono" style="color:{PAPER}">{P["plan_cpu_swing"][0]:.2f}'
                    f'</span> to <span class="mono" style="color:{PAPER}">{P["plan_cpu_swing"][1]:.2f}</span> '
                    f'log-odds.</div>', unsafe_allow_html=True)
        st.markdown('<div class="sub" style="margin-top:.8rem">The readable model matched the black box on a random '
                    'split (AUC 0.752 against 0.756). Its explanations exposed a shortcut. A user-grouped test then '
                    'showed that none of the three models beats chance on users it has not seen. A random split hides '
                    'that for all three.</div><div class="src" style="margin-top:.7rem">paper §6.11, Figure 10, '
                    'recoloured for dark mode</div>', unsafe_allow_html=True)

st.markdown(f'<div class="src" style="margin-top:1.6rem;padding-top:1rem;border-top:1px solid {HAIR}">Canonical KAAM, '
            f'split 42, seed 0: test AUC {B["metrics"]["test_auc"]:.4f} on {B["metrics"]["test_tasks"]:,} held-out '
            'tasks. Data: Alibaba cluster-trace-gpu-v2020 (CC BY 4.0). No user or job identifiers are used or shown.'
            '</div>', unsafe_allow_html=True)
