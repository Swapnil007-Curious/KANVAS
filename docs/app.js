/* KANVAS demo page: the interface. The model's arithmetic is in kaam.js. */
(function () {
  "use strict";

  const NS = "http://www.w3.org/2000/svg";
  const MINUS = "−";
  const REQUEST = ["plan_cpu", "plan_mem", "plan_gpu", "inst_num"];
  const MACHINE = ["cap_gpu", "avg_cpu_usr", "avg_gpu_util", "avg_cpu_kernel", "avg_load_1", "avg_net_receive"];
  const TABS = ["predict", "curves", "audit"];
  const GROUPS = [
    ["tp", "Caught failures (failed, flagged)", "a caught failure"],
    ["fn", "Missed failures (failed, not flagged)", "a missed failure"],
    ["fp", "False alarms (finished normally, flagged)", "a false alarm"],
    ["tn", "Correct all-clears (finished normally, not flagged)", "a correct all-clear"],
  ];

  let M = null;
  let EXAMPLES = [];
  const S = { raw: {}, example: null, tab: null, shared: false };
  const form = {};
  const bars = {};
  const plots = {};
  const levelDigits = {};
  let sharedDomain = null;

  const $ = (id) => document.getElementById(id);

  function el(tag, attrs, text) {
    const e = document.createElement(tag);
    for (const k in attrs || {}) e.setAttribute(k, attrs[k]);
    if (text != null) e.textContent = text;
    return e;
  }

  function sv(tag, attrs) {
    const e = document.createElementNS(NS, tag);
    for (const k in attrs || {}) e.setAttribute(k, attrs[k]);
    return e;
  }

  // ------------------------------------------------------------------ formatting
  const num = (v, d) => v.toLocaleString("en-US", { maximumFractionDigits: d });
  const fixed = (v, d) => v.toLocaleString("en-US", { minimumFractionDigits: d, maximumFractionDigits: d });
  const pct = (p) => (p * 100).toFixed(1) + "%";

  function signed(v, d) {
    const s = Math.abs(v).toFixed(d == null ? 2 : d);
    if (Number(s) === 0) return s;
    return (v > 0 ? "+" : MINUS) + s;
  }

  function fmt(n, raw) {
    const m = M.meta[n];
    const v = raw / m.scale;
    if (n === "cap_gpu") return num(v, 0) + "-GPU machine";
    if (m.group === "request") {
      const unit = Math.abs(v - 1) < 1e-12 ? m.unit.replace(/s$/, "") : m.unit;
      return num(v, levelDigits[n] != null ? levelDigits[n] : m.digits) + " " + unit;
    }
    const s = fixed(v, m.digits);
    if (m.unit === "%") return s + "%";
    return m.unit ? s + " " + m.unit : s;
  }

  function axisUnit(n) {
    return M.meta[n].unit || "load average";
  }

  // fewest decimals that show every level exactly (at most 3), more if two levels would still print alike
  function computeLevelDigits() {
    for (const n in M.levels) {
      const vs = M.levels[n].x.map((x) => x / M.meta[n].scale);
      let d = 0;
      while (d < 3 && vs.some((v) => Math.abs(Number(v.toFixed(d)) - v) > 1e-9)) d++;
      while (d < 6 && new Set(vs.map((v) => v.toFixed(d))).size < vs.length) d++;
      levelDigits[n] = d;
    }
  }

  function ticks(lo, hi, count) {
    const span = hi - lo;
    if (!(span > 0)) return { values: [lo], digits: 0 };
    let step = Math.pow(10, Math.floor(Math.log10(span / count)));
    const err = span / count / step;
    if (err >= 7.07) step *= 10;
    else if (err >= 3.16) step *= 5;
    else if (err >= 1.41) step *= 2;
    const values = [];
    for (let t = Math.ceil(lo / step) * step; t <= hi + step * 1e-9; t += step) values.push(Number(t.toFixed(12)));
    return { values: values, digits: Math.max(0, -Math.floor(Math.log10(step) + 1e-9)) };
  }

  // ------------------------------------------------------------------ state changes
  function setRaw(n, raw, from) {
    S.raw[n] = raw;
    if (S.example) {
      S.example = null;
      $("example").value = "";
    }
    update(from === n ? n : null);
  }

  function loadRow(row, example) {
    S.raw = Object.assign({}, row);
    S.example = example || null;
    update(null);
  }

  function typical() {
    const row = {};
    for (const n of M.features) row[n] = M.train[n].median;
    $("example").value = "";
    loadRow(row, null);
  }

  function update(skipNumberOf) {
    const r = KAAM.predict(M, S.raw);
    syncForm(skipNumberOf);
    renderRisk(r);
    renderBars(r);
    $("curves-now").textContent = "Current failure probability: " + pct(r.risk) + ".";
    updateMarkers(r);
  }

  // ------------------------------------------------------------------ form
  function buildField(n, parent) {
    const m = M.meta[n];
    const id = "in-" + n;
    const noteId = "note-" + n;
    const wrap = el("div", { class: "field" });
    const label = el("label", { for: id });
    label.append(el("span", null, m.label));
    if (!M.levels[n]) label.append(el("span", { class: "unit" }, axisUnit(n)));
    wrap.append(label);
    const f = { note: el("p", { class: "note", id: noteId }) };
    if (M.levels[n]) {
      const L = M.levels[n];
      const total = L.count.reduce((a, b) => a + b, 0);
      const sel = el("select", { id: id, "aria-describedby": noteId });
      L.x.forEach((x, i) => {
        const share = (100 * L.count[i]) / total;
        sel.append(el("option", { value: String(i) },
          fmt(n, x) + " (" + (share < 0.1 ? "<0.1" : share.toFixed(1)) + "% of training)"));
      });
      sel.addEventListener("change", () => {
        if (sel.value !== "custom") setRaw(n, L.x[Number(sel.value)], null);
      });
      wrap.append(sel);
      f.kind = "select";
      f.sel = sel;
    } else {
      const lo = M.preprocess.clip[n][0] / m.scale;
      const hi = M.preprocess.clip[n][1] / m.scale;
      const box = el("div", { class: "slider" });
      const range = el("input", {
        type: "range", id: id + "-range", min: String(lo), max: String(hi), step: String((hi - lo) / 1000),
        "aria-label": m.label + ", slider over the training range", "aria-describedby": noteId,
      });
      const box2 = el("input", {
        type: "number", id: id, step: "any", min: "0", inputmode: "decimal", "aria-describedby": noteId,
      });
      range.addEventListener("input", () => setRaw(n, Number(range.value) * m.scale, null));
      box2.addEventListener("input", () => {
        const v = parseFloat(box2.value);
        if (isFinite(v) && v >= 0) setRaw(n, v * m.scale, n);
      });
      box2.addEventListener("change", () => update(null));
      box.append(range, box2);
      wrap.append(box);
      f.kind = "range";
      f.range = range;
      f.num = box2;
    }
    wrap.append(f.note);
    form[n] = f;
    parent.append(wrap);
  }

  function syncForm(skipNumberOf) {
    for (const n of M.features) {
      const f = form[n];
      const m = M.meta[n];
      const raw = S.raw[n];
      const [lo, hi] = M.preprocess.clip[n];
      const v = KAAM.clipped(M, n, raw);
      let note = "";
      if (raw < lo) note = "This task: " + fmt(n, raw) + ". The model raises it to " + fmt(n, lo) +
        ", the training 1st percentile.";
      else if (raw > hi) note = "This task: " + fmt(n, raw) + ". The model caps it at " + fmt(n, hi) +
        ", the training 99th percentile.";
      if (f.kind === "select") {
        const old = f.sel.querySelector('option[value="custom"]');
        if (old) old.remove();
        const j = KAAM.levelIndex(M.levels[n], v);
        if (j >= 0) {
          f.sel.value = String(j);
        } else {
          const opt = el("option", { value: "custom" }, fmt(n, v) + " (not seen in training)");
          const after = M.levels[n].x.findIndex((x) => x > v);
          f.sel.insertBefore(opt, after < 0 ? null : f.sel.options[after]);
          f.sel.value = "custom";
          note = (note ? note + " " : "") + "This value is not among the training values; its curve value is interpolated.";
        }
      } else {
        f.range.value = String(v / m.scale);
        if (skipNumberOf !== n) f.num.value = String(Number((raw / m.scale).toFixed(m.digits + 1)));
      }
      f.note.textContent = note;
    }
  }

  function buildExamples() {
    const sel = $("example");
    sel.append(el("option", { value: "" }, "Choose a real test task..."));
    const scored = EXAMPLES.map((e) => {
      const risk = KAAM.predict(M, e.x).risk;
      const flagged = risk >= M.model.threshold;
      return { e: e, risk: risk, g: e.label ? (flagged ? "tp" : "fn") : (flagged ? "fp" : "tn") };
    });
    for (const [g, title] of GROUPS) {
      const og = el("optgroup", { label: title });
      scored.filter((s) => s.g === g).sort((a, b) => b.risk - a.risk).forEach((s) => {
        og.append(el("option", { value: String(s.e.id) },
          "Task " + s.e.id + ": risk " + pct(s.risk) + (s.e.paper_case ? ", case audit in the paper" : "")));
      });
      sel.append(og);
    }
    sel.addEventListener("change", () => {
      if (!sel.value) return;
      const ex = EXAMPLES.find((e) => String(e.id) === sel.value);
      loadRow(ex.x, ex);
    });
  }

  // ------------------------------------------------------------------ prediction and bars
  function renderRisk(r) {
    const thr = M.model.threshold;
    const base = 1 / (1 + Math.exp(-M.baseline_logit));
    const flagged = r.risk >= thr;
    $("risk").textContent = pct(r.risk);
    $("mini-risk").textContent = pct(r.risk);
    $("risk-label").textContent = "Average training task: " + pct(base) + ". Threshold used in the paper: " +
      pct(thr) + ".";
    $("meter").style.width = (r.risk * 100).toFixed(2) + "%";
    const tb = $("tick-base"), tt = $("tick-thr");
    tb.style.left = (base * 100).toFixed(2) + "%";
    tb.firstChild.textContent = "average " + pct(base);
    tt.style.left = (thr * 100).toFixed(2) + "%";
    tt.firstChild.textContent = "threshold " + pct(thr);
    $("verdict").textContent = flagged
      ? "Flagged as likely to fail: the probability is at or above the threshold."
      : "Not flagged: the probability is below the threshold.";
    const truth = $("truth");
    if (S.example) {
      const e = S.example;
      const g = e.label ? (flagged ? "tp" : "fn") : (flagged ? "fp" : "tn");
      const what = GROUPS.find((x) => x[0] === g)[2];
      let text = "Task " + e.id + " is a real task from the test set. It " +
        (e.label ? "failed" : "finished normally") + ", so this is " + what + ".";
      if (e.paper_case) text += " It is the " + e.paper_case.toLowerCase() + " in the paper's four case audits.";
      truth.textContent = text;
    } else {
      truth.textContent = "";
    }
    const sum = r.logit - M.baseline_logit;
    $("equation").textContent = "log-odds " + signed(M.baseline_logit) + " (average task) " +
      (sum < 0 ? MINUS : "+") + " " + Math.abs(sum).toFixed(2) + " (sum of bars) = " + signed(r.logit) +
      ", probability " + pct(r.risk);
  }

  function buildBars() {
    const list = $("bars");
    for (const n of M.features) {
      const li = el("li", { class: "bar-row" });
      const lab = el("div", { class: "bar-label" });
      const name = el("button", { type: "button", class: "bar-name" }, M.meta[n].label);
      name.addEventListener("click", () => goToCurve(n));
      const value = el("span", { class: "bar-value" });
      lab.append(name, value);
      const track = el("div", { class: "bar-track", "aria-hidden": "true" });
      const fill = el("span", { class: "bar-fill" });
      track.append(fill);
      const numEl = el("span", { class: "bar-num" });
      const sr = el("span", { class: "visually-hidden" });
      li.append(lab, track, numEl, sr);
      list.append(li);
      bars[n] = { li: li, value: value, fill: fill, num: numEl, sr: sr };
    }
  }

  function renderBars(r) {
    const order = M.features.slice().sort((a, b) => Math.abs(r.contrib[b]) - Math.abs(r.contrib[a]));
    const maxAbs = Math.max.apply(null, order.map((n) => Math.abs(r.contrib[n])));
    const scale = Math.max(0.5, Math.ceil(maxAbs * 2) / 2);
    const list = $("bars");
    for (const n of order) {
      const b = bars[n];
      const c = r.contrib[n];
      const w = (Math.abs(c) / scale) * 50;
      b.value.textContent = fmt(n, KAAM.clipped(M, n, S.raw[n]));
      b.fill.className = "bar-fill " + (c >= 0 ? "raise" : "lower");
      b.fill.style.left = (c >= 0 ? 50 : 50 - w).toFixed(2) + "%";
      b.fill.style.width = w.toFixed(2) + "%";
      b.num.textContent = signed(c);
      b.sr.textContent = c >= 0 ? " raises the risk" : " lowers the risk";
      list.append(b.li);
    }
  }

  // ------------------------------------------------------------------ tabs
  function selectTab(name, focus) {
    if (TABS.indexOf(name) < 0) name = "predict";
    S.tab = name;
    for (const t of TABS) {
      const on = t === name;
      const tab = $("tab-" + t);
      tab.setAttribute("aria-selected", on ? "true" : "false");
      tab.tabIndex = on ? 0 : -1;
      $("panel-" + t).hidden = !on;
    }
    document.body.classList.toggle("on-predict", name === "predict");
    if (focus) $("tab-" + name).focus();
    if (location.hash !== "#" + name) history.replaceState(null, "", "#" + name);
    if (name === "curves") drawCurves();
  }

  function initTabs() {
    for (const t of TABS) $("tab-" + t).addEventListener("click", () => selectTab(t));
    document.querySelector('[role="tablist"]').addEventListener("keydown", (ev) => {
      const i = TABS.indexOf(S.tab);
      let j = null;
      if (ev.key === "ArrowRight") j = (i + 1) % TABS.length;
      else if (ev.key === "ArrowLeft") j = (i + TABS.length - 1) % TABS.length;
      else if (ev.key === "Home") j = 0;
      else if (ev.key === "End") j = TABS.length - 1;
      if (j === null) return;
      ev.preventDefault();
      selectTab(TABS[j], true);
    });
    window.addEventListener("hashchange", () => {
      const h = location.hash.slice(1);
      if (TABS.indexOf(h) >= 0 && h !== S.tab) selectTab(h);
    });
  }

  function goToCurve(n) {
    selectTab("curves");
    const w = plots[n].wrap;
    w.focus({ preventScroll: true });
    w.scrollIntoView({ block: "center", behavior: matchMedia("(prefers-reduced-motion: reduce)").matches ? "auto" : "smooth" });
  }

  // ------------------------------------------------------------------ curves
  function buildCurves() {
    const grid = $("curves");
    for (const n of REQUEST.concat(MACHINE)) {
      const card = el("article", { class: "card curve-card" });
      const head = el("header");
      head.append(el("h3", { id: "ct-" + n }, M.meta[n].label), el("span", { class: "muted" }, axisUnit(n)));
      const readout = el("p", { class: "readout" });
      const wrap = el("div", { class: "plot", tabindex: "0", role: "slider", "aria-labelledby": "ct-" + n });
      card.append(head, readout, wrap);
      grid.append(card);
      plots[n] = { readout: readout, wrap: wrap, built: false, hovering: false, dragging: false };
      attachPlotEvents(n);
    }
    let queued = false;
    new ResizeObserver(() => {
      if (queued) return;
      queued = true;
      requestAnimationFrame(() => {
        queued = false;
        if (S.tab === "curves") drawCurves();
      });
    }).observe(grid);
    $("shared").addEventListener("change", (ev) => {
      S.shared = ev.target.checked;
      drawCurves();
    });
  }

  function centred(n) {
    const mu = M.phi_mean[n];
    return M.curves[n].phi.map((p) => p - mu);
  }

  function drawCurves() {
    if (!M || S.tab !== "curves") return;
    let lo = 0, hi = 0;
    for (const n of M.features) {
      const ys = centred(n);
      lo = Math.min(lo, Math.min.apply(null, ys));
      hi = Math.max(hi, Math.max.apply(null, ys));
    }
    sharedDomain = [lo, hi];
    for (const n of M.features) drawPlot(n);
    updateMarkers(KAAM.predict(M, S.raw));
  }

  function drawPlot(n) {
    const P = plots[n];
    const W = Math.max(220, Math.round(P.wrap.clientWidth));
    const H = 180;
    const pad = { l: 40, r: 10, t: 8, b: 42 };
    const m = M.meta[n];
    const sc = m.scale;
    const [clo, chi] = M.preprocess.clip[n];
    const x0 = clo / sc, x1 = chi / sc;
    const xs = M.curves[n].x.map((x) => x / sc);
    const ys = centred(n);
    let y0, y1;
    if (S.shared) [y0, y1] = sharedDomain;
    else {
      y0 = Math.min(0, Math.min.apply(null, ys));
      y1 = Math.max(0, Math.max.apply(null, ys));
    }
    const py = (y1 - y0) * 0.08 || 0.1;
    y0 -= py;
    y1 += py;
    const bottom = H - pad.b;
    const X = (v) => pad.l + ((v - x0) / (x1 - x0)) * (W - pad.l - pad.r);
    const Y = (v) => pad.t + ((y1 - v) / (y1 - y0)) * (bottom - pad.t);
    const s = sv("svg", { viewBox: "0 0 " + W + " " + H, width: String(W), height: String(H), "aria-hidden": "true",
      focusable: "false" });

    // thin-data edges: outside the training 5th to 95th percentiles
    const t = M.train[n];
    const p5 = X(t.p5 / sc), p95 = X(t.p95 / sc);
    if (p5 > pad.l + 0.5) s.append(sv("rect", { class: "thin", x: pad.l, y: pad.t, width: p5 - pad.l, height: bottom - pad.t }));
    if (p95 < W - pad.r - 0.5) s.append(sv("rect", { class: "thin", x: p95, y: pad.t, width: W - pad.r - p95, height: bottom - pad.t }));

    // y grid and labels
    const yt = ticks(y0, y1, 4);
    for (const v of yt.values) {
      s.append(sv("line", { class: Math.abs(v) < 1e-12 ? "zero" : "gridline", x1: pad.l, x2: W - pad.r, y1: Y(v), y2: Y(v) }));
      const lab = sv("text", { x: pad.l - 6, y: Y(v) + 3.5, "text-anchor": "end" });
      lab.textContent = Math.abs(v) < 1e-12 ? "0" : signed(v, yt.digits);
      s.append(lab);
    }
    if (yt.values.every((v) => Math.abs(v) > 1e-12)) {
      s.append(sv("line", { class: "zero", x1: pad.l, x2: W - pad.r, y1: Y(0), y2: Y(0) }));
    }

    // area between the curve and zero: red above (raises risk), blue below (lowers it)
    const pts = xs.map((x, i) => X(x).toFixed(1) + "," + Y(ys[i]).toFixed(1));
    const line = "M" + pts.join("L");
    const area = line + "L" + X(x1).toFixed(1) + "," + Y(0).toFixed(1) + "L" + X(x0).toFixed(1) + "," + Y(0).toFixed(1) + "Z";
    const defs = sv("defs");
    const cpHi = sv("clipPath", { id: "cp-hi-" + n });
    cpHi.append(sv("rect", { x: 0, y: 0, width: W, height: Math.max(0, Y(0)) }));
    const cpLo = sv("clipPath", { id: "cp-lo-" + n });
    cpLo.append(sv("rect", { x: 0, y: Y(0), width: W, height: Math.max(0, H - Y(0)) }));
    defs.append(cpHi, cpLo);
    s.append(defs);
    s.append(sv("path", { class: "area-raise", d: area, "clip-path": "url(#cp-hi-" + n + ")" }));
    s.append(sv("path", { class: "area-lower", d: area, "clip-path": "url(#cp-lo-" + n + ")" }));
    s.append(sv("path", { class: "line", d: line }));

    // discrete inputs: the values seen in training
    const L = M.levels[n];
    if (L && L.x.length <= 50) {
      L.x.forEach((x, i) => {
        s.append(sv("circle", { class: "level", cx: X(x / sc), cy: Y(L.phi[i] - M.phi_mean[n]), r: 2.5 }));
      });
    }

    // where training tasks sit
    const stripTop = bottom + 5, stripH = 11;
    if (L) {
      const maxC = Math.max.apply(null, L.count);
      L.x.forEach((x, i) => {
        const h = Math.max(1, (L.count[i] / maxC) * stripH);
        s.append(sv("rect", { class: "density", x: X(x / sc) - 1.5, y: stripTop + stripH - h, width: 3, height: h }));
      });
    } else if (t.hist) {
      const c = t.hist.counts, maxC = Math.max.apply(null, c);
      const bw = (W - pad.l - pad.r) / c.length;
      c.forEach((v, i) => {
        if (!v) return;
        const h = Math.max(1, (v / maxC) * stripH);
        s.append(sv("rect", { class: "density", x: pad.l + i * bw + 0.5, y: stripTop + stripH - h, width: Math.max(1, bw - 1), height: h }));
      });
    }

    // x labels
    const xt = ticks(x0, x1, Math.max(3, Math.floor((W - pad.l - pad.r) / 75)));
    for (const v of xt.values) {
      const x = X(v);
      const anchor = x < pad.l + 12 ? "start" : x > W - pad.r - 12 ? "end" : "middle";
      const lab = sv("text", { x: x, y: H - 6, "text-anchor": anchor });
      lab.textContent = num(v, xt.digits);
      s.append(lab);
    }

    // current input and hover cursor
    const cur = sv("line", { class: "cursor-line", y1: pad.t, y2: bottom });
    const marker = sv("circle", { class: "marker", r: 5.5 });
    const hl = sv("line", { class: "hover-line", y1: pad.t, y2: bottom, visibility: "hidden" });
    const hd = sv("circle", { class: "hover-dot", r: 3.5, visibility: "hidden" });
    s.append(cur, hl, hd, marker);

    P.wrap.replaceChildren(s);
    Object.assign(P, { built: true, X: X, Y: Y, x0: x0, x1: x1, pad: pad, W: W, cur: cur, marker: marker, hl: hl, hd: hd });
  }

  function updateMarkers(r) {
    for (const n of M.features) {
      const P = plots[n];
      if (!P) continue;
      const sc = M.meta[n].scale;
      const v = KAAM.clipped(M, n, S.raw[n]);
      const c = r.contrib[n];
      const text = fmt(n, v) + ", contribution " + signed(c);
      P.wrap.setAttribute("aria-valuemin", String(P.x0 != null ? P.x0 : M.preprocess.clip[n][0] / sc));
      P.wrap.setAttribute("aria-valuemax", String(P.x1 != null ? P.x1 : M.preprocess.clip[n][1] / sc));
      P.wrap.setAttribute("aria-valuenow", String(v / sc));
      P.wrap.setAttribute("aria-valuetext", text);
      if (!P.hovering) {
        P.readout.textContent = "Current: " + fmt(n, v) + " → " + signed(c) +
          (v !== S.raw[n] ? " (clipped from " + fmt(n, S.raw[n]) + ")" : "");
      }
      if (!P.built) continue;
      const x = P.X(v / sc), y = P.Y(c);
      P.cur.setAttribute("x1", x);
      P.cur.setAttribute("x2", x);
      P.marker.setAttribute("cx", x);
      P.marker.setAttribute("cy", y);
      P.marker.setAttribute("class", "marker " + (c >= 0 ? "raise" : "lower"));
    }
  }

  function nearestLevel(n, raw) {
    const xs = M.levels[n].x;
    let best = 0;
    for (let i = 1; i < xs.length; i++) if (Math.abs(xs[i] - raw) < Math.abs(xs[best] - raw)) best = i;
    return best;
  }

  function rawAtPointer(n, ev) {
    const P = plots[n];
    const rect = P.wrap.getBoundingClientRect();
    const px = ((ev.clientX - rect.left) / rect.width) * P.W;
    const f = (px - P.pad.l) / (P.W - P.pad.l - P.pad.r);
    const raw = (P.x0 + Math.min(1, Math.max(0, f)) * (P.x1 - P.x0)) * M.meta[n].scale;
    return M.levels[n] ? M.levels[n].x[nearestLevel(n, raw)] : raw;
  }

  function attachPlotEvents(n) {
    const P = plots[n];
    const w = P.wrap;
    w.addEventListener("pointerdown", (ev) => {
      if (!P.built || ev.button !== 0) return;
      P.dragging = true;
      w.setPointerCapture(ev.pointerId);
      w.focus({ preventScroll: true });
      setRaw(n, rawAtPointer(n, ev), null);
    });
    w.addEventListener("pointermove", (ev) => {
      if (!P.built) return;
      const raw = rawAtPointer(n, ev);
      if (P.dragging) {
        setRaw(n, raw, null);
        return;
      }
      if (ev.pointerType !== "mouse") return;
      const sc = M.meta[n].scale;
      const c = KAAM.phi(M, n, raw) - M.phi_mean[n];
      const x = P.X(raw / sc), y = P.Y(c);
      P.hovering = true;
      P.hl.setAttribute("x1", x);
      P.hl.setAttribute("x2", x);
      P.hl.setAttribute("visibility", "visible");
      P.hd.setAttribute("cx", x);
      P.hd.setAttribute("cy", y);
      P.hd.setAttribute("visibility", "visible");
      P.readout.textContent = "At " + fmt(n, raw) + " → " + signed(c);
    });
    const end = (ev) => {
      if (P.dragging && w.hasPointerCapture(ev.pointerId)) w.releasePointerCapture(ev.pointerId);
      P.dragging = false;
    };
    w.addEventListener("pointerup", end);
    w.addEventListener("pointercancel", end);
    w.addEventListener("pointerleave", () => {
      if (!P.built) return;
      P.hovering = false;
      P.hl.setAttribute("visibility", "hidden");
      P.hd.setAttribute("visibility", "hidden");
      updateMarkers(KAAM.predict(M, S.raw));
    });
    w.addEventListener("keydown", (ev) => {
      const [lo, hi] = M.preprocess.clip[n];
      const v = KAAM.clipped(M, n, S.raw[n]);
      const big = ev.key === "PageUp" || ev.key === "PageDown";
      const dir = { ArrowRight: 1, ArrowUp: 1, PageUp: 1, ArrowLeft: -1, ArrowDown: -1, PageDown: -1 }[ev.key];
      let raw = null;
      if (M.levels[n]) {
        const xs = M.levels[n].x;
        const i = nearestLevel(n, v);
        if (dir) raw = xs[Math.min(xs.length - 1, Math.max(0, i + dir * (big ? 5 : 1)))];
        else if (ev.key === "Home") raw = xs[0];
        else if (ev.key === "End") raw = xs[xs.length - 1];
      } else {
        const step = (hi - lo) / 100;
        if (dir) raw = Math.min(hi, Math.max(lo, v + dir * step * (big ? 10 : 1)));
        else if (ev.key === "Home") raw = lo;
        else if (ev.key === "End") raw = hi;
      }
      if (raw === null) return;
      ev.preventDefault();
      setRaw(n, raw, null);
    });
  }

  // ------------------------------------------------------------------ audit
  function buildAudit() {
    const P = M.paper;
    const table = el("table");
    table.append(el("caption", null, "Test ROC-AUC on the full trace (539,816 tasks)"));
    const thead = el("thead");
    const hr = el("tr");
    hr.append(el("th", { scope: "col" }, "Model"));
    for (const c of P.columns) hr.append(el("th", { scope: "col" }, c));
    thead.append(hr);
    const tbody = el("tbody");
    for (const row of P.rows) {
      const tr = el("tr", row.model === "KAAM" ? { class: "this-model" } : null);
      tr.append(el("th", { scope: "row" }, row.model));
      for (const c of row.cells) {
        const td = el("td", null, c.auc.toFixed(3));
        td.append(el("span", { class: "ci" }, c.lo.toFixed(3) + " to " + c.hi.toFixed(3)));
        tr.append(td);
      }
      tbody.append(tr);
    }
    table.append(thead, tbody);
    $("auc-table").append(table);

    const kaamFolds = P.grouped_folds.rows.find((r) => r.model === "KAAM").auc;
    const notes = $("audit-notes");
    const lines = [
      "Each cell is the mean over test sets of the mean over five seeds, with its 95% bootstrap interval below it " +
        "(users are resampled under the user-grouped split). 0.5 is chance. The time split trains on the earliest " +
        "70% of tasks and tests on the last 15%; known and new users are those with and without tasks in training. " +
        "The last two columns remove User 1 (the user with the most tasks), or Users 1 and 4, and then split at random.",
      "This page runs one of the KAAM fits behind the Random column: random split 42, seed 0, test ROC-AUC " +
        M.model.test_auc.toFixed(3) + " on " + M.model.test_tasks.toLocaleString("en-US") + " tasks. On the five " +
        "user-grouped folds below, KAAM ranges from " + Math.min.apply(null, kaamFolds).toFixed(2) + " to " +
        Math.max.apply(null, kaamFolds).toFixed(2) + ".",
    ];
    for (const t of lines) notes.append(el("p", null, t));
    const src = el("p");
    src.append("Source: ");
    const a = el("a", { href: "https://github.com/Swapnil007-Curious/KANVAS/blob/main/results/analysis/full/model_ci.csv" },
      "results/analysis/full/model_ci.csv");
    src.append(a, " and fold_auc.csv, as in the paper.");
    notes.append(src);

    const ft = el("table");
    ft.append(el("caption", null, "User-grouped split: test ROC-AUC on each fold"));
    const fh = el("thead");
    const fr = el("tr");
    fr.append(el("th", { scope: "col" }, "Model"));
    P.grouped_folds.folds.forEach((f, i) => {
      const th = el("th", { scope: "col" }, "Fold " + (i + 1));
      th.append(el("span", { class: "ci" }, f.test_users.toLocaleString("en-US") + " users, " +
        f.test_tasks.toLocaleString("en-US") + " tasks"));
      fr.append(th);
    });
    fh.append(fr);
    const fb = el("tbody");
    for (const row of P.grouped_folds.rows) {
      const tr = el("tr", row.model === "KAAM" ? { class: "this-model" } : null);
      tr.append(el("th", { scope: "row" }, row.model));
      for (const v of row.auc) tr.append(el("td", null, v.toFixed(3)));
      fb.append(tr);
    }
    ft.append(fh, fb);
    $("fold-table").append(ft);
  }

  // ------------------------------------------------------------------ theme
  function initTheme() {
    const btn = $("theme");
    const order = ["auto", "light", "dark"];
    let theme = "auto";
    try {
      theme = localStorage.getItem("kanvas-theme") || "auto";
    } catch (e) { /* storage blocked */ }
    if (order.indexOf(theme) < 0) theme = "auto";
    const apply = () => {
      if (theme === "auto") document.documentElement.removeAttribute("data-theme");
      else document.documentElement.setAttribute("data-theme", theme);
      btn.textContent = "Theme: " + theme;
    };
    btn.addEventListener("click", () => {
      theme = order[(order.indexOf(theme) + 1) % order.length];
      try {
        localStorage.setItem("kanvas-theme", theme);
      } catch (e) { /* storage blocked */ }
      apply();
    });
    apply();
  }

  // ------------------------------------------------------------------ start
  function json(url) {
    return fetch(url).then((r) => {
      if (!r.ok) throw new Error(url + ": HTTP " + r.status);
      return r.json();
    });
  }

  initTheme();
  Promise.all([json("model.json"), json("examples.json")]).then(([model, ex]) => {
    M = model;
    EXAMPLES = ex.examples;
    computeLevelDigits();
    document.querySelectorAll('[data-fill="train_tasks"]').forEach((e) => {
      e.textContent = M.model.train_tasks.toLocaleString("en-US");
    });
    for (const n of REQUEST) buildField(n, $("fs-request"));
    for (const n of MACHINE) buildField(n, $("fs-machine"));
    buildExamples();
    $("typical").addEventListener("click", typical);
    $("form").addEventListener("submit", (ev) => ev.preventDefault());
    $("mini-show").addEventListener("click", () => $("h-risk").scrollIntoView({ block: "start" }));
    new IntersectionObserver((entries) => {
      document.body.classList.toggle("result-visible", entries[0].isIntersecting);
    }).observe($("h-risk").parentElement);
    buildBars();
    buildCurves();
    buildAudit();
    initTabs();
    typical();
    $("status").remove();
    selectTab(location.hash.slice(1));
  }).catch((err) => {
    const s = $("status");
    s.className = "error";
    s.textContent = "Could not load the model (" + err.message + "). If you opened this file from disk, serve the " +
      "folder instead, for example with: python -m http.server -d docs";
  });
})();
