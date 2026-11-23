/* KANVAS motion engine. Installed once per page load; Streamlit reruns only replace DOM nodes, which this
 * watches. Two rules from fluid-interface design hold throughout:
 *  - a value always animates FROM what is on screen now (a spring that carries its velocity), so a new input
 *    mid-animation redirects the motion instead of restarting it;
 *  - every animation says something: a number changed, a bar moved, a curve was learned.
 * Elements opt in with data attributes written by app.py:
 *   data-kv="count"  data-key data-to data-dec data-sign data-group data-suffix [data-reveal]
 *   data-kv="style"  data-key data-to data-prop data-unit
 * Plotly charts opt in with layout.meta.kind = "wf" | "curve".
 */
(() => {
  if (window.__kv) return;
  const KV = (window.__kv = { vals: {}, plots: {} });
  const reduce = window.matchMedia("(prefers-reduced-motion: reduce)");
  const EASE = "cubic-bezier(.2,.85,.25,1)";
  const LABELS = __LABELS__;

  // ------------------------------------------------------------- springs (critically damped)
  const RESPONSE = 0.42, W = (2 * Math.PI) / RESPONSE;
  function spring(key, to, render) {
    const s = KV.vals[key] || (KV.vals[key] = { x: to, v: 0 });
    if (s.raf) cancelAnimationFrame(s.raf);
    if (reduce.matches || !isFinite(to)) { s.x = to; s.v = 0; render(to); return; }
    // the clock starts on the first painted frame: a rerun redraws charts and can block the page for a moment,
    // and a spring that ran out during that block would look like a jump
    const c1 = s.x - to, c2 = s.v + W * c1;
    let t0 = 0;
    const step = (now) => {
      if (!t0) t0 = now;
      const t = (now - t0) / 1000, e = Math.exp(-W * t);
      if (t > 1.7 * RESPONSE) { s.x = to; s.v = 0; s.raf = 0; render(to); return; }
      s.x = to + (c1 + c2 * t) * e;
      s.v = (c2 - W * (c1 + c2 * t)) * e;
      render(s.x);
      s.raf = requestAnimationFrame(step);
    };
    render(s.x);  // before first paint: the new node shows the old value, never a flash of the final one
    s.raf = requestAnimationFrame(step);
  }

  function fmt(x, d) {
    const dec = +d.dec || 0;
    const body = Math.abs(x).toLocaleString("en-US", {
      minimumFractionDigits: dec, maximumFractionDigits: dec, useGrouping: d.group === "1" });
    const zero = +body.replace(/,/g, "") === 0;
    return (x < 0 && !zero ? "−" : d.sign === "1" ? "+" : "") + body + (d.suffix || "");
  }

  // Streamlit renders markdown through React, which patches these nodes in place on a rerun: the text is written
  // through the text node React owns (replacing it would orphan React's reference).
  function setText(el, s) {
    const t = el.firstChild;
    if (t && t.nodeType === 3 && !t.nextSibling) { if (t.nodeValue !== s) t.nodeValue = s; } else el.textContent = s;
  }

  const seen = new WeakSet();
  const io = new IntersectionObserver((entries) => {
    for (const en of entries) {
      if (!en.isIntersecting) continue;
      const el = en.target, d = el.dataset;
      KV.vals[d.key] = { x: 0, v: 0 };  // revealed numbers count up from zero each time they come into view
      spring(d.key, +d.to, (x) => setText(el, fmt(x, d)));
    }
  });

  function animate(el) {
    const d = el.dataset;
    if (!(d.key in KV.vals)) KV.vals[d.key] = { x: 0, v: 0 };
    if (d.kv === "count") spring(d.key, +d.to, (x) => setText(el, fmt(x, d)));
    else if (d.kv === "style") { const unit = d.unit || ""; spring(d.key, +d.to, (x) => { el.style[d.prop] = x + unit; }); }
  }

  function bind(el) {
    if (seen.has(el)) return;
    seen.add(el);
    if (el.dataset.reveal === "1") io.observe(el); else animate(el);
  }

  // ------------------------------------------------------------- plotly charts
  const rel = (el, base) => {
    const r = el.getBoundingClientRect();
    return { x: r.left - base.left, y: r.top - base.top, w: r.width, h: r.height };
  };
  const fade = (el, delay, dur = 240) => el && el.animate(
    [{ opacity: 0 }, { opacity: 1 }], { duration: dur, delay, easing: "ease-out", fill: "backwards" });

  // FLIP: the element is already at its new place; start it at the old one and let it settle.
  function flip(el, from, to, scale, delay = 0) {
    if (!el || !from || !to) return;
    const dx = from.x - to.x, dy = from.y - to.y;
    let sx = 1, sy = 1;
    if (scale && to.w > 0.5 && to.h > 0.5) { sx = from.w / to.w; sy = from.h / to.h; }
    if (Math.abs(dx) < 0.5 && Math.abs(dy) < 0.5 && Math.abs(sx - 1) < 0.004 && Math.abs(sy - 1) < 0.004) return;
    el.style.transformBox = "fill-box";
    el.style.transformOrigin = "0 0";
    el.animate([{ transform: `translate(${dx}px, ${dy}px) scale(${sx}, ${sy})` }, { transform: "none" }],
               { duration: 560, delay, easing: EASE, fill: "backwards" });
  }

  function tickKey(t) {
    const s = (t.textContent || "").trim();
    return LABELS.find((l) => s.startsWith(l));
  }

  function waterfall(gd) {
    const fd = gd._fullData || [];
    const bi = fd.findIndex((t) => t.type === "bar");
    if (bi < 0) return;
    const keys = fd[bi].customdata || [], xs = fd[bi].x || [];
    const base = gd.getBoundingClientRect();
    const points = [...gd.querySelectorAll(".barlayer .trace.bars .point")];
    const ticks = [...gd.querySelectorAll(".ytick")];
    const now = { bars: {}, ticks: {} };
    points.forEach((g, i) => { const p = g.querySelector("path"); if (p) now.bars[keys[i]] = rel(p, base); });
    ticks.forEach((g) => { const k = tickKey(g); if (k) now.ticks[k] = rel(g, base); });
    const prev = KV.plots.wf;
    const shapes = [...gd.querySelectorAll(".shapelayer path")];
    const marks = [...gd.querySelectorAll(".scatterlayer .trace .points path.point")];
    const marksTxt = [...gd.querySelectorAll(".scatterlayer .trace .textpoint")];
    const labels = points.map((g) => g.querySelector("text"));
    if (!reduce.matches) {
      if (prev) {  // an update: bars glide to their new size and row, then labels and connectors arrive
        points.forEach((g, i) => flip(g.querySelector("path"), prev.bars[keys[i]], now.bars[keys[i]], true));
        ticks.forEach((g) => { const k = tickKey(g); flip(g, prev.ticks[k], now.ticks[k], false); });
        labels.forEach((t) => fade(t, 380));
        shapes.forEach((s) => fade(s, 380));
        fade(marks[1], 380); fade(marksTxt[1], 380);
      } else {     // first draw: each bar grows out of the point where it starts, in sequence
        points.forEach((g, i) => {
          const p = g.querySelector("path"); if (!p) return;
          p.style.transformBox = "fill-box";
          p.style.transformOrigin = (+xs[i] >= 0 ? "0%" : "100%") + " 50%";
          p.animate([{ transform: "scaleX(0)" }, { transform: "none" }],
                    { duration: 620, delay: 120 + 55 * i, easing: EASE, fill: "backwards" });
          fade(labels[i], 120 + 55 * i + 380);
        });
        shapes.forEach((s) => fade(s, 700, 400));
        fade(marks[1], 800); fade(marksTxt[1], 800);
      }
    }
    KV.plots.wf = now;
    // remeasure once everything has settled (automargin can nudge the layout after the first draw)
    setTimeout(() => {
      if (!gd.isConnected) return;
      const b = gd.getBoundingClientRect(), s = { bars: {}, ticks: {} };
      [...gd.querySelectorAll(".barlayer .trace.bars .point")].forEach((g, i) => {
        const p = g.querySelector("path"); if (p) s.bars[keys[i]] = rel(p, b); });
      [...gd.querySelectorAll(".ytick")].forEach((g) => { const k = tickKey(g); if (k) s.ticks[k] = rel(g, b); });
      if (KV.plots.wf === now) KV.plots.wf = s;
    }, prev ? 720 : 1300);
  }

  function curve(gd) {
    const meta = gd.layout.meta || {};
    const traces = [...gd.querySelectorAll(".scatterlayer .trace")];
    if (traces.length < 2) return;
    const halo = traces[traces.length - 2].querySelector(".points");
    const dot = traces[traces.length - 1].querySelector(".points");
    const txt = traces[traces.length - 1].querySelector(".text");
    const base = gd.getBoundingClientRect();
    const now = { feature: meta.feature, dot: dot && rel(dot, base), txt: txt && rel(txt, base) };
    const prev = KV.plots.curve;
    if (!reduce.matches) {
      if (prev && prev.feature === meta.feature) {  // same curve, new value: the dot travels, then lands
        flip(halo, prev.dot, now.dot, false); flip(dot, prev.dot, now.dot, false); flip(txt, prev.txt, now.txt, false);
        const moved = prev.dot && now.dot && Math.hypot(prev.dot.x - now.dot.x, prev.dot.y - now.dot.y) > 1;
        if (moved && halo) {
          halo.style.transformBox = "fill-box"; halo.style.transformOrigin = "50% 50%";
          halo.animate([{ transform: "scale(1)", opacity: 1 }, { transform: "scale(1.9)", opacity: 0.15 },
                        { transform: "scale(1)", opacity: 1 }], { duration: 620, delay: 470, easing: "ease-out" });
        }
      } else {  // a curve appears: draw the learned function left to right, then place the dot on it
        traces.slice(0, -2).forEach((t) => t.querySelectorAll(".js-line").forEach((p) => {
          if ((p.style.strokeDasharray || "none") !== "none") return;  // dotted guide: leave as drawn
          const L = p.getTotalLength(); if (!L) return;
          p.animate([{ strokeDasharray: `${L} ${L}`, strokeDashoffset: L },
                     { strokeDasharray: `${L} ${L}`, strokeDashoffset: 0 }],
                    { duration: 900, easing: "cubic-bezier(.45,0,.15,1)" });
        }));
        [halo, dot].forEach((g) => {
          if (!g) return;
          g.style.transformBox = "fill-box"; g.style.transformOrigin = "50% 50%";
          g.animate([{ transform: "scale(.2)", opacity: 0 }, { transform: "none", opacity: 1 }],
                    { duration: 420, delay: 760, easing: EASE, fill: "backwards" });
        });
        fade(txt, 900);
      }
    }
    KV.plots.curve = now;
  }

  const done = new WeakSet();
  function tryPlot(gd) {
    if (done.has(gd) || !gd._fullLayout || !gd.layout || !gd.querySelector(".main-svg")) return;
    const kind = (gd.layout.meta || {}).kind;
    if (kind === "wf" && !gd.querySelector(".barlayer .point path")) return;
    if (kind === "curve" && !gd.querySelector(".scatterlayer .trace")) return;
    done.add(gd);
    try { if (kind === "wf") waterfall(gd); else if (kind === "curve") curve(gd); } catch (e) { /* never break the app */ }
  }

  // ------------------------------------------------------------- wiring
  function scan(root) {
    if (root.nodeType !== 1) return;
    if (root.matches("[data-kv]")) bind(root);
    root.querySelectorAll("[data-kv]").forEach(bind);
    const gd = root.closest(".js-plotly-plot");
    if (gd) tryPlot(gd);
    root.querySelectorAll(".js-plotly-plot").forEach(tryPlot);
  }
  new MutationObserver((muts) => {
    for (const m of muts) {
      if (m.type === "attributes") {  // React patched a value in place: spring from what is on screen now
        const el = m.target;
        if (!seen.has(el)) bind(el); else if (el.dataset.reveal !== "1") animate(el);
      } else m.addedNodes.forEach(scan);
    }
  }).observe(document.body, { childList: true, subtree: true, attributes: true, attributeFilter: ["data-to"] });
  scan(document.body);
})();
