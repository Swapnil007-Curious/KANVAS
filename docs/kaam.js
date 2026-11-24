/* KAAM in the browser: the canonical model's arithmetic on the exported curves (model.json).
 *
 *   logit = bias + sum_i phi_i(clip(x_i)),   risk = 1 / (1 + exp(-logit))
 *
 * phi_i is the exact curve value when the clipped input is one of the feature's training levels, otherwise
 * piecewise-linear interpolation on the 200-point grid. scripts/export_web.py (web_phi) does the same in numpy,
 * and tests/test_web_parity.py checks this file against PyTorch with node.
 *
 * Works as a classic browser script (window.KAAM) and as a CommonJS module (require).
 */
(function (root, factory) {
  if (typeof module === "object" && module.exports) module.exports = factory();
  else root.KAAM = factory();
})(typeof self !== "undefined" ? self : this, function () {
  "use strict";

  function clip(v, lo, hi) {
    return v < lo ? lo : v > hi ? hi : v;
  }

  function sigmoid(z) {
    return 1 / (1 + Math.exp(-z));
  }

  /* largest i with xs[i] <= v (xs ascending), or -1 */
  function floorIndex(xs, v) {
    let lo = 0, hi = xs.length - 1;
    if (v < xs[0]) return -1;
    while (lo < hi) {
      const mid = (lo + hi + 1) >> 1;
      if (xs[mid] <= v) lo = mid;
      else hi = mid - 1;
    }
    return lo;
  }

  function interp(v, xs, ys) {
    const n = xs.length;
    if (v <= xs[0]) return ys[0];
    if (v >= xs[n - 1]) return ys[n - 1];
    const i = Math.min(floorIndex(xs, v), n - 2);
    const t = (v - xs[i]) / (xs[i + 1] - xs[i]);
    return ys[i] + t * (ys[i + 1] - ys[i]);
  }

  /* index of the level equal to v (relative tolerance 1e-9), or -1 */
  function levelIndex(level, v) {
    const xs = level.x;
    // first index with xs[j] >= v, clamped to the last level
    let lo = 0, hi = xs.length - 1;
    while (lo < hi) {
      const mid = (lo + hi) >> 1;
      if (xs[mid] < v) lo = mid + 1;
      else hi = mid;
    }
    return Math.abs(xs[lo] - v) <= 1e-9 * Math.max(1, Math.abs(v)) ? lo : -1;
  }

  function clipped(model, name, raw) {
    const b = model.preprocess.clip[name];
    return clip(raw, b[0], b[1]);
  }

  function phi(model, name, raw) {
    const v = clipped(model, name, raw);
    const level = model.levels[name];
    if (level) {
      const j = levelIndex(level, v);
      if (j >= 0) return level.phi[j];
    }
    const c = model.curves[name];
    return interp(v, c.x, c.phi);
  }

  /* row: {feature: raw value}. Contributions are measured against the average training task, so
   * baseline_logit + sum(contrib) == logit. */
  function predict(model, row) {
    let logit = model.bias;
    const phis = {}, contrib = {};
    for (const n of model.features) {
      const p = phi(model, n, row[n]);
      phis[n] = p;
      contrib[n] = p - model.phi_mean[n];
      logit += p;
    }
    return { logit: logit, risk: sigmoid(logit), phi: phis, contrib: contrib };
  }

  return { clip, sigmoid, interp, levelIndex, clipped, phi, predict };
});
