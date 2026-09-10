// ============================================================
// Biostatistics calculators — Sample Size / Power / Descriptive
// Client-side only. Uses standard normal (z) & Student's t
// approximations suitable for planning and teaching purposes.
// ============================================================

/* ---------------------- Math helpers ---------------------- */

// Lookup tables for the fixed <select> options used in the UI.
// Two-sided critical z for a given alpha (also usable for confidence
// level via alpha = 1 - confidence).
var Z_TWO_SIDED = { "0.10": 1.6449, "0.05": 1.9600, "0.01": 2.5758 };
// One-sided z corresponding to a given power (1 - beta).
var Z_POWER = { "0.80": 0.8416, "0.90": 1.2816, "0.95": 1.6449 };

function zTwoSidedFromAlpha(alpha) {
  var key = Number(alpha).toFixed(2);
  return Z_TWO_SIDED[key] !== undefined ? Z_TWO_SIDED[key] : 1.9600;
}
function zTwoSidedFromConfidence(conf) {
  var alpha = (1 - Number(conf)).toFixed(2);
  return Z_TWO_SIDED[alpha] !== undefined ? Z_TWO_SIDED[alpha] : 1.9600;
}
function zFromPower(power) {
  var key = Number(power).toFixed(2);
  return Z_POWER[key] !== undefined ? Z_POWER[key] : 0.8416;
}

// Abramowitz & Stegun 7.1.26 approximation of erf (|error| < 1.5e-7)
function erf(x) {
  var sign = x < 0 ? -1 : 1;
  x = Math.abs(x);
  var a1 = 0.254829592, a2 = -0.284496736, a3 = 1.421413741,
      a4 = -1.453152027, a5 = 1.061405429, p = 0.3275911;
  var t = 1 / (1 + p * x);
  var y = 1 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * Math.exp(-x * x);
  return sign * y;
}
function normalCDF(x) { return 0.5 * (1 + erf(x / Math.SQRT2)); }

// Log-gamma (Lanczos approximation) — needed for the incomplete beta function
function gammaln(xx) {
  var cof = [76.18009172947146, -86.50532032941677, 24.01409824083091,
             -1.231739572450155, 0.1208650973866179e-2, -0.5395239384953e-5];
  var x = xx, y = xx, tmp = x + 5.5;
  tmp -= (x + 0.5) * Math.log(tmp);
  var ser = 1.000000000190015;
  for (var j = 0; j < 6; j++) { y += 1; ser += cof[j] / y; }
  return -tmp + Math.log(2.5066282746310005 * ser / x);
}

function betacf(x, a, b) {
  var MAXIT = 100, EPS = 3e-7, FPMIN = 1e-30;
  var qab = a + b, qap = a + 1, qam = a - 1;
  var c = 1, d = 1 - qab * x / qap;
  if (Math.abs(d) < FPMIN) d = FPMIN;
  d = 1 / d;
  var h = d;
  for (var m = 1; m <= MAXIT; m++) {
    var m2 = 2 * m;
    var aa = m * (b - m) * x / ((qam + m2) * (a + m2));
    d = 1 + aa * d; if (Math.abs(d) < FPMIN) d = FPMIN;
    c = 1 + aa / c; if (Math.abs(c) < FPMIN) c = FPMIN;
    d = 1 / d; h *= d * c;
    aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2));
    d = 1 + aa * d; if (Math.abs(d) < FPMIN) d = FPMIN;
    c = 1 + aa / c; if (Math.abs(c) < FPMIN) c = FPMIN;
    d = 1 / d;
    var del = d * c; h *= del;
    if (Math.abs(del - 1) < EPS) break;
  }
  return h;
}

function betainc(x, a, b) {
  if (x <= 0) return 0;
  if (x >= 1) return 1;
  var bt = Math.exp(gammaln(a + b) - gammaln(a) - gammaln(b) + a * Math.log(x) + b * Math.log(1 - x));
  if (x < (a + 1) / (a + b + 2)) {
    return bt * betacf(x, a, b) / a;
  } else {
    return 1 - bt * betacf(1 - x, b, a) / b;
  }
}

// Two-tailed p-value for Student's t with `df` degrees of freedom
function tTwoTailedP(t, df) {
  var x = df / (df + t * t);
  return betainc(x, df / 2, 0.5);
}

// Two-sided critical t value (inverse of tTwoTailedP) via bisection.
// tTwoTailedP(t, df) is monotonically decreasing in t for t >= 0, so a
// simple bisection search is fast and accurate to well beyond what a
// planning/teaching tool needs.
function tCriticalValue(alpha, df) {
  var lo = 0, hi = 1000;
  for (var i = 0; i < 200; i++) {
    var mid = (lo + hi) / 2;
    var p = tTwoTailedP(mid, df);
    if (p > alpha) { lo = mid; } else { hi = mid; }
  }
  return (lo + hi) / 2;
}

function parseNumbers(text) {
  return text
    .split(/[\s,;]+/)
    .map(function (s) { return s.trim(); })
    .filter(function (s) { return s.length > 0; })
    .map(Number)
    .filter(function (n) { return !isNaN(n); });
}

function mean(arr) { return arr.reduce(function (a, b) { return a + b; }, 0) / arr.length; }

function sampleVariance(arr) {
  var m = mean(arr);
  var sq = arr.reduce(function (a, b) { return a + (b - m) * (b - m); }, 0);
  return sq / (arr.length - 1);
}

function quantile(sortedArr, q) {
  var pos = (sortedArr.length - 1) * q;
  var base = Math.floor(pos);
  var rest = pos - base;
  if (sortedArr[base + 1] !== undefined) {
    return sortedArr[base] + rest * (sortedArr[base + 1] - sortedArr[base]);
  }
  return sortedArr[base];
}

function fmt(n, d) {
  if (n === undefined || n === null || isNaN(n)) return "—";
  d = d === undefined ? 3 : d;
  var s = Number(n).toFixed(d);
  return trimTrail(s);
}
function trimTrail(s) {
  if (s.indexOf(".") === -1) return s;
  s = s.replace(/0+$/, "").replace(/\.$/, "");
  return s === "" || s === "-" ? "0" : s;
}

function showBox(id, html) {
  var el = document.getElementById(id);
  el.innerHTML = html;
  el.classList.remove("hidden");
}

function sigWord(p, alpha) {
  return p < alpha
    ? "Statistically significant at α = " + alpha + " (reject H₀)."
    : "Not statistically significant at α = " + alpha + " (fail to reject H₀).";
}

/* ==================== SAMPLE SIZE ==================== */

function calcSSProp1() {
  var p = parseFloat(document.getElementById("ssP1_p").value);
  var d = parseFloat(document.getElementById("ssP1_d").value);
  var conf = document.getElementById("ssP1_conf").value;
  var N = parseFloat(document.getElementById("ssP1_N").value);
  var z = zTwoSidedFromConfidence(conf);

  var n0 = (z * z * p * (1 - p)) / (d * d);
  var n = n0;
  var note = "";
  if (N && N > 0) {
    n = n0 / (1 + (n0 - 1) / N);
    note = " (finite population correction applied, N = " + N + ")";
  }
  n = Math.ceil(n);

  showBox("ssP1_result",
    '<div class="result-label">Required Sample Size</div>' +
    '<div class="result-figure">n = ' + n + '</div>' +
    '<div class="result-note">z = ' + z + ', p = ' + p + ', margin of error = ' + d + note + '</div>');
}

function calcSSProp2() {
  var p1 = parseFloat(document.getElementById("ssP2_p1").value);
  var p2 = parseFloat(document.getElementById("ssP2_p2").value);
  var alpha = document.getElementById("ssP2_alpha").value;
  var power = document.getElementById("ssP2_power").value;
  var k = parseFloat(document.getElementById("ssP2_ratio").value) || 1;

  var za = zTwoSidedFromAlpha(alpha);
  var zb = zFromPower(power);

  var n1 = Math.pow(za + zb, 2) * (p1 * (1 - p1) + (p2 * (1 - p2)) / k) / Math.pow(p1 - p2, 2);
  var n2 = k * n1;
  n1 = Math.ceil(n1);
  n2 = Math.ceil(n2);

  showBox("ssP2_result",
    '<div class="result-label">Required Sample Size</div>' +
    '<div class="result-figure">n₁ = ' + n1 + '&nbsp;&nbsp;·&nbsp;&nbsp;n₂ = ' + n2 + '</div>' +
    '<div class="result-note">Total N = ' + (n1 + n2) + '. α = ' + alpha + ' (two‑sided), power = ' + (power * 100) + '%, allocation ratio = ' + k + '.</div>');
}

function calcSSMean2() {
  var delta = parseFloat(document.getElementById("ssM2_delta").value);
  var sd = parseFloat(document.getElementById("ssM2_sd").value);
  var alpha = document.getElementById("ssM2_alpha").value;
  var power = document.getElementById("ssM2_power").value;
  var k = parseFloat(document.getElementById("ssM2_ratio").value) || 1;

  var za = zTwoSidedFromAlpha(alpha);
  var zb = zFromPower(power);

  var n1 = Math.pow(za + zb, 2) * sd * sd * (1 + 1 / k) / (delta * delta);
  var n2 = k * n1;
  n1 = Math.ceil(n1);
  n2 = Math.ceil(n2);

  showBox("ssM2_result",
    '<div class="result-label">Required Sample Size</div>' +
    '<div class="result-figure">n₁ = ' + n1 + '&nbsp;&nbsp;·&nbsp;&nbsp;n₂ = ' + n2 + '</div>' +
    '<div class="result-note">Total N = ' + (n1 + n2) + '. α = ' + alpha + ' (two‑sided), power = ' + (power * 100) + '%, allocation ratio = ' + k + '.</div>');
}

function calcSSMean1() {
  var sd = parseFloat(document.getElementById("ssM1_sd").value);
  var e = parseFloat(document.getElementById("ssM1_e").value);
  var conf = document.getElementById("ssM1_conf").value;
  var z = zTwoSidedFromConfidence(conf);

  var n = Math.ceil(Math.pow((z * sd) / e, 2));

  showBox("ssM1_result",
    '<div class="result-label">Required Sample Size</div>' +
    '<div class="result-figure">n = ' + n + '</div>' +
    '<div class="result-note">z = ' + z + ', SD = ' + sd + ', margin of error = ' + e + '</div>');
}

/* ======================== POWER ======================== */

function calcPowerProp2() {
  var p1 = parseFloat(document.getElementById("pwP2_p1").value);
  var p2 = parseFloat(document.getElementById("pwP2_p2").value);
  var n = parseFloat(document.getElementById("pwP2_n").value);
  var alpha = document.getElementById("pwP2_alpha").value;
  var za = zTwoSidedFromAlpha(alpha);

  var seAlt = Math.sqrt((p1 * (1 - p1)) / n + (p2 * (1 - p2)) / n);
  var zBeta = Math.abs(p1 - p2) / seAlt - za;
  var power = normalCDF(zBeta);
  power = Math.max(0, Math.min(1, power));

  showBox("pwP2_result",
    '<div class="result-label">Estimated Power</div>' +
    '<div class="result-figure">' + (power * 100).toFixed(1) + '%</div>' +
    '<div class="result-note">With n = ' + n + ' per group, α = ' + alpha + ' (two‑sided), p₁ = ' + p1 + ', p₂ = ' + p2 + '.</div>');
}

function calcPowerMean2() {
  var delta = parseFloat(document.getElementById("pwM2_delta").value);
  var sd = parseFloat(document.getElementById("pwM2_sd").value);
  var n = parseFloat(document.getElementById("pwM2_n").value);
  var alpha = document.getElementById("pwM2_alpha").value;
  var za = zTwoSidedFromAlpha(alpha);

  var se = sd * Math.sqrt(2 / n);
  var zBeta = Math.abs(delta) / se - za;
  var power = normalCDF(zBeta);
  power = Math.max(0, Math.min(1, power));

  showBox("pwM2_result",
    '<div class="result-label">Estimated Power</div>' +
    '<div class="result-figure">' + (power * 100).toFixed(1) + '%</div>' +
    '<div class="result-note">With n = ' + n + ' per group, α = ' + alpha + ' (two‑sided), Δ = ' + delta + ', SD = ' + sd + '.</div>');
}

/* =================== DESCRIPTIVE STATS =================== */

function calcDescriptive() {
  var raw = document.getElementById("descInput").value;
  var data = parseNumbers(raw);
  if (data.length < 2) {
    showBox("descResult", '<div class="result-note">Please enter at least two numeric values.</div>');
    return;
  }
  var sorted = data.slice().sort(function (a, b) { return a - b; });
  var n = data.length;
  var m = mean(data);
  var variance = sampleVariance(data);
  var sd = Math.sqrt(variance);
  var se = sd / Math.sqrt(n);
  var median = quantile(sorted, 0.5);
  var q1 = quantile(sorted, 0.25);
  var q3 = quantile(sorted, 0.75);
  var iqr = q3 - q1;
  var min = sorted[0];
  var max = sorted[n - 1];
  var range = max - min;

  // Mode
  var freq = {};
  data.forEach(function (v) { freq[v] = (freq[v] || 0) + 1; });
  var maxFreq = Math.max.apply(null, Object.values(freq));
  var modes = Object.keys(freq).filter(function (k) { return freq[k] === maxFreq; });
  var modeStr = (maxFreq === 1) ? "None (all unique)" : modes.join(", ");

  var ciZ = 1.96;
  var ciLow = m - ciZ * se;
  var ciHigh = m + ciZ * se;

  var tiles = [
    ["n", n], ["Mean", fmt(m)], ["Median", fmt(median)], ["Mode", modeStr],
    ["Std. Dev.", fmt(sd)], ["Variance", fmt(variance)], ["Std. Error", fmt(se)],
    ["Min", fmt(min)], ["Max", fmt(max)], ["Range", fmt(range)],
    ["Q1", fmt(q1)], ["Q3", fmt(q3)], ["IQR", fmt(iqr)]
  ];

  var html = tiles.map(function (t) {
    return '<div class="stat-tile"><div class="v">' + t[1] + '</div><div class="k">' + t[0] + '</div></div>';
  }).join("");

  var box = document.getElementById("descResult");
  box.innerHTML =
    '<div class="result-label">Descriptive Statistics</div>' +
    '<div class="stat-results-grid">' + html + '</div>' +
    '<div class="result-note">95% CI for the mean (normal approximation): [' + fmt(ciLow) + ', ' + fmt(ciHigh) + ']</div>';
  box.classList.remove("hidden");
}

/* ========================= T-TESTS ========================= */

function calcTTestOne() {
  var data = parseNumbers(document.getElementById("tt1Input").value);
  var mu0 = parseFloat(document.getElementById("tt1_mu0").value);
  var alpha = parseFloat(document.getElementById("tt1_alpha").value);

  if (data.length < 2) {
    showBox("tt1_result", '<div class="result-note">Please enter at least two numeric values.</div>');
    return;
  }
  var n = data.length;
  var m = mean(data);
  var sd = Math.sqrt(sampleVariance(data));
  var se = sd / Math.sqrt(n);
  var t = (m - mu0) / se;
  var df = n - 1;
  var p = tTwoTailedP(Math.abs(t), df);

  showBox("tt1_result",
    '<div class="result-label">One‑Sample t‑test Result</div>' +
    '<div class="result-figure">t = ' + fmt(t, 3) + '</div>' +
    '<div class="stat-results-grid">' +
      '<div class="stat-tile"><div class="v">' + n + '</div><div class="k">n</div></div>' +
      '<div class="stat-tile"><div class="v">' + fmt(m) + '</div><div class="k">Sample Mean</div></div>' +
      '<div class="stat-tile"><div class="v">' + df + '</div><div class="k">df</div></div>' +
      '<div class="stat-tile"><div class="v">' + fmt(p, 4) + '</div><div class="k">p‑value</div></div>' +
    '</div>' +
    '<div class="result-note">' + sigWord(p, alpha) + '</div>');
}

function calcTTestTwo() {
  var g1 = parseNumbers(document.getElementById("tt2_g1").value);
  var g2 = parseNumbers(document.getElementById("tt2_g2").value);
  var alpha = parseFloat(document.getElementById("tt2_alpha").value);

  if (g1.length < 2 || g2.length < 2) {
    showBox("tt2_result", '<div class="result-note">Please enter at least two numeric values in each group.</div>');
    return;
  }
  var n1 = g1.length, n2 = g2.length;
  var m1 = mean(g1), m2 = mean(g2);
  var v1 = sampleVariance(g1), v2 = sampleVariance(g2);
  var se = Math.sqrt(v1 / n1 + v2 / n2);
  var t = (m1 - m2) / se;
  var df = Math.pow(v1 / n1 + v2 / n2, 2) /
    ((Math.pow(v1 / n1, 2) / (n1 - 1)) + (Math.pow(v2 / n2, 2) / (n2 - 1)));
  var p = tTwoTailedP(Math.abs(t), df);

  showBox("tt2_result",
    '<div class="result-label">Welch\'s Two‑Sample t‑test Result</div>' +
    '<div class="result-figure">t = ' + fmt(t, 3) + '</div>' +
    '<div class="stat-results-grid">' +
      '<div class="stat-tile"><div class="v">' + fmt(m1) + '</div><div class="k">Mean (Group 1)</div></div>' +
      '<div class="stat-tile"><div class="v">' + fmt(m2) + '</div><div class="k">Mean (Group 2)</div></div>' +
      '<div class="stat-tile"><div class="v">' + fmt(df, 2) + '</div><div class="k">df (Welch)</div></div>' +
      '<div class="stat-tile"><div class="v">' + fmt(p, 4) + '</div><div class="k">p‑value</div></div>' +
    '</div>' +
    '<div class="result-note">' + sigWord(p, alpha) + ' (Welch\'s t‑test, no equal‑variance assumption.)</div>');
}

/* ====================== CHI-SQUARE 2x2 ====================== */

function calcChiSquare() {
  var a = parseFloat(document.getElementById("chi_a").value) || 0;
  var b = parseFloat(document.getElementById("chi_b").value) || 0;
  var c = parseFloat(document.getElementById("chi_c").value) || 0;
  var d = parseFloat(document.getElementById("chi_d").value) || 0;

  var n = a + b + c + d;
  var rowSums = [a + b, c + d];
  var colSums = [a + c, b + d];

  if (n === 0 || rowSums[0] === 0 || rowSums[1] === 0 || colSums[0] === 0 || colSums[1] === 0) {
    showBox("chi_result", '<div class="result-note">Please enter valid counts (no empty row/column).</div>');
    return;
  }

  var chi2 = (n * Math.pow(a * d - b * c, 2)) / (rowSums[0] * rowSums[1] * colSums[0] * colSums[1]);
  var yatesInner = Math.max(0, Math.abs(a * d - b * c) - n / 2);
  var chi2Yates = (n * Math.pow(yatesInner, 2)) / (rowSums[0] * rowSums[1] * colSums[0] * colSums[1]);

  var p = 2 * (1 - normalCDF(Math.sqrt(chi2)));
  var pYates = 2 * (1 - normalCDF(Math.sqrt(chi2Yates)));

  var or = (b === 0 || c === 0) ? NaN : (a * d) / (b * c);
  var rr = (rowSums[0] === 0 || rowSums[1] === 0 || a === 0) ? NaN :
    (a / rowSums[0]) / (c / rowSums[1]);

  showBox("chi_result",
    '<div class="result-label">Chi‑square Test (2×2)</div>' +
    '<div class="result-figure">&chi;&sup2; = ' + fmt(chi2, 3) + '</div>' +
    '<div class="stat-results-grid">' +
      '<div class="stat-tile"><div class="v">' + fmt(p, 4) + '</div><div class="k">p‑value</div></div>' +
      '<div class="stat-tile"><div class="v">' + fmt(chi2Yates, 3) + '</div><div class="k">&chi;&sup2; (Yates)</div></div>' +
      '<div class="stat-tile"><div class="v">' + fmt(pYates, 4) + '</div><div class="k">p (Yates)</div></div>' +
      '<div class="stat-tile"><div class="v">' + (isNaN(or) ? "—" : fmt(or, 3)) + '</div><div class="k">Odds Ratio</div></div>' +
      '<div class="stat-tile"><div class="v">' + (isNaN(rr) ? "—" : fmt(rr, 3)) + '</div><div class="k">Risk Ratio</div></div>' +
    '</div>' +
    '<div class="result-note">' + sigWord(p, 0.05) + ' df = 1.</div>');
}

/* ==================== CONFIDENCE INTERVALS ==================== */

function ciRangeHTML(label, low, high, point, note, unit) {
  var mult = unit === "%" ? 100 : 1;
  var suffix = unit === "%" ? "%" : "";
  return (
    '<div class="result-label">' + label + '</div>' +
    '<div class="result-figure">[' + fmt(low * mult) + suffix + ', ' + fmt(high * mult) + suffix + ']</div>' +
    '<div class="result-note">Point estimate: ' + fmt(point * mult) + suffix + '. ' + note + '</div>'
  );
}

function calcCIMeanData() {
  var data = parseNumbers(document.getElementById("ciMD_data").value);
  var conf = document.getElementById("ciMD_conf").value;
  if (data.length < 2) {
    showBox("ciMD_result", '<div class="result-note">Please enter at least two numeric values.</div>');
    return;
  }
  var n = data.length;
  var m = mean(data);
  var sd = Math.sqrt(sampleVariance(data));
  var se = sd / Math.sqrt(n);
  var df = n - 1;
  var alpha = (1 - Number(conf)).toFixed(2);
  var t = tCriticalValue(alpha, df);
  var margin = t * se;

  showBox("ciMD_result", ciRangeHTML(
    (conf * 100) + "% Confidence Interval for the Mean",
    m - margin, m + margin, m,
    "n = " + n + ", SD = " + fmt(sd) + ", t(" + df + ") = " + fmt(t, 3) + ", margin of error = " + fmt(margin) + "."
  ));
}

function calcCIMeanSummary() {
  var m = parseFloat(document.getElementById("ciMS_mean").value);
  var sd = parseFloat(document.getElementById("ciMS_sd").value);
  var n = parseFloat(document.getElementById("ciMS_n").value);
  var conf = document.getElementById("ciMS_conf").value;

  var se = sd / Math.sqrt(n);
  var df = n - 1;
  var alpha = (1 - Number(conf)).toFixed(2);
  var t = tCriticalValue(alpha, df);
  var margin = t * se;

  showBox("ciMS_result", ciRangeHTML(
    (conf * 100) + "% Confidence Interval for the Mean",
    m - margin, m + margin, m,
    "n = " + n + ", t(" + df + ") = " + fmt(t, 3) + ", margin of error = " + fmt(margin) + "."
  ));
}

function radioValue(name) {
  var els = document.getElementsByName(name);
  for (var i = 0; i < els.length; i++) { if (els[i].checked) return els[i].value; }
  return null;
}

// Wilson score interval for a single proportion. Returns {low, high}.
function wilsonInterval(x, n, z) {
  var phat = x / n;
  var z2 = z * z;
  var denom = 1 + z2 / n;
  var center = phat + z2 / (2 * n);
  var adj = z * Math.sqrt((phat * (1 - phat)) / n + z2 / (4 * n * n));
  return { low: (center - adj) / denom, high: (center + adj) / denom };
}

// Agresti–Coull interval for a single proportion. Returns {low, high}.
function agrestiCoullInterval(x, n, z) {
  var z2 = z * z;
  var nTilde = n + z2;
  var pTilde = (x + z2 / 2) / nTilde;
  var se = Math.sqrt((pTilde * (1 - pTilde)) / nTilde);
  return { low: pTilde - z * se, high: pTilde + z * se };
}

// Newcombe / MOVER combination of two independent single-proportion
// intervals into an interval for their difference (p1 - p2).
function moverDiff(p1, low1, high1, p2, low2, high2) {
  var diff = p1 - p2;
  var low = diff - Math.sqrt(Math.pow(p1 - low1, 2) + Math.pow(high2 - p2, 2));
  var high = diff + Math.sqrt(Math.pow(high1 - p1, 2) + Math.pow(p2 - low2, 2));
  return { low: low, high: high };
}

function calcCIProp() {
  var x = parseFloat(document.getElementById("ciP_x").value);
  var n = parseFloat(document.getElementById("ciP_n").value);
  var conf = document.getElementById("ciP_conf").value;
  var method = radioValue("ciP_method") || "wald";
  var units = radioValue("ciP_units") || "percent";
  var unit = units === "percent" ? "%" : "";

  if (!(n > 0) || x < 0 || x > n) {
    showBox("ciP_result", '<div class="result-note">Please enter a valid number of events (0 ≤ x ≤ n).</div>');
    return;
  }
  var phat = x / n;
  var z = zTwoSidedFromConfidence(conf);
  var low, high, methodLabel, note;

  if (method === "wilson") {
    var wi = wilsonInterval(x, n, z);
    low = wi.low; high = wi.high;
    methodLabel = "Wilson Score";
    note = "n = " + n + ", z = " + z + ".";
  } else if (method === "agresti") {
    var ac = agrestiCoullInterval(x, n, z);
    low = ac.low; high = ac.high;
    methodLabel = "Agresti–Coull";
    note = "n = " + n + ", z = " + z + ".";
  } else {
    var se = Math.sqrt((phat * (1 - phat)) / n);
    var margin = z * se;
    low = Math.max(0, phat - margin);
    high = Math.min(1, phat + margin);
    methodLabel = "Wald";
    note = "n = " + n + ", z = " + z + ", margin of error = " + fmt(margin * 100) + " percentage points.";
  }

  showBox("ciP_result", ciRangeHTML(
    (conf * 100) + "% Confidence Interval for the Proportion (" + methodLabel + ")",
    low, high, phat, note, unit
  ));
}

function calcCIDiffMeans() {
  var g1 = parseNumbers(document.getElementById("ciDM_g1").value);
  var g2 = parseNumbers(document.getElementById("ciDM_g2").value);
  var conf = document.getElementById("ciDM_conf").value;

  if (g1.length < 2 || g2.length < 2) {
    showBox("ciDM_result", '<div class="result-note">Please enter at least two numeric values in each group.</div>');
    return;
  }
  var n1 = g1.length, n2 = g2.length;
  var m1 = mean(g1), m2 = mean(g2);
  var v1 = sampleVariance(g1), v2 = sampleVariance(g2);
  var se = Math.sqrt(v1 / n1 + v2 / n2);
  var df = Math.pow(v1 / n1 + v2 / n2, 2) /
    ((Math.pow(v1 / n1, 2) / (n1 - 1)) + (Math.pow(v2 / n2, 2) / (n2 - 1)));
  var alpha = (1 - Number(conf)).toFixed(2);
  var t = tCriticalValue(alpha, df);
  var diff = m1 - m2;
  var margin = t * se;

  showBox("ciDM_result", ciRangeHTML(
    (conf * 100) + "% Confidence Interval for the Difference of Means (Group 1 − Group 2)",
    diff - margin, diff + margin, diff,
    "df (Welch) = " + fmt(df, 2) + ", t = " + fmt(t, 3) + ", margin of error = " + fmt(margin) + "."
  ));
}

function calcCIDiffProps() {
  var x1 = parseFloat(document.getElementById("ciDP_x1").value);
  var n1 = parseFloat(document.getElementById("ciDP_n1").value);
  var x2 = parseFloat(document.getElementById("ciDP_x2").value);
  var n2 = parseFloat(document.getElementById("ciDP_n2").value);
  var conf = document.getElementById("ciDP_conf").value;
  var method = radioValue("ciDP_method") || "wald";
  var units = radioValue("ciDP_units") || "percent";
  var unit = units === "percent" ? "%" : "";

  if (!(n1 > 0) || !(n2 > 0) || x1 < 0 || x1 > n1 || x2 < 0 || x2 > n2) {
    showBox("ciDP_result", '<div class="result-note">Please enter valid event counts (0 ≤ x ≤ n) for both groups.</div>');
    return;
  }
  var p1 = x1 / n1, p2 = x2 / n2;
  var z = zTwoSidedFromConfidence(conf);
  var diff = p1 - p2;
  var low, high, methodLabel, note;

  if (method === "wilson") {
    var w1 = wilsonInterval(x1, n1, z), w2 = wilsonInterval(x2, n2, z);
    var mw = moverDiff(p1, w1.low, w1.high, p2, w2.low, w2.high);
    low = Math.max(-1, mw.low); high = Math.min(1, mw.high);
    methodLabel = "Wilson / Newcombe";
    note = "z = " + z + ".";
  } else if (method === "agresti") {
    var a1 = agrestiCoullInterval(x1, n1, z), a2 = agrestiCoullInterval(x2, n2, z);
    var ma = moverDiff(p1, a1.low, a1.high, p2, a2.low, a2.high);
    low = Math.max(-1, ma.low); high = Math.min(1, ma.high);
    methodLabel = "Agresti–Coull / MOVER";
    note = "z = " + z + ".";
  } else {
    var se = Math.sqrt((p1 * (1 - p1)) / n1 + (p2 * (1 - p2)) / n2);
    var margin = z * se;
    low = Math.max(-1, diff - margin); high = Math.min(1, diff + margin);
    methodLabel = "Wald";
    note = "z = " + z + ", margin of error = " + fmt(margin * 100) + " percentage points.";
  }

  showBox("ciDP_result", ciRangeHTML(
    (conf * 100) + "% Confidence Interval for the Difference of Proportions (" + methodLabel + ")",
    low, high, diff, note, unit
  ));
}

/* ---- Odds Ratio / Risk Ratio CIs (log-scale Wald, from a 2x2 table) ---- */

function haldaneCorrect(a, b, c, d) {
  if (a === 0 || b === 0 || c === 0 || d === 0) {
    return { a: a + 0.5, b: b + 0.5, c: c + 0.5, d: d + 0.5, corrected: true };
  }
  return { a: a, b: b, c: c, d: d, corrected: false };
}

function calcCIOddsRatio() {
  var a0 = parseFloat(document.getElementById("ciOR_a").value) || 0;
  var b0 = parseFloat(document.getElementById("ciOR_b").value) || 0;
  var c0 = parseFloat(document.getElementById("ciOR_c").value) || 0;
  var d0 = parseFloat(document.getElementById("ciOR_d").value) || 0;
  var conf = document.getElementById("ciOR_conf").value;

  if (a0 + b0 + c0 + d0 <= 0) {
    showBox("ciOR_result", '<div class="result-note">Please enter valid cell counts.</div>');
    return;
  }
  var cc = haldaneCorrect(a0, b0, c0, d0);
  var or = (cc.a * cc.d) / (cc.b * cc.c);
  var seLn = Math.sqrt(1 / cc.a + 1 / cc.b + 1 / cc.c + 1 / cc.d);
  var z = zTwoSidedFromConfidence(conf);
  var lnOR = Math.log(or);
  var low = Math.exp(lnOR - z * seLn);
  var high = Math.exp(lnOR + z * seLn);

  showBox("ciOR_result",
    '<div class="result-label">' + (conf * 100) + '% Confidence Interval for the Odds Ratio</div>' +
    '<div class="result-figure">[' + fmt(low, 3) + ', ' + fmt(high, 3) + ']</div>' +
    '<div class="result-note">Point estimate: OR = ' + fmt(or, 3) + '. z = ' + z + '.' +
    (cc.corrected ? ' A 0.5 continuity correction was applied (a zero cell was present).' : '') + '</div>');
}

function calcCIRiskRatio() {
  var a0 = parseFloat(document.getElementById("ciRR_a").value) || 0;
  var b0 = parseFloat(document.getElementById("ciRR_b").value) || 0;
  var c0 = parseFloat(document.getElementById("ciRR_c").value) || 0;
  var d0 = parseFloat(document.getElementById("ciRR_d").value) || 0;
  var conf = document.getElementById("ciRR_conf").value;

  if (a0 + b0 + c0 + d0 <= 0) {
    showBox("ciRR_result", '<div class="result-note">Please enter valid cell counts.</div>');
    return;
  }
  var cc = haldaneCorrect(a0, b0, c0, d0);
  var n1 = cc.a + cc.b, n2 = cc.c + cc.d;
  var rr = (cc.a / n1) / (cc.c / n2);
  var seLn = Math.sqrt(1 / cc.a - 1 / n1 + 1 / cc.c - 1 / n2);
  var z = zTwoSidedFromConfidence(conf);
  var lnRR = Math.log(rr);
  var low = Math.exp(lnRR - z * seLn);
  var high = Math.exp(lnRR + z * seLn);

  showBox("ciRR_result",
    '<div class="result-label">' + (conf * 100) + '% Confidence Interval for the Risk Ratio</div>' +
    '<div class="result-figure">[' + fmt(low, 3) + ', ' + fmt(high, 3) + ']</div>' +
    '<div class="result-note">Point estimate: RR = ' + fmt(rr, 3) + '. z = ' + z + '.' +
    (cc.corrected ? ' A 0.5 continuity correction was applied (a zero cell was present).' : '') + '</div>');
}

/* ==================== NON-PARAMETRIC TESTS ==================== */

// Ranks an array of values (ascending), assigning the average rank to ties.
// Returns { ranks: [rank per original position], tieSizes: [size of each tied group] }.
function rankWithTies(values) {
  var indexed = values.map(function (v, i) { return { v: v, i: i }; });
  indexed.sort(function (a, b) { return a.v - b.v; });
  var ranks = new Array(values.length);
  var tieSizes = [];
  var i = 0;
  while (i < indexed.length) {
    var j = i;
    while (j + 1 < indexed.length && indexed[j + 1].v === indexed[i].v) j++;
    var avgRank = (i + 1 + j + 1) / 2; // average of the 1-based rank positions i+1..j+1
    for (var k = i; k <= j; k++) { ranks[indexed[k].i] = avgRank; }
    tieSizes.push(j - i + 1);
    i = j + 1;
  }
  return { ranks: ranks, tieSizes: tieSizes };
}

function calcWilcoxonSignedRank() {
  var data = parseNumbers(document.getElementById("wsr_data").value);
  var mu0 = parseFloat(document.getElementById("wsr_mu0").value);
  var alpha = parseFloat(document.getElementById("wsr_alpha").value);

  var diffs = data.map(function (v) { return v - mu0; }).filter(function (d) { return d !== 0; });
  var n = diffs.length;
  if (n < 1) {
    showBox("wsr_result", '<div class="result-note">Please enter sample data (values equal to μ₀ are dropped before ranking).</div>');
    return;
  }
  var absDiffs = diffs.map(Math.abs);
  var rankInfo = rankWithTies(absDiffs);
  var ranks = rankInfo.ranks;
  var wPlus = 0, wMinus = 0;
  for (var i = 0; i < n; i++) {
    if (diffs[i] > 0) wPlus += ranks[i]; else wMinus += ranks[i];
  }
  var meanW = (n * (n + 1)) / 4;
  var tieSum = rankInfo.tieSizes.reduce(function (s, t) { return s + (t * t * t - t); }, 0);
  var varW = (n * (n + 1) * (2 * n + 1)) / 24 - tieSum / 48;
  var z = wPlus > meanW
    ? (wPlus - meanW - 0.5) / Math.sqrt(varW)
    : (wPlus - meanW + 0.5) / Math.sqrt(varW);
  var p = 2 * (1 - normalCDF(Math.abs(z)));

  showBox("wsr_result",
    '<div class="result-label">Wilcoxon Signed‑Rank Test</div>' +
    '<div class="result-figure">z = ' + fmt(z, 3) + '</div>' +
    '<div class="stat-results-grid">' +
      '<div class="stat-tile"><div class="v">' + n + '</div><div class="k">n (non‑zero diffs)</div></div>' +
      '<div class="stat-tile"><div class="v">' + fmt(wPlus, 1) + '</div><div class="k">W+</div></div>' +
      '<div class="stat-tile"><div class="v">' + fmt(wMinus, 1) + '</div><div class="k">W−</div></div>' +
      '<div class="stat-tile"><div class="v">' + fmt(p, 4) + '</div><div class="k">p‑value</div></div>' +
    '</div>' +
    '<div class="result-note">' + sigWord(p, alpha) + '</div>');
}

function calcMannWhitneyU() {
  var g1 = parseNumbers(document.getElementById("mwu_g1").value);
  var g2 = parseNumbers(document.getElementById("mwu_g2").value);
  var alpha = parseFloat(document.getElementById("mwu_alpha").value);

  var n1 = g1.length, n2 = g2.length;
  if (n1 < 1 || n2 < 1) {
    showBox("mwu_result", '<div class="result-note">Please enter at least one value in each group.</div>');
    return;
  }
  var combined = g1.concat(g2);
  var rankInfo = rankWithTies(combined);
  var ranks = rankInfo.ranks;
  var R1 = 0;
  for (var i = 0; i < n1; i++) { R1 += ranks[i]; }
  var U1 = R1 - (n1 * (n1 + 1)) / 2;
  var U2 = n1 * n2 - U1;
  var U = Math.min(U1, U2);
  var meanU = (n1 * n2) / 2;
  var N = n1 + n2;
  var tieSum = rankInfo.tieSizes.reduce(function (s, t) { return s + (t * t * t - t); }, 0);
  var varU = (n1 * n2 / 12) * ((N + 1) - tieSum / (N * (N - 1)));
  var z = U < meanU
    ? (U - meanU + 0.5) / Math.sqrt(varU)
    : (U - meanU - 0.5) / Math.sqrt(varU);
  var p = 2 * (1 - normalCDF(Math.abs(z)));

  showBox("mwu_result",
    '<div class="result-label">Mann‑Whitney U Test</div>' +
    '<div class="result-figure">z = ' + fmt(z, 3) + '</div>' +
    '<div class="stat-results-grid">' +
      '<div class="stat-tile"><div class="v">' + fmt(U1, 1) + '</div><div class="k">U (Group 1)</div></div>' +
      '<div class="stat-tile"><div class="v">' + fmt(U2, 1) + '</div><div class="k">U (Group 2)</div></div>' +
      '<div class="stat-tile"><div class="v">' + n1 + ' / ' + n2 + '</div><div class="k">n₁ / n₂</div></div>' +
      '<div class="stat-tile"><div class="v">' + fmt(p, 4) + '</div><div class="k">p‑value</div></div>' +
    '</div>' +
    '<div class="result-note">' + sigWord(p, alpha) + '</div>');
}

/* ==================== SVG CHART HELPER ==================== */

function svgChart(config) {
  var W = config.width || 600, H = config.height || 220;
  var padL = 40, padR = 14, padT = 14, padB = 26;
  var allX = [], allY = [];
  config.series.forEach(function (s) {
    s.points.forEach(function (p) { allX.push(p[0]); allY.push(p[1]); });
  });
  var xMin = config.xDomain ? config.xDomain[0] : Math.min.apply(null, allX);
  var xMax = config.xDomain ? config.xDomain[1] : Math.max.apply(null, allX);
  var yMin = config.yDomain ? config.yDomain[0] : Math.min.apply(null, allY);
  var yMax = config.yDomain ? config.yDomain[1] : Math.max.apply(null, allY);
  if (xMin === xMax) { xMax = xMin + 1; }
  if (yMin === yMax) { yMin -= 1; yMax += 1; }
  if (!config.yDomain) {
    var yPad = (yMax - yMin) * 0.08;
    yMin -= yPad; yMax += yPad;
  }

  function sx(x) { return padL + (x - xMin) / (xMax - xMin) * (W - padL - padR); }
  function sy(y) { return H - padB - (y - yMin) / (yMax - yMin) * (H - padT - padB); }

  var svg = '<svg viewBox="0 0 ' + W + ' ' + H + '" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="' + (config.ariaLabel || "chart") + '">';
  svg += '<line x1="' + padL + '" y1="' + (H - padB) + '" x2="' + (W - padR) + '" y2="' + (H - padB) + '" stroke="#e4eaf1" stroke-width="1"/>';
  svg += '<line x1="' + padL + '" y1="' + padT + '" x2="' + padL + '" y2="' + (H - padB) + '" stroke="#e4eaf1" stroke-width="1"/>';
  svg += '<text x="' + padL + '" y="' + (H - padB + 16) + '" font-size="10" fill="#5b6b7c">' + fmt(xMin, 1) + '</text>';
  svg += '<text x="' + (W - padR) + '" y="' + (H - padB + 16) + '" font-size="10" fill="#5b6b7c" text-anchor="end">' + fmt(xMax, 1) + '</text>';
  svg += '<text x="' + (padL - 6) + '" y="' + (H - padB) + '" font-size="10" fill="#5b6b7c" text-anchor="end">' + fmt(yMin, 2) + '</text>';
  svg += '<text x="' + (padL - 6) + '" y="' + (padT + 8) + '" font-size="10" fill="#5b6b7c" text-anchor="end">' + fmt(yMax, 2) + '</text>';

  config.series.forEach(function (s) {
    var ptsStr = s.points.map(function (p) { return sx(p[0]).toFixed(2) + ',' + sy(p[1]).toFixed(2); }).join(' ');
    var dash = s.dashed ? ' stroke-dasharray="5,4"' : '';
    svg += '<polyline points="' + ptsStr + '" fill="none" stroke="' + s.color + '" stroke-width="2"' + dash + ' stroke-linejoin="round" stroke-linecap="round"/>';
    if (s.showDots) {
      s.points.forEach(function (p) {
        svg += '<circle cx="' + sx(p[0]).toFixed(2) + '" cy="' + sy(p[1]).toFixed(2) + '" r="2.6" fill="' + s.color + '"/>';
      });
    }
  });
  svg += '</svg>';
  return svg;
}

/* ==================== SURVIVAL ANALYSIS ==================== */

function parseTimeStatusPairs(text) {
  var rows = [];
  text.split(/\n|;/).forEach(function (line) {
    var parts = line.split(",");
    if (parts.length < 2) return;
    var t = parseFloat(parts[0]);
    var s = parseFloat(parts[1]);
    if (!isNaN(t) && (s === 0 || s === 1)) rows.push({ time: t, status: s });
  });
  return rows;
}

function calcKaplanMeier() {
  var rows = parseTimeStatusPairs(document.getElementById("km_data").value);
  var conf = document.getElementById("km_conf").value;
  if (rows.length < 1) {
    showBox("km_result", '<div class="result-note">Please enter at least one valid "time,status" row.</div>');
    return;
  }
  rows.sort(function (a, b) { return a.time - b.time; });
  var n = rows.length;
  var times = Array.from(new Set(rows.map(function (r) { return r.time; }))).sort(function (a, b) { return a - b; });
  var z = zTwoSidedFromConfidence(conf);

  var atRisk = n;
  var survival = 1;
  var greenwoodSum = 0;
  var tableRows = [];
  var chartPoints = [];
  var median = null;

  times.forEach(function (t) {
    var events = rows.filter(function (r) { return r.time === t && r.status === 1; }).length;
    var censored = rows.filter(function (r) { return r.time === t && r.status === 0; }).length;
    if (events > 0) {
      var prevSurvival = survival;
      survival = survival * (1 - events / atRisk);
      if (atRisk - events > 0) { greenwoodSum += events / (atRisk * (atRisk - events)); }
      var se = survival * Math.sqrt(greenwoodSum);
      var ciLow = Math.max(0, survival - z * se);
      var ciHigh = Math.min(1, survival + z * se);
      tableRows.push({ time: t, atRisk: atRisk, events: events, censored: censored, survival: survival, ciLow: ciLow, ciHigh: ciHigh });
      chartPoints.push({ t: t, s: survival });
      if (median === null && survival <= 0.5) { median = t; }
    }
    atRisk -= (events + censored);
  });

  var rowsHTML = tableRows.map(function (r) {
    return '<tr><td>' + fmt(r.time, 2) + '</td><td>' + r.atRisk + '</td><td>' + r.events + '</td><td>' + r.censored +
      '</td><td>' + fmt(r.survival, 4) + '</td><td>[' + fmt(r.ciLow, 3) + ', ' + fmt(r.ciHigh, 3) + ']</td></tr>';
  }).join("");

  var stepPts = [[0, 1]];
  var prevS = 1;
  chartPoints.forEach(function (cp) {
    stepPts.push([cp.t, prevS]);
    stepPts.push([cp.t, cp.s]);
    prevS = cp.s;
  });
  var maxT = times[times.length - 1];
  if (stepPts[stepPts.length - 1][0] < maxT) { stepPts.push([maxT, prevS]); }

  var chartSVG = svgChart({
    width: 600, height: 220,
    xDomain: [0, maxT], yDomain: [0, 1],
    ariaLabel: "Kaplan-Meier survival curve",
    series: [{ points: stepPts, color: "#14a693" }]
  });

  var box = document.getElementById("km_result");
  box.innerHTML =
    '<div class="result-label">Kaplan‑Meier Survival Estimate</div>' +
    '<div class="result-figure">' + (median !== null ? fmt(median, 2) : "Not reached") + '</div>' +
    '<div class="result-note">Median survival time' + (median !== null ? "" : " (survival never drops to 50%)") +
    '. n = ' + n + ', total events = ' + tableRows.reduce(function (s, r) { return s + r.events; }, 0) + '.</div>' +
    '<div class="chart-wrap">' + chartSVG + '<div class="chart-caption">Survival probability S(t) vs. time, with ' + (conf * 100) + '% Greenwood CI in the table below.</div></div>' +
    '<div class="table-wrap"><table class="data-table"><thead><tr><th>Time</th><th>At risk</th><th>Events</th><th>Censored</th><th>S(t)</th><th>' + (conf * 100) + '% CI</th></tr></thead><tbody>' +
    rowsHTML + '</tbody></table></div>';
  box.classList.remove("hidden");
}

function calcLogRankTest() {
  var g1 = parseTimeStatusPairs(document.getElementById("lr_g1").value);
  var g2 = parseTimeStatusPairs(document.getElementById("lr_g2").value);
  var alpha = parseFloat(document.getElementById("lr_alpha").value);

  if (g1.length < 1 || g2.length < 1) {
    showBox("lr_result", '<div class="result-note">Please enter valid "time,status" data for both groups.</div>');
    return;
  }
  var allTimes = Array.from(new Set(g1.concat(g2).map(function (r) { return r.time; }))).sort(function (a, b) { return a - b; });
  var atRisk1 = g1.length, atRisk2 = g2.length;
  var O1 = 0, E1 = 0, V = 0;

  allTimes.forEach(function (t) {
    var d1 = g1.filter(function (r) { return r.time === t && r.status === 1; }).length;
    var c1 = g1.filter(function (r) { return r.time === t && r.status === 0; }).length;
    var d2 = g2.filter(function (r) { return r.time === t && r.status === 1; }).length;
    var c2 = g2.filter(function (r) { return r.time === t && r.status === 0; }).length;
    var d = d1 + d2;
    var n1cur = atRisk1, n2cur = atRisk2, ncur = n1cur + n2cur;
    if (d > 0 && ncur > 1) {
      E1 += (n1cur * d) / ncur;
      O1 += d1;
      V += (n1cur * n2cur * d * (ncur - d)) / (ncur * ncur * (ncur - 1));
    }
    atRisk1 -= (d1 + c1);
    atRisk2 -= (d2 + c2);
  });

  if (V <= 0) {
    showBox("lr_result", '<div class="result-note">Not enough overlapping risk sets to compute the test — check your data.</div>');
    return;
  }
  var chi2 = Math.pow(O1 - E1, 2) / V;
  var p = 2 * (1 - normalCDF(Math.sqrt(chi2)));

  showBox("lr_result",
    '<div class="result-label">Log‑Rank Test</div>' +
    '<div class="result-figure">&chi;&sup2; = ' + fmt(chi2, 3) + '</div>' +
    '<div class="stat-results-grid">' +
      '<div class="stat-tile"><div class="v">' + fmt(O1, 1) + '</div><div class="k">Observed (Group 1)</div></div>' +
      '<div class="stat-tile"><div class="v">' + fmt(E1, 2) + '</div><div class="k">Expected (Group 1)</div></div>' +
      '<div class="stat-tile"><div class="v">1</div><div class="k">df</div></div>' +
      '<div class="stat-tile"><div class="v">' + fmt(p, 4) + '</div><div class="k">p‑value</div></div>' +
    '</div>' +
    '<div class="result-note">' + sigWord(p, alpha) + '</div>');
}

/* ==================== TIME SERIES ANALYSIS ==================== */

function calcTrendAnalysis() {
  var y = parseNumbers(document.getElementById("tsT_data").value);
  var alpha = parseFloat(document.getElementById("tsT_alpha").value);
  var n = y.length;
  if (n < 3) {
    showBox("tsT_result", '<div class="result-note">Please enter at least three sequential values.</div>');
    return;
  }
  var x = y.map(function (_, i) { return i + 1; });
  var xbar = mean(x), ybar = mean(y);
  var Sxx = 0, Sxy = 0;
  for (var i = 0; i < n; i++) {
    Sxx += Math.pow(x[i] - xbar, 2);
    Sxy += (x[i] - xbar) * (y[i] - ybar);
  }
  var b = Sxy / Sxx;
  var a = ybar - b * xbar;
  var SSres = 0, SStot = 0;
  var fitted = [];
  for (var j = 0; j < n; j++) {
    var pred = a + b * x[j];
    fitted.push(pred);
    SSres += Math.pow(y[j] - pred, 2);
    SStot += Math.pow(y[j] - ybar, 2);
  }
  var r2 = SStot > 0 ? 1 - SSres / SStot : 1;
  var df = n - 2;
  var mse = SSres / df;
  var seB = Math.sqrt(mse / Sxx);
  var t = b / seB;
  var p = tTwoTailedP(Math.abs(t), df);

  var direction = Math.abs(t) < 1e-9 ? "flat" : (b > 0 ? "increasing" : "decreasing");
  var sigNote = p < alpha
    ? "The " + direction + " trend is statistically significant at α = " + alpha + "."
    : "The trend is not statistically significant at α = " + alpha + ".";

  var scatterPts = x.map(function (xi, i) { return [xi, y[i]]; });
  var linePts = [[x[0], a + b * x[0]], [x[n - 1], a + b * x[n - 1]]];
  var chartSVG = svgChart({
    width: 600, height: 220,
    ariaLabel: "Time series with fitted linear trend",
    series: [
      { points: scatterPts, color: "#5b6b7c", showDots: true, dashed: false },
      { points: linePts, color: "#14a693" }
    ]
  });

  var box = document.getElementById("tsT_result");
  box.innerHTML =
    '<div class="result-label">Linear Trend Fit</div>' +
    '<div class="result-figure">b = ' + fmt(b, 4) + ' per period</div>' +
    '<div class="stat-results-grid">' +
      '<div class="stat-tile"><div class="v">' + fmt(a, 3) + '</div><div class="k">Intercept (a)</div></div>' +
      '<div class="stat-tile"><div class="v">' + fmt(r2, 4) + '</div><div class="k">R²</div></div>' +
      '<div class="stat-tile"><div class="v">' + fmt(t, 3) + '</div><div class="k">t (df=' + df + ')</div></div>' +
      '<div class="stat-tile"><div class="v">' + fmt(p, 4) + '</div><div class="k">p‑value</div></div>' +
    '</div>' +
    '<div class="result-note">' + sigNote + '</div>' +
    '<div class="chart-wrap">' + chartSVG + '<div class="chart-caption">Data points (grey) with fitted trend line (teal).</div></div>';
  box.classList.remove("hidden");
}

function calcMovingAverage() {
  var y = parseNumbers(document.getElementById("tsMA_data").value);
  var w = parseInt(document.getElementById("tsMA_window").value, 10);
  var n = y.length;
  if (n < 2 || !(w >= 2) || w > n) {
    showBox("tsMA_result", '<div class="result-note">Please enter a series and a window size between 2 and the number of values.</div>');
    return;
  }
  var smoothed = [];
  for (var i = w - 1; i < n; i++) {
    var sum = 0;
    for (var k = i - w + 1; k <= i; k++) { sum += y[k]; }
    smoothed.push({ x: i + 1, v: sum / w });
  }

  var originalPts = y.map(function (v, i) { return [i + 1, v]; });
  var smoothedPts = smoothed.map(function (s) { return [s.x, s.v]; });
  var chartSVG = svgChart({
    width: 600, height: 220,
    ariaLabel: "Original series and moving-average smoothed series",
    series: [
      { points: originalPts, color: "#c3cdd7", showDots: true },
      { points: smoothedPts, color: "#14a693" }
    ]
  });

  var tableRows = smoothed.map(function (s) {
    return '<tr><td>' + s.x + '</td><td>' + fmt(y[s.x - 1], 3) + '</td><td>' + fmt(s.v, 3) + '</td></tr>';
  }).join("");

  var box = document.getElementById("tsMA_result");
  box.innerHTML =
    '<div class="result-label">Moving Average (window = ' + w + ')</div>' +
    '<div class="chart-wrap">' + chartSVG + '<div class="chart-caption">Original series (grey) vs. smoothed series (teal).</div></div>' +
    '<div class="table-wrap"><table class="data-table"><thead><tr><th>Period</th><th>Original</th><th>MA(' + w + ')</th></tr></thead><tbody>' +
    tableRows + '</tbody></table></div>';
  box.classList.remove("hidden");
}
