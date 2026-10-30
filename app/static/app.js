"use strict";

const $ = (id) => document.getElementById(id);
const NASA_COLS = ["unit", "cycle", "setting1", "setting2", "setting3",
  ...Array.from({ length: 21 }, (_, i) => `sensor${i + 1}`)];
const ADVICE = {
  normal: ["Normal", "Sin acción inmediata; seguir monitoreando."],
  inspeccion: ["Inspección", "Programar una inspección en los próximos ciclos."],
  critico: ["Crítico", "Programar mantenimiento de inmediato."],
};
const SATURATION = 120;

let info = null;        // /api/info
let S = null;           // resultado actual: { cycles, rul, rulReal, history }
let parsed = null;      // grupos leídos del texto en "Mis datos"
let mode = "demo";

const fmt = (v) => v.toLocaleString("es-PE", { minimumFractionDigits: 1, maximumFractionDigits: 1 });

async function api(path, options) {
  const res = await fetch(path, options);
  if (!res.ok) {
    let msg = `Error ${res.status}`;
    try { const j = await res.json(); if (j.detail) msg = typeof j.detail === "string" ? j.detail : JSON.stringify(j.detail); } catch (_) { /* sin cuerpo */ }
    throw new Error(msg);
  }
  return res.json();
}

function showError(msg) {
  $("error").textContent = msg || "";
  $("error").hidden = !msg;
}

// ---------- gráficos SVG ----------
function niceTicks(min, max, n = 5) {
  const span = max - min || 1;
  const raw = span / n;
  const mag = Math.pow(10, Math.floor(Math.log10(raw)));
  const step = [1, 2, 2.5, 5, 10].map((k) => k * mag).find((s) => s >= raw);
  const out = [];
  for (let v = Math.ceil(min / step) * step; v <= max + 1e-9; v += step) out.push(+v.toFixed(10));
  return out;
}

function lineChart(el, { series, band, cursor, yMin, yMax, xLabel, yLabel }) {
  const W = 800, H = 300, m = { l: 52, r: 14, t: 14, b: 38 };
  const xs = series.flatMap((s) => s.x);
  const xMin = Math.min(...xs), xMax = Math.max(...xs);
  const ys = series.flatMap((s) => s.y);
  let lo = yMin ?? Math.min(...ys), hi = yMax ?? Math.max(...ys);
  if (hi === lo) { hi += 1; lo -= 1; }
  if (yMin === undefined) { const pad = (hi - lo) * 0.06; lo -= pad; hi += pad; }
  const X = (v) => m.l + ((v - xMin) / (xMax - xMin || 1)) * (W - m.l - m.r);
  const Y = (v) => H - m.b - ((v - lo) / (hi - lo)) * (H - m.t - m.b);
  const path = (s) => s.x.map((x, i) => `${i ? "L" : "M"}${X(x).toFixed(1)},${Y(s.y[i]).toFixed(1)}`).join("");

  let g = "";
  for (const t of niceTicks(lo, hi)) {
    g += `<line class="grid" x1="${m.l}" x2="${W - m.r}" y1="${Y(t)}" y2="${Y(t)}"/><text x="${m.l - 6}" y="${Y(t) + 4}" text-anchor="end">${+t.toPrecision(5)}</text>`;
  }
  for (const t of niceTicks(xMin, xMax, 8)) {
    g += `<text x="${X(t)}" y="${H - m.b + 16}" text-anchor="middle">${t}</text>`;
  }
  let b = "";
  if (band) {
    const top = band.x.map((x, i) => `${X(x).toFixed(1)},${Y(band.hi[i]).toFixed(1)}`);
    const bot = band.x.map((x, i) => `${X(x).toFixed(1)},${Y(band.lo[i]).toFixed(1)}`).reverse();
    b = `<polygon points="${top.concat(bot).join(" ")}" fill="var(--band)"/>`;
  }
  const lines = series.map((s) =>
    `<path d="${path(s)}" fill="none" stroke="${s.color}" stroke-width="${s.width || 2}"${s.dash ? ` stroke-dasharray="${s.dash}"` : ""} stroke-linejoin="round"/>`).join("");
  let cur = "";
  if (cursor !== undefined) {
    cur = `<line x1="${X(cursor)}" x2="${X(cursor)}" y1="${m.t}" y2="${H - m.b}" stroke="var(--muted)" stroke-dasharray="4 3"/>`;
    for (const s of series) {
      const i = s.x.indexOf(cursor);
      if (i >= 0 && !s.noDot) cur += `<circle cx="${X(cursor)}" cy="${Y(s.y[i])}" r="4.5" fill="${s.color}" stroke="var(--panel)" stroke-width="2"/>`;
    }
  }
  let lg = "", lx = m.l + 10;
  const items = [...series.map((s) => ({ name: s.name, color: s.color, dash: s.dash })),
    ...(band ? [{ name: band.name, color: "var(--band)", fill: true }] : [])];
  for (const it of items) {
    lg += it.fill
      ? `<rect x="${lx}" y="${m.t + 4}" width="16" height="10" fill="${it.color}" stroke="var(--pred)" stroke-opacity=".4"/>`
      : `<line x1="${lx}" x2="${lx + 16}" y1="${m.t + 9}" y2="${m.t + 9}" stroke="${it.color}" stroke-width="3"${it.dash ? ` stroke-dasharray="${it.dash}"` : ""}/>`;
    lg += `<text x="${lx + 22}" y="${m.t + 13}">${it.name}</text>`;
    lx += 30 + it.name.length * 6.6;
  }
  el.innerHTML = `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="${yLabel} frente a ${xLabel}">${g}${b}${lines}${cur}
    <g class="legend">${lg}</g>
    <text x="${(W + m.l - m.r) / 2}" y="${H - 4}" text-anchor="middle">${xLabel}</text>
    <text transform="translate(13 ${(H - m.b + m.t) / 2}) rotate(-90)" text-anchor="middle">${yLabel}</text></svg>`;
}

function movingMean(arr, w) {
  return arr.map((_, i) => { const s = arr.slice(Math.max(0, i - w + 1), i + 1); return s.reduce((a, b) => a + b, 0) / s.length; });
}

// ---------- visualización ----------
function render() {
  if (!S) return;
  const n = S.cycles.length;
  const idx = +$("cycle").value - 1;
  const cyc = S.cycles[idx];
  $("cycle-out").textContent = `${cyc} de ${S.cycles[n - 1]}`;

  const rul = S.rul[idx];
  const cons = Math.max(rul - info.overestimate_p90, 0);
  const status = cons <= info.thresholds.critical ? "critico" : cons <= info.thresholds.inspection ? "inspeccion" : "normal";
  $("rul").textContent = fmt(rul);
  $("cons").textContent = fmt(cons);
  $("badge").textContent = ADVICE[status][0];
  $("badge").className = `badge ${status}`;
  $("advice").textContent = ADVICE[status][1];
  $("rul-note").textContent = idx === n - 1 ? "En el último ciclo observado." : "Con los datos hasta este ciclo.";

  const real = S.rulReal;
  $("real-card").hidden = !real;
  if (real) {
    $("real").textContent = String(real[idx]);
    const e = rul - real[idx];
    $("real-note").textContent = `Error de la predicción: ${e >= 0 ? "+" : "−"}${fmt(Math.abs(e))} ciclos (${e >= 0 ? "sobreestima" : "subestima"}).`;
  }

  const warns = [];
  if (idx + 1 < info.min_history) warns.push(`Solo hay ${idx + 1} ciclos hasta este punto; el modelo usa ventanas de ${info.min_history}, así que la estimación es poco fiable.`);
  if (rul >= SATURATION) warns.push(`El modelo se entrenó con la RUL acotada a ${info.rul_cap} ciclos: indica motor saludable, pero no distingue valores mayores.`);
  $("warnings").innerHTML = warns.map((w) => `<li>${w}</li>`).join("");

  const series = [{ name: "RUL estimada", x: S.cycles, y: S.rul, color: "var(--pred)", width: 2.2 }];
  if (real) series.push({ name: `RUL real (acotada a ${info.rul_cap})`, x: S.cycles, y: real.map((v) => Math.min(v, info.rul_cap)), color: "var(--real)", dash: "6 4", width: 2, noDot: true });
  lineChart($("chart-rul"), {
    series, cursor: cyc, yMin: 0, yMax: Math.max(info.rul_cap + 15, ...S.rul),
    band: { name: "margen conservador", x: S.cycles, hi: S.rul, lo: S.rul.map((v) => Math.max(v - info.overestimate_p90, 0)) },
    xLabel: "ciclo", yLabel: "RUL (ciclos)",
  });

  const sensor = $("sensor").value;
  const raw = S.history[sensor];
  lineChart($("chart-sensor"), {
    series: [
      { name: sensor, x: S.cycles, y: raw, color: "var(--muted)", width: 1.2, noDot: true },
      { name: "media móvil de 5 ciclos", x: S.cycles, y: movingMean(raw, 5), color: "var(--pred)", width: 2.2 },
    ],
    cursor: cyc, xLabel: "ciclo", yLabel: sensor,
  });
}

function setResult(res, history) {
  S = { cycles: res.cycles, rul: res.rul, rulReal: res.rul_real || null, history };
  const n = res.cycles.length;
  $("cycle").max = n;
  $("cycle").value = n;
  $("cycle-hint").textContent = res.rul_real
    ? "Mueve el control para ver qué estimaría el modelo en cada momento de la vida del motor, usando solo los ciclos anteriores."
    : "Mueve el control para ver la estimación en ciclos anteriores, usando solo los datos hasta ese punto.";
  const cur = $("sensor").value;
  $("sensor").innerHTML = info.input_columns.map((c) => `<option>${c}</option>`).join("");
  $("sensor").value = info.input_columns.includes(cur) ? cur : "sensor4";
  $("results").hidden = false;
  render();
}

// ---------- demostración ----------
async function loadDemo() {
  showError("");
  try {
    const res = await api(`/api/demo/engines/${$("engine").value}`);
    setResult(res, res.history);
  } catch (e) { $("results").hidden = true; showError(e.message); }
}

// ---------- datos propios ----------
function parseText(txt) {
  const lines = txt.split(/\r?\n/).map((l) => l.trim()).filter(Boolean);
  if (!lines.length) throw new Error("Pega o sube primero el historial del motor.");
  const sep = lines[0].includes(",") ? /\s*,\s*/ : /\s+/;
  const first = lines[0].split(sep);
  const hasHeader = first.some((t) => t !== "" && Number.isNaN(Number(t)));
  let cols, body;
  if (hasHeader) {
    cols = first.map((s) => s.toLowerCase().replace(/^["']|["']$/g, ""));
    body = lines.slice(1);
  } else {
    if (first.length !== NASA_COLS.length) throw new Error(`Sin encabezado se esperan ${NASA_COLS.length} columnas (formato NASA) y se encontraron ${first.length}.`);
    cols = NASA_COLS;
    body = lines;
  }
  const missing = info.input_columns.filter((c) => !cols.includes(c));
  if (missing.length) throw new Error(`Faltan columnas: ${missing.join(", ")}.`);
  const want = [...info.input_columns, ...(cols.includes("cycle") ? ["cycle"] : [])];
  const unitIdx = cols.indexOf("unit");
  const groups = new Map();
  body.forEach((line, k) => {
    const t = line.split(sep);
    if (t.length !== cols.length) throw new Error(`La fila ${k + 1 + (hasHeader ? 1 : 0)} tiene ${t.length} valores y se esperaban ${cols.length}.`);
    const nums = t.map(Number);
    if (nums.some((v) => !Number.isFinite(v))) throw new Error(`La fila ${k + 1 + (hasHeader ? 1 : 0)} contiene valores no numéricos.`);
    const unit = unitIdx >= 0 ? nums[unitIdx] : 1;
    if (!groups.has(unit)) groups.set(unit, Object.fromEntries(want.map((c) => [c, []])));
    const g = groups.get(unit);
    for (const c of want) g[c].push(nums[cols.indexOf(c)]);
  });
  if (!groups.size) throw new Error("No hay filas de datos.");
  return groups;
}

async function runCustom(reparse = true) {
  showError("");
  $("run").disabled = true;
  try {
    if (reparse) {
      parsed = parseText($("raw").value);
      const units = [...parsed.keys()];
      $("unit-wrap").hidden = units.length < 2;
      $("unit").innerHTML = units.map((u) => `<option value="${u}">${u}</option>`).join("");
    }
    const data = parsed.get(Number($("unit").value || [...parsed.keys()][0]));
    const res = await api("/api/predict", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ data }) });
    // el backend ordena por ciclo; la gráfica de sensores debe seguir el mismo orden
    const order = data.cycle ? data.cycle.map((c, i) => [c, i]).sort((a, b) => a[0] - b[0]).map((p) => p[1]) : data[info.input_columns[0]].map((_, i) => i);
    const history = Object.fromEntries(info.input_columns.map((c) => [c, order.map((i) => data[c][i])]));
    setResult(res, history);
  } catch (e) { $("results").hidden = true; showError(e.message); }
  finally { $("run").disabled = false; }
}

async function loadExample() {
  showError("");
  try {
    const list = await api("/api/demo/engines");
    const pick = list.filter((e) => e.n_cycles >= 60).sort((a, b) => a.rul_real - b.rul_real)[0] || list[0];
    const res = await api(`/api/demo/engines/${pick.unit}`);
    const cols = ["cycle", ...info.input_columns];
    const rows = res.cycles.map((c, i) => [c, ...info.input_columns.map((k) => res.history[k][i])].join(","));
    $("raw").value = [cols.join(","), ...rows].join("\n");
  } catch (e) { showError(e.message); }
}

function setMode(next) {
  mode = next;
  $("tab-demo").classList.toggle("active", next === "demo");
  $("tab-custom").classList.toggle("active", next === "custom");
  $("tab-demo").setAttribute("aria-selected", next === "demo");
  $("tab-custom").setAttribute("aria-selected", next === "custom");
  $("pane-demo").hidden = next !== "demo";
  $("pane-custom").hidden = next !== "custom";
  $("results").hidden = true;
  showError("");
  if (next === "demo") loadDemo();
}

async function init() {
  try {
    info = await api("/api/info");
    const engines = await api("/api/demo/engines");
    $("engine").innerHTML = engines.map((e) => `<option value="${e.unit}">Motor ${e.unit} · ${e.n_cycles} ciclos observados</option>`).join("");
  } catch (e) { showError(`No se pudo conectar con el servicio: ${e.message}`); return; }
  $("tab-demo").onclick = () => setMode("demo");
  $("tab-custom").onclick = () => setMode("custom");
  $("engine").onchange = loadDemo;
  $("cycle").oninput = render;
  $("sensor").onchange = render;
  $("run").onclick = () => runCustom(true);
  $("unit").onchange = () => runCustom(false);
  $("example").onclick = loadExample;
  $("file").onchange = async (ev) => {
    const f = ev.target.files[0];
    if (!f) return;
    if (f.size > 5 * 1024 * 1024) { showError("El archivo supera los 5 MB."); return; }
    $("raw").value = await f.text();
  };
  loadDemo();
}

init();
