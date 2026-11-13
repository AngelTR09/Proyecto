"use strict";

// Catálogo de los 14 sensores que usa el modelo (NASA C-MAPSS FD001).
// Las posiciones (x, y) son del esquema SVG de la sección "Sensores".
const GROUPS = {
  temp:  { label: "Temperatura", color: "#c0392b" },
  pres:  { label: "Presión", color: "#1f6fb5" },
  speed: { label: "Velocidad", color: "#1d7a46" },
  flow:  { label: "Flujo y relaciones", color: "#8a5a00" },
};

const SENSOR_INFO = [
  { id: "sensor2", n: 2, symbol: "T24", name: "Temperatura total a la salida del compresor de baja presión (LPC)", unit: "°R (Rankine)", group: "temp", x: 200, y: 85,
    station: "Salida del LPC",
    what: "Qué tan caliente sale el aire después de la primera etapa de compresión.",
    why: "Si el compresor se ensucia o se erosiona, necesita más trabajo para comprimir el aire y la temperatura de salida sube.",
    trend: "up", search: "turbofan low pressure compressor LPC cutaway" },
  { id: "sensor3", n: 3, symbol: "T30", name: "Temperatura total a la salida del compresor de alta presión (HPC)", unit: "°R", group: "temp", x: 320, y: 88,
    station: "Salida del HPC",
    what: "Temperatura del aire justo antes de entrar a la cámara de combustión.",
    why: "El HPC es el componente que falla en FD001. Cuando se degrada, su eficiencia cae y el aire sale más caliente.",
    trend: "up", search: "high pressure compressor HPC turbofan engine" },
  { id: "sensor4", n: 4, symbol: "T50", name: "Temperatura total a la salida de la turbina de baja presión (LPT)", unit: "°R", group: "temp", x: 572, y: 130,
    station: "Salida de la LPT",
    what: "Temperatura de los gases al final de la turbina, antes de la tobera.",
    why: "Es el indicador clásico de cuánto trabaja el motor: un núcleo menos eficiente quema más para dar el mismo empuje y los gases salen más calientes (EGT).",
    trend: "up", search: "low pressure turbine LPT turbofan exhaust gas temperature EGT" },
  { id: "sensor7", n: 7, symbol: "P30", name: "Presión total a la salida del HPC", unit: "psia", group: "pres", x: 352, y: 88,
    station: "Salida del HPC",
    what: "Presión del aire comprimido que entra a la cámara de combustión.",
    why: "Un compresor degradado entrega menos presión para el mismo régimen de giro.",
    trend: "down", search: "compressor pressure ratio turbofan HPC outlet pressure" },
  { id: "sensor8", n: 8, symbol: "Nf", name: "Velocidad física del fan", unit: "rpm", group: "speed", x: 80, y: 50,
    station: "Fan (entrada)",
    what: "Revoluciones reales del fan, la gran hélice delantera que da la mayor parte del empuje.",
    why: "Para mantener el empuje con un núcleo degradado, el control ajusta el régimen del fan.",
    trend: "up", search: "turbofan engine fan blades N1 speed" },
  { id: "sensor9", n: 9, symbol: "Nc", name: "Velocidad física del núcleo", unit: "rpm", group: "speed", x: 255, y: 88,
    station: "Eje de alta presión",
    what: "Revoluciones reales del eje de alta presión (compresor HPC y turbina HPT).",
    why: "Sube cuando el núcleo debe girar más rápido para compensar la pérdida de eficiencia. En los motores de demostración sube en cerca de 8 de cada 10.",
    trend: "upw", search: "turbofan high pressure shaft N2 core speed" },
  { id: "sensor11", n: 11, symbol: "Ps30", name: "Presión estática a la salida del HPC", unit: "psia", group: "pres", x: 320, y: 172,
    station: "Salida del HPC",
    what: "Presión estática (sin el componente del movimiento del aire) medida en la salida del HPC.",
    why: "Es el sensor más correlacionado con el desgaste en FD001: el aire sale a distinta presión cuando el compresor pierde rendimiento.",
    trend: "up", search: "static pressure Ps30 compressor discharge pressure sensor" },
  { id: "sensor12", n: 12, symbol: "phi", name: "Relación flujo de combustible / Ps30", unit: "pps/psi", group: "flow", x: 378, y: 130,
    station: "Cámara de combustión",
    what: "Cuánto combustible se inyecta por cada unidad de presión del aire comprimido.",
    why: "Resume cómo trabaja el control de combustible frente a la presión disponible.",
    trend: "down", search: "turbofan combustion chamber fuel injectors annular combustor" },
  { id: "sensor13", n: 13, symbol: "NRf", name: "Velocidad corregida del fan", unit: "rpm", group: "speed", x: 80, y: 210,
    station: "Fan (entrada)",
    what: "Velocidad del fan normalizada por las condiciones del aire (temperatura y presión de entrada).",
    why: "Permite comparar regímenes entre vuelos a distinta altitud o clima.",
    trend: "up", search: "corrected fan speed NR turbofan" },
  { id: "sensor14", n: 14, symbol: "NRc", name: "Velocidad corregida del núcleo", unit: "rpm", group: "speed", x: 255, y: 172,
    station: "Eje de alta presión",
    what: "Velocidad del núcleo normalizada por las condiciones del aire de entrada.",
    why: "Igual que Nc, pero comparable entre condiciones de vuelo. Es una señal débil: en los motores de demostración sube en poco más de la mitad.",
    trend: "upw", search: "corrected core speed turbofan spool" },
  { id: "sensor15", n: 15, symbol: "BPR", name: "Relación de derivación (bypass ratio)", unit: "adimensional", group: "flow", x: 170, y: 48,
    station: "Conducto de derivación",
    what: "Cociente entre el aire que rodea el núcleo y el que pasa por él.",
    why: "Cambia cuando el núcleo ofrece más resistencia o rinde menos que el conducto exterior.",
    trend: "up", search: "turbofan bypass ratio diagram" },
  { id: "sensor17", n: 17, symbol: "htBleed", name: "Entalpía de la sangría de aire", unit: "adimensional (índice)", group: "flow", x: 352, y: 172,
    station: "Sangrado del compresor",
    what: "Energía del aire que se extrae del compresor para otros usos (refrigeración, presurización).",
    why: "Refleja el estado térmico del aire comprimido.",
    trend: "up", search: "engine bleed air system compressor bleed valve" },
  { id: "sensor20", n: 20, symbol: "W31", name: "Flujo de aire de enfriamiento de la turbina de alta presión (HPT)", unit: "lbm/s", group: "flow", x: 445, y: 88,
    station: "Turbina de alta presión",
    what: "Aire que se desvía del compresor para enfriar los álabes de la HPT.",
    why: "Los álabes soportan temperaturas superiores a su punto de fusión; el control ajusta este flujo según la carga térmica.",
    trend: "down", search: "high pressure turbine blade cooling holes HPT" },
  { id: "sensor21", n: 21, symbol: "W32", name: "Flujo de aire de enfriamiento de la turbina de baja presión (LPT)", unit: "lbm/s", group: "flow", x: 512, y: 88,
    station: "Turbina de baja presión",
    what: "Aire de refrigeración enviado a la LPT.",
    why: "Acompaña a W31 y cae de forma parecida cuando el motor se desgasta.",
    trend: "down", search: "low pressure turbine cooling air LPT blades" },
];

const TREND = { up: ["↑ Sube", "Tiende a subir al acercarse la falla."],
  upw: ["↑ Sube a veces", "Señal débil: en los motores de demostración sube en parte de ellos, no en todos."], down: ["↓ Baja", "Tiende a bajar al acercarse la falla."] };

// Importancia en el modelo desplegado: |SHAP| medio (ciclos de RUL) de cada variable, del componente XGBoost
// (outputs/shap_importancia.csv). rel0 = cambio respecto a la línea base, ewm = nivel suavizado, slope = pendiente.
const IMPORTANCE = {
  sensor2: { rel0: 0.25, ewm30: 1.54, slope30: 0.58 }, sensor3: { rel0: 0.06, ewm30: 3.01, slope30: 0.25 },
  sensor4: { rel0: 4.60, ewm30: 0.84, slope30: 1.23 }, sensor7: { rel0: 0.57, ewm30: 0.29, slope30: 1.04 },
  sensor8: { rel0: 0.80, ewm30: 0.25, slope30: 1.35 }, sensor9: { rel0: 2.36, ewm30: 0.61, slope30: 1.81 },
  sensor11: { rel0: 9.62, ewm30: 1.10, slope30: 2.27 }, sensor12: { rel0: 1.33, ewm30: 0.16, slope30: 1.53 },
  sensor13: { rel0: 0.70, ewm30: 0.35, slope30: 1.17 }, sensor14: { rel0: 1.46, ewm30: 0.39, slope30: 1.48 },
  sensor15: { rel0: 0.64, ewm30: 0.33, slope30: 0.54 }, sensor17: { rel0: 0.23, ewm30: 3.88, slope30: 0.35 },
  sensor20: { rel0: 0.44, ewm30: 0.21, slope30: 0.48 }, sensor21: { rel0: 0.38, ewm30: 0.13, slope30: 0.67 },
};
const KIND = {
  rel0: { name: "Cambio respecto a su línea base", short: "el cambio respecto a su línea base",
    why: "Cada motor sale de fábrica con lecturas algo distintas, así que el valor absoluto no dice cuánto se ha desgastado. Lo que delata el desgaste es cuánto se alejó de la media de sus propios primeros 10 ciclos." },
  ewm30: { name: "Nivel suavizado (30 ciclos)", short: "su nivel suavizado",
    why: "La lectura de un ciclo es ruidosa. Una media móvil exponencial de unos 30 ciclos quita el ruido y deja ver el nivel real, que es lo que informa del estado del componente." },
  slope30: { name: "Tendencia de los últimos 30 ciclos", short: "su tendencia reciente",
    why: "Mide qué tan rápido está cambiando el sensor ahora: dos motores con el mismo nivel pero distinta velocidad de cambio no están igual de cerca de la falla." },
};
const IMP_TOTAL = Object.values(IMPORTANCE).reduce((a, o) => a + o.rel0 + o.ewm30 + o.slope30, 0);
const impSum = (id) => { const o = IMPORTANCE[id]; return o.rel0 + o.ewm30 + o.slope30; };
const IMP_RANK = Object.keys(IMPORTANCE).sort((a, b) => impSum(b) - impSum(a));

// Mismas definiciones que app/features.py, para mostrar en el navegador lo que ve el modelo.
function baselineSeries(x) {
  const out = []; let sum = 0, base = 0;
  x.forEach((v, i) => { sum += v; const cum = sum / (i + 1); if (i === 9) base = cum; out.push(i < 10 ? cum : base); });
  return out;
}
function ewmSeries(x, span = 30) {
  const a = 2 / (span + 1); const out = [x[0]];
  for (let i = 1; i < x.length; i++) out.push(a * x[i] + (1 - a) * out[i - 1]);
  return out;
}

let selSensor = null;
let activeGroup = "all";

function engineSvg() {
  const markers = SENSOR_INFO.map((s) => `
    <g class="mk" data-id="${s.id}" tabindex="0" role="button" aria-label="Sensor ${s.n}, ${s.symbol}">
      <circle cx="${s.x}" cy="${s.y}" r="13" fill="${GROUPS[s.group].color}"/>
      <text x="${s.x}" y="${s.y + 4}" text-anchor="middle">${s.n}</text>
    </g>`).join("");
  return `<svg viewBox="0 0 800 270" role="img" aria-label="Esquema de un motor turbofán con la ubicación de los sensores">
    <path class="nacelle" d="M40,40 L690,40 Q735,40 745,80 L745,190 Q735,230 690,230 L40,230"/>
    <rect class="part fan" x="62" y="52" width="26" height="166" rx="4"/>
    <rect class="part" x="100" y="96" width="92" height="78" rx="3"/>
    <path class="part" d="M200,96 L340,108 L340,162 L200,174 Z"/>
    <rect class="part comb" x="346" y="104" width="86" height="62" rx="12"/>
    <rect class="part hot" x="440" y="100" width="40" height="70" rx="3"/>
    <rect class="part hot" x="490" y="90" width="64" height="90" rx="3"/>
    <path class="part" d="M570,104 L720,118 L720,152 L570,166 Z"/>
    <line class="shaft" x1="100" x2="560" y1="135" y2="135"/>
    <g class="lbl">
      <text x="75" y="248" text-anchor="middle">Fan</text>
      <text x="146" y="196" text-anchor="middle">LPC</text>
      <text x="270" y="196" text-anchor="middle">HPC</text>
      <text x="389" y="196" text-anchor="middle">Cámara</text>
      <text x="460" y="196" text-anchor="middle">HPT</text>
      <text x="522" y="206" text-anchor="middle">LPT</text>
      <text x="645" y="196" text-anchor="middle">Tobera</text>
      <text x="400" y="30" text-anchor="middle">Conducto de derivación (bypass)</text>
      <text x="40" y="260" >Aire →</text>
    </g>
    ${markers}
  </svg>`;
}

function trendTag(t) {
  return `<span class="trend ${t}">${TREND[t][0]}</span>`;
}

function renderSensorCards() {
  const list = SENSOR_INFO.filter((s) => activeGroup === "all" || s.group === activeGroup);
  $("sensor-cards").innerHTML = list.map((s) => `
    <button type="button" class="scard${selSensor === s.id ? " on" : ""}" data-id="${s.id}">
      <span class="dot" style="background:${GROUPS[s.group].color}">${s.n}</span>
      <span class="sc-txt"><b>${s.symbol}</b><small>${s.name}</small></span>
      ${trendTag(s.trend)}
    </button>`).join("");
}

function analysisBlock(s) {
  const o = IMPORTANCE[s.id], tot = impSum(s.id);
  const kinds = ["rel0", "ewm30", "slope30"];
  const top = kinds.reduce((a, k) => (o[k] > o[a] ? k : a), "rel0");
  const max = Math.max(...kinds.map((k) => o[k]));
  const bars = kinds.map((k) => `<li class="${k === top ? "main" : ""}"><span>${KIND[k].name}</span>
    <span class="bar"><i style="width:${(o[k] / max) * 100}%"></i></span><span class="val">${(o[k] / IMP_TOTAL * 100).toFixed(1).replace(".", ",")} %</span></li>`).join("");
  const rank = IMP_RANK.indexOf(s.id) + 1;
  return `<div class="analysis">
    <h4>Cómo lo analiza el modelo</h4>
    <p>El modelo no usa la lectura cruda. De este sensor calcula tres variables, y entre las tres su aporte a la predicción es el siguiente
    (porcentaje del total explicado por todas las variables):</p>
    <ul class="imp">${bars}</ul>
    <p class="muted"><b>${s.symbol}</b> aporta en conjunto el <b>${(tot / IMP_TOTAL * 100).toFixed(1).replace(".", ",")} %</b> de la explicación del modelo
    (puesto ${rank} de 14). Lo que más usa de él es <b>${KIND[top].short}</b>. ${KIND[top].why}</p>
  </div>`;
}

function renderSensorDetail() {
  const s = SENSOR_INFO.find((x) => x.id === selSensor);
  const el = $("sensor-detail");
  if (!s) { el.innerHTML = `<p class="hint">Elige un sensor en el esquema o en la lista para ver qué mide y por qué importa.</p>`; return; }
  const q = encodeURIComponent(s.search);
  el.innerHTML = `
    <div class="sd-head">
      <span class="dot lg" style="background:${GROUPS[s.group].color}">${s.n}</span>
      <div><h3>${s.symbol} · ${s.name}</h3>
      <p class="muted">Columna <code>${s.id}</code> · ${GROUPS[s.group].label} · unidad: ${s.unit} · ubicación: ${s.station}</p></div>
    </div>
    <div class="sd-body">
      <figure class="sd-fig">
        <img src="img/${s.id}.svg" alt="Imagen representativa de ${s.symbol}" onerror="this.closest('figure').classList.add('noimg')">
        <figcaption class="ph">Aquí va la imagen representativa.<br>Guárdala como <code>app/static/img/${s.id}.svg</code>.<br>
          <a href="https://www.google.com/search?tbm=isch&q=${q}" target="_blank" rel="noopener">Buscar: “${s.search}”</a></figcaption>
      </figure>
      <dl>
        <dt>Qué mide</dt><dd>${s.what}</dd>
        <dt>Por qué importa</dt><dd>${s.why}</dd>
        <dt>Comportamiento en FD001</dt><dd>${trendTag(s.trend)} ${TREND[s.trend][1]}</dd>
      </dl>
    </div>
    ${analysisBlock(s)}
    <div id="sensor-live"></div>`;
  drawLive();
}

// Si ya hay un motor cargado en "Predictor", muestra su curva real de este sensor.
function drawLive() {
  const el = $("sensor-live");
  if (!el || !selSensor) return;
  if (typeof S === "undefined" || !S || !S.history || !S.history[selSensor]) {
    el.innerHTML = `<p class="hint">Carga un motor en la pestaña “Predictor” para ver aquí cómo se comporta este sensor a lo largo de su vida.</p>`;
    return;
  }
  const raw = S.history[selSensor];
  const base = baselineSeries(raw), ewm = ewmSeries(raw), rel = raw.map((v, i) => v - base[i]);
  el.innerHTML = `<h4>Este sensor en el motor cargado: lo que lee y lo que ve el modelo</h4>
    <p class="hint">Arriba, la lectura cruda, su línea base (media de los primeros 10 ciclos) y su nivel suavizado. Abajo, el cambio respecto a la línea base: la variable que el modelo usa.</p>
    <div id="chart-sensor-info" class="chart"></div><div id="chart-sensor-rel" class="chart"></div>`;
  lineChart($("chart-sensor-info"), {
    series: [
      { name: "lectura", x: S.cycles, y: raw, color: "var(--muted)", width: 1.2, noDot: true },
      { name: "línea base", x: S.cycles, y: base, color: "var(--real)", width: 2, dash: "6 4", noDot: true },
      { name: "nivel suavizado", x: S.cycles, y: ewm, color: "var(--pred)", width: 2.2, noDot: true },
    ],
    xLabel: "ciclo", yLabel: selSensor,
  });
  lineChart($("chart-sensor-rel"), {
    series: [{ name: "cambio respecto a la línea base", x: S.cycles, y: rel, color: "var(--pred)", width: 2, noDot: true }],
    xLabel: "ciclo", yLabel: "lectura − línea base",
  });
}

function pickSensor(id) {
  selSensor = id;
  document.querySelectorAll("#engine-diagram .mk").forEach((g) => g.classList.toggle("on", g.dataset.id === id));
  renderSensorCards();
  renderSensorDetail();
}

function initSensors() {
  $("engine-diagram").innerHTML = engineSvg();
  $("sensor-legend").innerHTML = Object.entries(GROUPS).map(([k, g]) =>
    `<button type="button" class="chip" data-g="${k}"><i style="background:${g.color}"></i>${g.label}</button>`).join("");
  $("sensor-legend").insertAdjacentHTML("afterbegin", `<button type="button" class="chip on" data-g="all">Todos</button>`);
  $("sensor-legend").onclick = (e) => {
    const b = e.target.closest(".chip"); if (!b) return;
    activeGroup = b.dataset.g;
    document.querySelectorAll("#sensor-legend .chip").forEach((c) => c.classList.toggle("on", c === b));
    renderSensorCards();
  };
  $("engine-diagram").onclick = (e) => { const g = e.target.closest(".mk"); if (g) pickSensor(g.dataset.id); };
  $("engine-diagram").onkeydown = (e) => { const g = e.target.closest(".mk"); if (g && (e.key === "Enter" || e.key === " ")) { e.preventDefault(); pickSensor(g.dataset.id); } };
  $("sensor-cards").onclick = (e) => { const b = e.target.closest(".scard"); if (b) pickSensor(b.dataset.id); };
  pickSensor("sensor11");
}

// ---------- navegación entre secciones ----------
const VIEWS = ["sensores", "predictor", "modelado"];
function showView() {
  const v = VIEWS.includes(location.hash.slice(1)) ? location.hash.slice(1) : "sensores";
  VIEWS.forEach((k) => {
    $(`view-${k}`).hidden = k !== v;
    const a = document.querySelector(`.nav a[href="#${k}"]`);
    a.classList.toggle("active", k === v);
    a.toggleAttribute("aria-current", k === v);
  });
  if (v === "sensores") drawLive();
  window.scrollTo(0, 0);
}

initSensors();
window.addEventListener("hashchange", showView);
showView();
