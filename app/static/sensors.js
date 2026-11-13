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
  el.innerHTML = `<h4>Este sensor en el motor cargado</h4><div id="chart-sensor-info" class="chart"></div>`;
  lineChart($("chart-sensor-info"), {
    series: [
      { name: selSensor, x: S.cycles, y: raw, color: "var(--muted)", width: 1.2, noDot: true },
      { name: "media móvil de 5 ciclos", x: S.cycles, y: movingMean(raw, 5), color: "var(--pred)", width: 2.2 },
    ],
    xLabel: "ciclo", yLabel: selSensor,
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
const VIEWS = ["predictor", "sensores", "modelado"];
function showView() {
  const v = VIEWS.includes(location.hash.slice(1)) ? location.hash.slice(1) : "predictor";
  VIEWS.forEach((k) => {
    $(`view-${k}`).hidden = k !== v;
    const a = document.querySelector(`.nav a[href="#${k}"]`);
    a.classList.toggle("active", k === v);
    a.toggleAttribute("aria-current", k === v);
  });
  if (v === "sensores") drawLive();
}

initSensors();
window.addEventListener("hashchange", showView);
showView();
