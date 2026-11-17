// Cruise Lab viewer: pick a flight, then up to four channels to plot against
// the flight's normal twin (same seed, no anomaly), with the anomaly window shaded.
'use strict';

const $ = (s, el = document) => el.querySelector(s);
const api = (p) => fetch(p).then((r) => { if (!r.ok) throw new Error(p + ' ' + r.status); return r.json(); });
const store = {
  get(k) { try { return localStorage.getItem('cruiselab.' + k); } catch (e) { return null; } },
  set(k, v) { try { localStorage.setItem('cruiselab.' + k, v); } catch (e) { /* private mode */ } },
};

const S = { flights: [], catalog: {}, channels: [], chan: {}, meta: null, panels: [], series: {}, syncing: false };

const SUBSYSTEM_ORDER = ['Sensors (autopilot inputs)', 'Autopilot commands', 'Flight plan (selected)',
  'Flight controls (surfaces)', 'Engine 1', 'Engine 2', 'Engine 3', 'Engine 4', 'Attitude', 'Velocities & rates',
  'Position', 'Aerodynamics', 'Accelerations', 'Forces', 'Moments', 'Propulsion (total)', 'Flight path', 'Derived',
  'Fuel tanks', 'Mass & inertia', 'Atmosphere & wind', 'Aircraft systems', 'Initial conditions', 'Geometry constants',
  'Earth model', 'Simulator'];

// default four channels for each kind of anomaly target
const DEFAULTS = {
  ALT: ['sensor/ALT', 'sensor/VS', 'attitude/theta-deg', 'fcs/elevator-pos-deg'],
  CAS: ['sensor/CAS', 'velocities/mach', 'ap/throttle-cmd', 'propulsion/engine[0]/n1'],
  VS: ['sensor/VS', 'position/h-sl-ft', 'attitude/theta-deg', 'fcs/elevator-pos-deg'],
  PITCH: ['sensor/PITCH', 'ap/elevator-cmd', 'fcs/elevator-pos-deg', 'position/h-sl-ft'],
  ROLL: ['sensor/ROLL', 'ap/aileron-cmd', 'fcs/left-aileron-pos-deg', 'attitude/psi-deg'],
  HDG: ['sensor/HDG', 'attitude/phi-deg', 'ap/aileron-cmd', 'aero/beta-deg'],
  BETA: ['sensor/BETA', 'ap/rudder-cmd', 'fcs/rudder-pos-deg', 'attitude/psi-deg'],
  Q: ['sensor/Q', 'ap/elevator-cmd', 'attitude/theta-deg', 'position/h-sl-ft'],
  P: ['sensor/P', 'ap/aileron-cmd', 'attitude/phi-deg', 'attitude/psi-deg'],
  elevator: ['ap/elevator-cmd', 'fcs/elevator-cmd-norm', 'fcs/elevator-pos-deg', 'position/h-sl-ft'],
  aileron: ['ap/aileron-cmd', 'fcs/aileron-cmd-norm', 'fcs/left-aileron-pos-deg', 'attitude/phi-deg'],
  rudder: ['ap/rudder-cmd', 'fcs/rudder-cmd-norm', 'fcs/rudder-pos-deg', 'aero/beta-deg'],
  normal: ['sensor/ALT', 'sensor/CAS', 'sensor/HDG', 'propulsion/engine[0]/n1'],
};
function defaultsFor(meta) {
  const a = meta.anomaly;
  if (!a) return DEFAULTS.normal;
  if (a.engine) {
    const e = a.engine - 1;
    return [`sensor/TLA${a.engine}`, `propulsion/engine[${e}]/n1`, `propulsion/engine[${e}]/fuel-flow-rate-pps`, 'fcs/rudder-pos-deg'];
  }
  return DEFAULTS[a.variable] || DEFAULTS.normal;
}

function css(name) { return getComputedStyle(document.documentElement).getPropertyValue(name).trim(); }
function fmt(x, d = 2) { return x === null || x === undefined ? '–' : (+x).toFixed(d); }
function esc(s) { return String(s ?? '').replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c])); }

function flightLabel(f) {
  const a = f.anomaly;
  if (!a) return `${f.id} · normal${f.hard_normal ? ' (' + f.hard_normal.replace('_', ' ') + ')' : ''}`;
  const tgt = a.variable || `engine ${a.engine}`;
  return `${f.id} · ${a.group} · ${a.type} · ${tgt} · ${a.profile}`;
}

// ------------------------------------------------------------------ boot
async function boot() {
  const [m, ch] = await Promise.all([api('/api/manifest'), api('/api/channels')]);
  S.flights = m.flights; S.catalog = m.catalog; S.channels = ch;
  ch.forEach((c) => { S.chan[c.name] = c; });
  for (let i = 0; i < 4; i++) S.panels.push(makePanel(i));
  $('#kindFilter').value = store.get('kind') || 'all';
  $('#kindFilter').onchange = () => { store.set('kind', $('#kindFilter').value); fillFlights(); selectFlight($('#flightSel').value); };
  $('#flightSel').onchange = () => selectFlight($('#flightSel').value);
  $('#twinChk').onchange = () => S.panels.forEach(drawPanel);
  $('#constChk').onchange = () => S.panels.forEach(fillOptions);
  $('#ctxChk').onchange = renderEffects;
  fillFlights();
  const last = new URLSearchParams(location.search).get('flight') || store.get('flight');
  if (last && [...$('#flightSel').options].some((o) => o.value === last)) $('#flightSel').value = last;
  selectFlight($('#flightSel').value);
}

function fillFlights() {
  const kind = $('#kindFilter').value, sel = $('#flightSel'), prev = sel.value;
  sel.innerHTML = '';
  for (const g of ['normal', 'observation', 'control']) {
    if (kind !== 'all' && kind !== g) continue;
    const og = document.createElement('optgroup');
    og.label = { normal: 'Normal flights', observation: 'Observation anomalies (before the autopilot)', control: 'Control anomalies (after the autopilot)' }[g];
    S.flights.filter((f) => f.kind === g).forEach((f) => og.append(new Option(flightLabel(f), f.id)));
    if (og.children.length) sel.append(og);
  }
  if ([...sel.options].some((o) => o.value === prev)) sel.value = prev;
}

async function selectFlight(id) {
  if (!id) return;
  store.set('flight', id);
  S.meta = await api('/api/flight/' + encodeURIComponent(id));
  S.series = {};
  S.panels.forEach((p) => { p.xrange = undefined; });
  renderAnomaly(); renderScenario(); renderEffects();
  const def = defaultsFor(S.meta).map((c) => (S.chan[c] ? c : 'sensor/ALT'));
  S.panels.forEach((p, i) => { p.channel = def[i]; fillOptions(p); drawPanel(p); });
}

// ------------------------------------------------------------------ cards
function renderAnomaly() {
  const m = S.meta, a = m.anomaly, el = $('#anomCard');
  if (!a) {
    el.innerHTML = `<div class="cardhead"><span><span class="badge normal">normal</span> ${esc(m.id)}</span></div>
      <p class="desc">No anomaly injected. ${m.hard_normal ? `This is a <b>hard normal</b> flight (${esc(m.hard_normal.replace('_', ' '))}): unusual but healthy conditions that a detector must not flag.` : 'Use it to see natural variability: turbulence, wind, route turns, step climbs and Mach changes all happen in normal flights too.'}</p>`;
    return;
  }
  const tgt = a.variable ? a.variable : `engine ${a.engine}`;
  const mag = a.magnitude === null ? 'value frozen at onset' : `${fmt(a.magnitude, 4)} ${esc(a.unit)}`;
  const extra = a.bursts ? `<dt>bursts</dt><dd>${a.bursts.length} (20–60 s on, 30–120 s off)</dd>` :
    a.freq_hz ? `<dt>frequency</dt><dd>${a.freq_hz} Hz</dd>` : a.spikes ? `<dt>spikes</dt><dd>${a.spikes.length}</dd>` : '';
  el.innerHTML = `<div class="cardhead"><span><span class="badge ${a.group}">${a.group}</span> ${esc(m.id)} · <b>${esc(a.type)}</b> on <b>${esc(tgt)}</b></span></div>
    <dl class="kv">
      <dt>where</dt><dd>${a.group === 'observation' ? 'before the autopilot: corrupts what it <i>reads</i>' : 'after the autopilot: corrupts its <i>command</i> on the way to the aircraft'}</dd>
      <dt>magnitude</dt><dd>${mag}${a.magnitude !== null ? ` · sign ${a.sign > 0 ? '+' : '−'} · severity ${a.severity}` : ''}</dd>
      <dt>profile</dt><dd>${esc(a.profile)}</dd>
      <dt>window</dt><dd>${fmt(a.onset_s, 1)} s → ${fmt(a.end_s, 1)} s (duration ${fmt(a.duration_s, 1)} s = ${fmt(a.duration_s / 60, 1)} min)</dd>
      ${extra}
      <dt>injected at</dt><dd>${m.injection_channels.map((c) => `<span class="chan-link" data-c="${esc(c)}">${esc(c)}</span>`).join(', ')}</dd>
    </dl>
    <div class="formula">${esc(a.formula)}</div>
    <p class="desc">${esc(a.description)}</p>`;
  el.querySelectorAll('.chan-link').forEach((s) => { s.onclick = () => assign(0, s.dataset.c); });
}

function renderScenario() {
  const s = S.meta.scenario, el = $('#scenCard');
  const turb = { 0: 'calm', 3: 'light', 4: 'moderate' }[s.turb_severity] || s.turb_severity;
  const turns = s.turns.length ? s.turns.map(([t, o]) => `${Math.round(t)} s: ${o > 0 ? '+' : ''}${Math.round(o)}°`).join(', ') : 'none';
  el.innerHTML = `<div class="cardhead">Flight conditions <span class="muted small">seed ${s.seed}</span></div>
    <dl class="kv">
      <dt>cruise</dt><dd>${Math.round(s.cruise_alt)} ft · Mach ${fmt(s.cruise_mach, 3)} · heading ${Math.round(s.heading)}°</dd>
      <dt>duration</dt><dd>${fmt(s.duration_s / 60, 1)} min</dd>
      <dt>route turns</dt><dd>${turns}</dd>
      <dt>step climb</dt><dd>${s.step_climb ? `+${s.step_climb[1]} ft at ${Math.round(s.step_climb[0])} s` : 'none'}</dd>
      <dt>Mach change</dt><dd>${s.mach_change ? `${s.mach_change[1] > 0 ? '+' : ''}${fmt(s.mach_change[1], 3)} at ${Math.round(s.mach_change[0])} s` : 'none'}</dd>
      <dt>wind</dt><dd>${Math.round(s.wind_kts)} kt from ${Math.round(s.wind_from_deg)}° (veer ${Math.round(s.wind_veer_deg)}°)</dd>
      <dt>turbulence</dt><dd>${turb}</dd>
      <dt>ISA deviation</dt><dd>${fmt(s.isa_dev_c, 1)} °C</dd>
      <dt>payload</dt><dd>${Math.round(s.payload_lbs).toLocaleString()} lb</dd>
      <dt>AP gains</dt><dd>${s.gain_scale.map((g) => fmt(g, 2)).join(' · ')} (×nominal)</dd>
    </dl>`;
}

const ROLE_CLS = { 'injection point': 'r-inj', 'strong response': 'r-strong', 'moderate response': 'r-mod', 'weak response': 'r-weak', 'no visible effect': 'r-none' };

function renderEffects() {
  const el = $('#effList'), m = S.meta;
  if (!m.effects) { el.innerHTML = '<p class="muted">Normal flight: there is nothing to compare against. Pick an anomalous flight to see which channels respond.</p>'; return; }
  const ctx = $('#ctxChk').checked;
  const rows = Object.entries(m.effects)
    .filter(([c]) => S.chan[c] && (ctx || !S.chan[c].context))
    .sort((a, b) => (b[1].score ?? 1e9) - (a[1].score ?? 1e9)).slice(0, 60);
  el.innerHTML = `<p class="muted small" style="margin:0 0 4px">score = largest departure from the normal twin after onset ÷ the channel's normal spread. Click 1–4 to plot.</p>` +
    rows.map(([c, e]) => `<div class="eff"><span class="nm" title="${esc(c)}">${esc(c)}</span>
      <span class="sc"><span class="${ROLE_CLS[e.role]}">${e.role === 'injection point' ? 'injected' : e.role.split(' ')[0]}</span> ${e.score === null ? '' : fmt(e.score, 1)}</span>
      <span class="to">${[1, 2, 3, 4].map((k) => `<button data-c="${esc(c)}" data-p="${k - 1}">${k}</button>`).join('')}</span></div>`).join('');
  el.querySelectorAll('button').forEach((b) => { b.onclick = () => assign(+b.dataset.p, b.dataset.c); });
}

function assign(i, c) {
  const p = S.panels[i];
  p.channel = c; fillOptions(p); drawPanel(p);
  p.el.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

// ------------------------------------------------------------------ panels
function makePanel(i) {
  const el = $('#panelTpl').content.firstElementChild.cloneNode(true);
  $('#grid').append(el);
  const p = { i, el, channel: null };
  $('.pnum', el).textContent = i + 1;
  $('.search', el).oninput = () => fillOptions(p);
  $('.chan', el).onchange = () => { p.channel = $('.chan', el).value; drawPanel(p); };
  $('.truthChk', el).onchange = () => drawPanel(p);
  $('.engChk', el).onchange = () => drawPanel(p);
  return p;
}

function fillOptions(p) {
  const sel = $('.chan', p.el), q = $('.search', p.el).value.trim().toLowerCase();
  const all = $('#constChk').checked, eff = (S.meta && S.meta.effects) || {};
  const groups = {};
  for (const c of S.channels) {
    if (!all && !c.changing && c.name !== p.channel) continue;
    if (q && !(c.name.toLowerCase().includes(q) || (c.meaning || '').toLowerCase().includes(q)) && c.name !== p.channel) continue;
    (groups[c.subsystem] = groups[c.subsystem] || []).push(c);
  }
  sel.innerHTML = '';
  const rank = (g) => { const k = SUBSYSTEM_ORDER.indexOf(g); return k < 0 ? 999 : k; };
  const order = Object.keys(groups).sort((a, b) => rank(a) - rank(b));
  for (const g of order) {
    const og = document.createElement('optgroup');
    og.label = `${g} (${groups[g].length})`;
    for (const c of groups[g]) {
      const e = eff[c.name], mark = !e ? '' : e.role === 'injection point' ? '◆ ' : e.role === 'strong response' ? '● ' : '';
      og.append(new Option(`${mark}${c.name}${c.unit ? '  [' + c.unit + ']' : ''}`, c.name));
    }
    sel.append(og);
  }
  if (p.channel) sel.value = p.channel;
}

function siblings(c) {
  let m = c.match(/^sensor\/TLA(\d)$/);
  if (m) return [1, 2, 3, 4].filter((k) => k !== +m[1]).map((k) => `sensor/TLA${k}`);
  m = c.match(/^(.*(?:engine|throttle-cmd-norm|throttle-pos-norm))\[(\d)\](.*)$/);
  if (m) return [0, 1, 2, 3].filter((k) => k !== +m[2]).map((k) => `${m[1]}[${k}]${m[3]}`).filter((x) => S.chan[x]);
  return [];
}

// angles that wrap at 0/360: break the line instead of drawing a vertical jump
function unwrapGaps(name, y) {
  if (!y || !/(HDG|psi|track|course|heading)/i.test(name)) return y;
  const out = y.slice();
  for (let i = 1; i < y.length; i++) if (y[i] !== null && y[i - 1] !== null && Math.abs(y[i] - y[i - 1]) > 180) out[i] = null;
  return out;
}

async function getSeries(chans) {
  const id = S.meta.id, need = chans.filter((c) => !(c in S.series));
  if (need.length) {
    const r = await api(`/api/series/${encodeURIComponent(id)}?ch=${encodeURIComponent(need.join(','))}`);
    S.t = { t: r.t, label: r.label, twinT: r.twin && r.twin.t };
    need.forEach((c) => { S.series[c] = { y: unwrapGaps(c, r.series[c]), twin: r.twin ? unwrapGaps(c, r.twin.series[c]) : null }; });
  }
  return S.t;
}

function roleHtml(c) {
  const m = S.meta, a = m.anomaly, info = S.chan[c] || {};
  if (!a) return `<span class="muted">Normal flight: no anomaly. The trace shows natural variability only.</span>`;
  const e = (m.effects || {})[c];
  if (m.injection_channels.includes(c)) {
    return `<b class="r-inj">◆ Injection point</b>: <b>${esc(a.type)}</b> is applied here, ${fmt(a.onset_s, 0)}–${fmt(a.end_s, 0)} s
      (${fmt(a.duration_s, 0)} s, ${esc(a.profile)}). <span class="muted">${esc(a.formula)}</span>`;
  }
  if (!e) return `<b class="r-none">No visible effect</b> of <b>${esc(a.type)}</b> on this channel (departure from the normal twin &lt; 0.05× its normal spread).`;
  const vis = e.first_s === null ? 'only small departures' : `first visible ${fmt(e.first_s, 1)} s after onset`;
  const why = info.context ? ' <span class="muted">(environment / bookkeeping channel: it changes because the aircraft is somewhere else)</span>' : '';
  return `<b class="${ROLE_CLS[e.role]}">${esc(e.role[0].toUpperCase() + e.role.slice(1))}</b> to <b>${esc(a.type)}</b> on ${esc(a.variable || 'engine ' + a.engine)}:
    up to ${fmt(e.score, 1)}× the channel's normal spread, ${vis}.${why}`;
}

async function drawPanel(p) {
  const c = p.channel;
  if (!c || !S.meta) return;
  const isSensor = c.startsWith('sensor/'), sib = siblings(c);
  $('.optTruth', p.el).style.display = isSensor ? '' : 'none';
  $('.optEng', p.el).style.display = sib.length ? '' : 'none';
  const wantTruth = isSensor && $('.truthChk', p.el).checked, wantEng = sib.length && $('.engChk', p.el).checked;
  const chans = [c, ...(wantTruth ? ['truth:' + c] : []), ...(wantEng ? sib : [])];
  $('.role', p.el).innerHTML = roleHtml(c);
  const info = S.chan[c] || {};
  $('.meaning', p.el).textContent = [info.meaning, info.unit && `unit: ${info.unit}`, info.status].filter(Boolean).join(' · ');
  const T = await getSeries(chans);
  if (p.channel !== c) return;                       // user picked another channel meanwhile
  const a = S.meta.anomaly, traces = [];
  const line = (w, color, dash) => ({ width: w, color, dash: dash || 'solid' });
  if (wantEng) sib.forEach((s) => traces.push({ x: T.t, y: S.series[s].y, name: s.replace(/^.*?(engine\[\d\]|norm\[\d\]|TLA\d).*$/, '$1'), mode: 'lines', line: line(1, css('--s-other')), opacity: 0.55 }));
  if ($('#twinChk').checked && a && S.series[c].twin) {
    traces.push({ x: T.twinT, y: S.series[c].twin, name: 'normal twin (no anomaly)', mode: 'lines', line: line(1.3, css('--s-twin')) });
  }
  if (wantTruth) traces.push({ x: T.t, y: S.series['truth:' + c].y, name: 'true value (simulator)', mode: 'lines', line: line(1.3, css('--s-truth'), 'dash') });
  traces.push({ x: T.t, y: S.series[c].y, name: a ? 'this flight' : 'this flight (normal)', mode: 'lines', line: line(1.5, css('--s-flight')) });

  const shapes = [], annotations = [];
  if (a) {
    const spans = a.bursts || [[a.onset_s, a.end_s]];
    spans.forEach(([x0, x1]) => shapes.push({ type: 'rect', xref: 'x', yref: 'paper', x0, x1, y0: 0, y1: 1, fillcolor: css('--win'), line: { width: 0 }, layer: 'below' }));
    [a.onset_s, a.end_s].forEach((x) => shapes.push({ type: 'line', xref: 'x', yref: 'paper', x0: x, x1: x, y0: 0, y1: 1, line: { color: css('--win-line'), width: 1, dash: 'dot' } }));
    annotations.push({ x: a.onset_s, y: 1, xref: 'x', yref: 'paper', text: 'anomaly', showarrow: false, xanchor: 'left', yanchor: 'top', font: { size: 10, color: css('--win-line') } });
  }
  const text = css('--text2'), grid = css('--grid');
  const layout = {
    margin: { l: 58, r: 10, t: 6, b: 34 }, paper_bgcolor: 'rgba(0,0,0,0)', plot_bgcolor: 'rgba(0,0,0,0)',
    font: { family: 'system-ui, Segoe UI, sans-serif', size: 11, color: text },
    xaxis: { title: { text: 'time since start of cruise [s]', font: { size: 11 } }, gridcolor: grid, zeroline: false, range: p.xrange },
    yaxis: { title: { text: info.unit || '', font: { size: 11 } }, gridcolor: grid, zeroline: false },
    legend: { orientation: 'h', x: 0, y: 1.02, yanchor: 'bottom', font: { size: 10 } },
    hovermode: 'x unified', shapes, annotations, showlegend: true,
  };
  const div = $('.plot', p.el);
  await Plotly.react(div, traces, layout, { displaylogo: false, responsive: true, modeBarButtonsToRemove: ['lasso2d', 'select2d'] });
  if (!div._synced) {
    div._synced = true;
    div.on('plotly_relayout', (ev) => {
      if (S.syncing) return;
      const r = ev['xaxis.range[0]'] !== undefined ? [ev['xaxis.range[0]'], ev['xaxis.range[1]']] : ev['xaxis.autorange'] ? null : undefined;
      if (r === undefined) return;
      S.syncing = true;
      Promise.all(S.panels.filter((q) => q !== p).map((q) => { q.xrange = r || undefined; const d = $('.plot', q.el); return d.data ? Plotly.relayout(d, r ? { 'xaxis.range': r } : { 'xaxis.autorange': true }) : null; }))
        .finally(() => { S.syncing = false; });
      p.xrange = r || undefined;
    });
  }
}

boot().catch((e) => { document.body.insertAdjacentHTML('afterbegin', `<p style="color:#c0392b;padding:12px">Failed to load: ${esc(e.message)}</p>`); });
