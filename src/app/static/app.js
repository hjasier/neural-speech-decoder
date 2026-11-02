"use strict";
const $ = id => document.getElementById(id);
const fmt = (value, digits = 2) => Number(value).toLocaleString("es-ES", {minimumFractionDigits: digits, maximumFractionDigits: digits});
const sleep = ms => new Promise(resolve => setTimeout(resolve, Math.max(0, ms)));
let sample = null, streamId = null, running = false, starting = false, loading = false;
let generation = 0, origin = 0, blockBins = 0, receivedBins = 0, latest = null, ready = false;
const canvas = $("signal"), context = canvas.getContext("2d");

async function api(operation, payload) {
  const options = payload === undefined ? {} : {method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify(payload)};
  const response = await fetch(`/api/${operation}`, options);
  const result = await response.json();
  if (!response.ok) {
    const error = new Error(result.error || "No se pudo completar la petición.");
    error.status = response.status;
    throw error;
  }
  return result;
}
function showError(message = "") { $("error").textContent = message; $("error").hidden = !message; }
function metric(id, value, unit, digits = 2) {
  $(id).innerHTML = `${value === null ? "—" : fmt(value, digits)} <small>${unit}</small>`;
}
function serverBusy() { return latest?.stream && !["finished", "cancelled", "error"].includes(latest.stream.status); }
function controls() {
  const busy = loading || starting || running || serverBusy() || latest?.call_active;
  $("load").disabled = !ready || busy;
  $("play").disabled = !ready || !sample || sample.features.length !== sample.bins || busy;
  $("stop").disabled = latest?.call_active || !(starting || running || serverBusy()) || latest?.stream?.status === "cancelling";
}
function resetView() {
  context.fillStyle = "#f5f9fc";
  context.fillRect(0, 0, canvas.width, canvas.height);
  $("signal-empty").hidden = false;
  $("cursor").hidden = true;
  $("received-progress").style.width = $("processed-progress").style.width = "0%";
  $("delivery-percent").textContent = "0 %";
  $("clock").textContent = "0,00 s";
  $("phase").textContent = "Preparada"; $("phase").className = "phase";
  $("transcript").textContent = "Las palabras aparecerán a medida que llegue la señal.";
  $("transcript").classList.add("empty");
  $("transcript-note").textContent = "El texto parcial puede cambiar al llegar nuevos bloques.";
  $("phones-box").hidden = true; $("phones").textContent = "";
  metric("received", 0, "s"); metric("processed", 0, "s"); metric("pending", 0, "ms", 0); metric("packet", null, "ms", 0);
}
function drawBlock(rows, start) {
  $("signal-empty").hidden = true;
  const width = canvas.width / sample.bins;
  rows.forEach((row, index) => {
    for (let channel = 0; channel < 128; channel++) {
      const z = (row[channel] - sample.view_mean[channel]) / sample.view_scale[channel];
      const amount = Math.min(Math.abs(z) / 3, 1);
      const color = z >= 0 ? [36, 89, 144] : [89, 181, 173];
      const baseline = [245, 249, 252];
      context.fillStyle = `rgb(${baseline.map((value, i) => Math.round(value + (color[i] - value) * amount)).join(",")})`;
      context.fillRect((start + index) * width, 127 - channel, Math.ceil(width), 1);
    }
  });
}
function animateClock(token) {
  if (!running || token !== generation) return;
  $("clock").textContent = `${fmt(Math.min(sample.bins * sample.bin_ms, performance.now() - origin) / 1000)} s`;
  requestAnimationFrame(() => animateClock(token));
}
async function loadSample() {
  loading = true; controls(); showError();
  $("load").textContent = "Cargando…";
  try {
    sample = await api("sample", {});
    streamId = null; receivedBins = 0; origin = 0;
    describeSample();
    resetView(); $("delivery-label").textContent = "Muestra preparada para transmitir";
  } catch (error) { showError(error.message); }
  finally { loading = false; $("load").textContent = sample ? "Otra muestra" : "Elegir muestra"; controls(); }
}
function describeSample() {
  $("sample-title").textContent = `Ensayo ${sample.trial_index} · ${fmt(sample.bins * sample.bin_ms / 1000)} s`;
  $("sample-info").textContent = `${sample.session} · validación · ${latest?.call_active ? "llamada en directo" : "muestra aleatoria"}`;
  $("duration").textContent = `${fmt(sample.bins * sample.bin_ms / 1000)} s`;
  $("reference").textContent = sample.reference;
  $("reference-box").hidden = false; $("reference-box").open = false;
}
async function synchronize(state) {
  if (loading || starting || running || !state.sample_id) return;
  const changed = sample?.id !== state.sample_id;
  const newStream = state.stream && state.stream.id !== streamId;
  if (!changed && !newStream && (!state.stream || state.stream.received_bins <= receivedBins)) return;
  const offset = changed || newStream ? 0 : receivedBins;
  const view = await api(`view?sample_id=${encodeURIComponent(state.sample_id)}&from=${offset}`);
  if (loading || starting || running) return;
  if (changed) {
    sample = {...view.sample, features: []};
    describeSample();
  }
  if (changed || newStream) { resetView(); receivedBins = 0; }
  streamId = state.stream?.id || null;
  if (view.features.length) {
    drawBlock(view.features, view.start);
    view.features.forEach((row, index) => { sample.features[view.start + index] = row; });
  }
  receivedBins = view.received_bins;
  const percent = 100 * receivedBins / sample.bins;
  $("received-progress").style.width = `${percent}%`;
  $("delivery-percent").textContent = `${fmt(percent, 0)} %`;
  $("clock").textContent = `${fmt(receivedBins * sample.bin_ms / 1000)} s`;
  $("delivery-label").textContent = state.call_active ? "Recibiendo la señal de la llamada" : "Siguiendo la transmisión del servidor";
}
async function sendWithRetry(operation, payload, token) {
  let networkFailures = 0;
  while (running && token === generation) {
    try { return await api(operation, payload); }
    catch (error) {
      if (error.status !== 429 && (error.status || ++networkFailures > 3)) throw error;
      // Un reintento conserva secuencia y contenido: el receptor no lo procesa dos veces.
      await sleep(120);
    }
  }
  return null;
}
async function transmit() {
  if (!sample || starting || running || serverBusy()) return;
  const token = ++generation;
  starting = true; streamId = null; controls(); showError(); resetView(); receivedBins = 0;
  try {
    const flow = await api("start", {sample_id: sample.id});
    if (token !== generation) { await api("cancel", {stream_id: flow.id}); return; }
    streamId = flow.id; starting = false; running = true; origin = performance.now(); controls();
    $("phase").textContent = "Parcial"; $("phase").className = "phase live";
    $("delivery-label").textContent = "Transmitiendo bloques por HTTP";
    animateClock(token);
    for (let start = 0, sequence = 0; start < sample.bins; start += blockBins, sequence++) {
      const stop = Math.min(start + blockBins, sample.bins);
      // Reloj absoluto de la señal: el cálculo del receptor no determina las llegadas.
      await sleep(origin + stop * sample.bin_ms - performance.now());
      if (!running || token !== generation) return;
      const rows = sample.features.slice(start, stop);
      const ack = await sendWithRetry("block", {stream_id: streamId, sequence, features: rows}, token);
      if (!ack || !running || token !== generation) return;
      drawBlock(rows, start); receivedBins = ack.received_bins;
      const percent = 100 * receivedBins / sample.bins;
      $("received-progress").style.width = `${percent}%`;
      $("delivery-percent").textContent = `${fmt(percent, 0)} %`;
    }
    $("delivery-label").textContent = "Señal enviada · terminando la decodificación";
    await sendWithRetry("finish", {stream_id: streamId}, token);
  } catch (error) {
    if (token === generation) {
      showError(error.message); running = false;
      if (streamId) { try { await api("cancel", {stream_id: streamId}); } catch (_) {} }
    }
  } finally { starting = false; controls(); }
}
async function stop() {
  ++generation; running = false; starting = false; controls();
  const target = serverBusy() ? latest.stream.id : streamId;
  if (!target) return;
  try { await api("cancel", {stream_id: target}); }
  catch (error) { showError(error.message); }
}
function render(flow) {
  if (!flow || flow.id !== streamId || !sample) return;
  metric("received", flow.received_bins * sample.bin_ms / 1000, "s");
  metric("processed", flow.processed_bins * sample.bin_ms / 1000, "s");
  metric("pending", flow.pending_ms, "ms", 0);
  metric("packet", flow.processed_bins ? flow.packet_ms : null, "ms", 0);
  $("processed-progress").style.width = `${100 * flow.processed_bins / sample.bins}%`;
  $("cursor").hidden = !flow.processed_bins;
  $("cursor").style.left = `${Math.min(100 * flow.processed_bins / sample.bins, 99.7)}%`;
  if (flow.text) {
    if ($("transcript").textContent !== flow.text) $("transcript").textContent = flow.text;
    $("transcript").classList.remove("empty");
  } else if (!["cancelled", "error"].includes(flow.status)) {
    $("transcript").textContent = "Esperando las primeras palabras…";
  }
  $("phones").textContent = flow.phones.join(" "); $("phones-box").hidden = !flow.phones.length;
  const labels = {receiving: "Parcial", finishing: "Terminando", cancelling: "Deteniendo", cancelled: "Interrumpida", finished: "Final", error: "Detenida"};
  $("phase").textContent = labels[flow.status] || flow.status;
  $("phase").className = `phase ${flow.status === "finished" ? "done" : ["receiving", "finishing"].includes(flow.status) ? "live" : ""}`;
  if (["finished", "cancelled", "error"].includes(flow.status)) {
    running = false;
    $("delivery-label").textContent = flow.status === "finished" ? "Transmisión completada" : "Transmisión detenida";
    if (flow.status === "finished") {
      $("clock").textContent = `${fmt(sample.bins * sample.bin_ms / 1000)} s`;
      $("transcript-note").textContent = "Frase finalizada. Puedes repetirla o elegir otra muestra.";
      if (!flow.text) $("transcript").textContent = "Sin palabras decodificadas.";
    }
    if (flow.error) showError(flow.error);
  }
}
async function poll() {
  try {
    latest = await api("state"); ready = latest.ready; blockBins = latest.block_bins;
    $("connection").textContent = `${latest.call_active ? "Llamada en directo · " : ""}CPU · ${latest.cpu_threads} hilos`;
    const dot = document.createElement("i"); $("connection").prepend(dot);
    $("connection").className = "connection connected";
    try { await synchronize(latest); } catch (error) { if (error.status !== 409) throw error; }
    render(latest.stream); if (latest.error) showError(latest.error);
  } catch (_) {
    ready = false; $("connection").textContent = "Servidor desconectado";
    $("connection").className = "connection";
    if (running) { running = false; ++generation; showError("Se perdió la conexión con el servidor."); }
  }
  controls(); setTimeout(poll, 250);
}
$("load").addEventListener("click", loadSample);
$("play").addEventListener("click", transmit);
$("stop").addEventListener("click", stop);
window.addEventListener("pagehide", () => {
  if (streamId && running) navigator.sendBeacon("/api/cancel", new Blob([JSON.stringify({stream_id: streamId})], {type: "application/json"}));
});
resetView(); poll();
