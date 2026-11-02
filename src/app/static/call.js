"use strict";
const $ = id => document.getElementById(id);
let room = null, microphone = null, call = null, connecting = false, muted = false;
let latest = null, startedAt = 0, ending = false, heartbeatAt = 0;
let amplitude = 0, waveFrame = 0, generation = 0, soundMuted = false, captionsOn = true;
const wave = $("wave"), ctx = wave.getContext("2d");
const reducedMotion = matchMedia("(prefers-reduced-motion: reduce)").matches;
const phases = {connecting: "Conectando…", listening: "Llamada de audio", user_speaking: "Llamada de audio", decoding: "Llamada de audio", speaking: "Llamada de audio", ended: "Llamada finalizada", error: "Error de llamada", idle: "Llamada de audio"};

async function api(operation, payload) {
  const options = {cache: "no-store", signal: AbortSignal.timeout(10000)};
  if (payload !== undefined) Object.assign(options, {method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify(payload)});
  const response = await fetch(`/api/${operation}`, options);
  const result = await response.json();
  if (!response.ok) throw new Error(result.error || "No se pudo completar la petición.");
  return result;
}
function showError(text = "") { $("error").textContent = text; $("error").hidden = !text; }
function setPhase(status) { $("phase").textContent = phases[status] || phases.idle; }
function controls() {
  const occupied = latest?.calls?.active && !["ended", "error"].includes(latest.calls.active.status);
  const transmitting = latest?.stream && !["finished", "cancelled", "error"].includes(latest.stream.status);
  $("start-control").hidden = !!call;
  $("start").disabled = !latest?.ready || !latest?.calls?.available || occupied || transmitting || connecting || ending;
  $("start-label").textContent = connecting ? "Conectando…" : "Llamar";
  $("end-control").hidden = !call;
  $("call-controls").hidden = !call;
  $("hangup").disabled = ending;
  $("mute").disabled = connecting || ending || !microphone;
  $("sound").disabled = connecting || ending;
  $("timer").hidden = !startedAt;
  wave.hidden = !call;
}
function micError(error) {
  if (error.name === "NotAllowedError" || error.name === "PermissionDeniedError") return "Permite el acceso al micrófono en el navegador y vuelve a llamar.";
  if (error.name === "NotFoundError") return "No se encontró un micrófono.";
  if (error.name === "NotReadableError") return "El micrófono está ocupado por otra aplicación.";
  return error.message || "No se pudo conectar la llamada.";
}
async function start() {
  if (connecting || call) return;
  if (!window.isSecureContext || !navigator.mediaDevices?.getUserMedia) {
    showError("El micrófono requiere localhost o una conexión HTTPS."); return;
  }
  if (!window.LivekitClient) { showError("No se pudo cargar el audio. Recarga la página."); return; }
  const token = ++generation;
  connecting = true; showError(); controls(); setPhase("connecting"); $("transcript").textContent = "";
  try {
    microphone = await LivekitClient.createLocalAudioTrack({echoCancellation: true, noiseSuppression: true, autoGainControl: true});
    if (token !== generation) { microphone.stop(); microphone = null; return; }
    call = await api("calls/start", {});
    controls();
    room = new LivekitClient.Room({adaptiveStream: false, dynacast: false});
    const thisRoom = room;
    room.on(LivekitClient.RoomEvent.TrackSubscribed, (track, publication, participant) => {
      if (track.kind !== LivekitClient.Track.Kind.Audio || participant.identity !== `decoder-${call?.id}`) return;
      const element = track.attach(); element.autoplay = true; element.playsInline = true; element.muted = soundMuted;
      $("remote-audio").appendChild(element);
      element.play().catch(() => { $("audio-unlock").hidden = false; });
    });
    room.on(LivekitClient.RoomEvent.TrackUnsubscribed, track => { track.detach().forEach(element => element.remove()); });
    room.on(LivekitClient.RoomEvent.AudioPlaybackStatusChanged, () => { $("audio-unlock").hidden = room?.canPlaybackAudio !== false; });
    room.on(LivekitClient.RoomEvent.Reconnecting, () => { $("phase").textContent = "Reconectando…"; });
    room.on(LivekitClient.RoomEvent.Disconnected, () => {
      if (!ending && room === thisRoom) { showError("Se perdió la conexión de audio."); end(); }
    });
    await room.connect(call.url, call.token);
    if (token !== generation) return;
    await room.startAudio().catch(() => { $("audio-unlock").hidden = false; });
    await room.localParticipant.publishTrack(microphone);
    if (token !== generation) return;
    startedAt = performance.now(); heartbeatAt = 0; muted = false;
    $("mute").setAttribute("aria-pressed", "false"); $("mute").setAttribute("aria-label", "Silenciar micrófono"); $("mute-label").textContent = "Silenciar";
    setPhase("listening");
  } catch (error) { showError(micError(error)); await end(); }
  finally { connecting = false; controls(); }
}
async function end() {
  if (ending) return;
  ending = true; ++generation; connecting = false; controls();
  const closingCall = call, closingRoom = room;
  call = null; room = null;
  if (microphone) { microphone.stop(); microphone = null; }
  try {
    if (closingCall) await api("calls/end", {id: closingCall.id, control: closingCall.control});
  } catch (_) {
    if (closingCall) navigator.sendBeacon("/api/calls/end", new Blob([JSON.stringify({id: closingCall.id, control: closingCall.control})], {type: "application/json"}));
  }
  try { if (closingRoom) await closingRoom.disconnect(); }
  finally {
    $("remote-audio").replaceChildren(); $("audio-unlock").hidden = true;
    amplitude = 0; startedAt = 0; ending = false; setPhase("ended"); controls();
  }
}
async function toggleMute() {
  if (!microphone) return;
  try {
    if (muted) await microphone.unmute(); else await microphone.mute();
    muted = !muted; $("mute").setAttribute("aria-pressed", String(muted));
    $("mute").setAttribute("aria-label", muted ? "Activar micrófono" : "Silenciar micrófono");
    $("mute-label").textContent = muted ? "Activar micro" : "Silenciar";
  } catch (error) { showError(micError(error)); }
}
function toggleSound() {
  soundMuted = !soundMuted;
  $("remote-audio").querySelectorAll("audio").forEach(element => { element.muted = soundMuted; });
  $("sound").setAttribute("aria-pressed", String(soundMuted));
  $("sound").setAttribute("aria-label", soundMuted ? "Activar sonido" : "Silenciar sonido");
  $("sound-label").textContent = soundMuted ? "Sin sonido" : "Sonido";
}
function toggleCaptions() {
  captionsOn = !captionsOn; $("captions").hidden = !captionsOn;
  $("captions-toggle").classList.toggle("selected", captionsOn);
  $("captions-toggle").setAttribute("aria-pressed", String(captionsOn));
  $("captions-toggle").setAttribute("aria-label", captionsOn ? "Ocultar subtítulos" : "Mostrar subtítulos");
}
function transcript(flow, active) {
  if (!flow || flow.id !== active?.stream_id) return;
  const text = flow.text || "";
  if ($("transcript").textContent !== text) {
    $("transcript").textContent = text;
    $("captions").scrollTop = $("captions").scrollHeight;
  }
}
async function poll() {
  try {
    latest = await api("state");
    const available = latest.ready && latest.calls?.available;
    const active = latest.calls?.active;
    if (call && active?.id === call.id) {
      amplitude = muted ? 0 : Math.min(1, active.level * 14);
      if (!connecting && !ending) setPhase(active.status);
      transcript(latest.stream, active);
      if (["ended", "error"].includes(active.status)) {
        if (active.error) showError(active.error);
        await end();
      } else if (performance.now() - heartbeatAt > 10000) {
        await api("calls/ping", {id: call.id, control: call.control}); heartbeatAt = performance.now();
      }
    } else if (!call && !connecting && !ending) {
      const occupied = active && !["ended", "error"].includes(active.status);
      const transmitting = latest.stream && !["finished", "cancelled", "error"].includes(latest.stream.status);
      if (occupied || transmitting) $("phase").textContent = "Línea ocupada";
      else if (["Línea ocupada", "Conectando…", "Sin conexión"].includes($("phase").textContent)) setPhase("idle");
    }
    if (!available) showError(latest.calls?.error || latest.error || "No se pudo conectar con el servidor.");
    controls();
  } catch (_) {
    latest = null; amplitude = 0; controls();
    if (call) { showError("Se perdió la conexión con el servidor."); await end(); }
    else $("phase").textContent = "Sin conexión";
  }
  setTimeout(poll, 250);
}
function drawWave() {
  ctx.clearRect(0, 0, wave.width, wave.height);
  const speaking = call && latest?.calls?.active?.status === "speaking";
  const level = speaking ? Math.min(1, room?.remoteParticipants.get(`decoder-${call.id}`)?.audioLevel || 0) : amplitude;
  const phase = reducedMotion ? 0 : waveFrame++ * .11;
  for (let index = 0; index < 49; index++) {
    const envelope = Math.sin(Math.PI * index / 48);
    const height = 3 + level * 54 * envelope * (.35 + .65 * Math.abs(Math.sin(index * .71 + phase)));
    ctx.fillStyle = level > .04 ? "#e0effa" : "#b6c9d840";
    ctx.beginPath(); ctx.roundRect(25 + index * 12, (64 - height) / 2, 4, height, 2); ctx.fill();
  }
  if (startedAt) {
    const elapsed = Math.floor((performance.now() - startedAt) / 1000);
    $("timer").textContent = `${String(Math.floor(elapsed / 60)).padStart(2, "0")}:${String(elapsed % 60).padStart(2, "0")}`;
  }
  requestAnimationFrame(drawWave);
}
$("start").addEventListener("click", start);
$("hangup").addEventListener("click", end);
$("mute").addEventListener("click", toggleMute);
$("sound").addEventListener("click", toggleSound);
$("captions-toggle").addEventListener("click", toggleCaptions);
$("audio-unlock").addEventListener("click", async () => { if (room) { await room.startAudio(); $("audio-unlock").hidden = !room.canPlaybackAudio; } });
window.addEventListener("pagehide", () => {
  if (call) navigator.sendBeacon("/api/calls/end", new Blob([JSON.stringify({id: call.id, control: call.control})], {type: "application/json"}));
  if (microphone) microphone.stop();
  if (room) room.disconnect();
});
controls(); drawWave(); poll();
