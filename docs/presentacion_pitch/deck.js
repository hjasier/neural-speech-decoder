/* Streaming mockup from a recording. Phonemes and words are illustrative, not model outputs. */
(() => {
  'use strict';

  const data = window.PITCH_DATA;
  const numberFormat = new Intl.NumberFormat('es-ES', { useGrouping: 'always' });
  const ns = 'http://www.w3.org/2000/svg';
  const reducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)');
  let motionOff = reducedMotion.matches;
  let frame = 0;
  let lastPlanReduced = null;
  let focusBeforeDialog = null;
  let appScenario = null;

  /** @param {string} name @param {Record<string, string|number>} attrs @returns {SVGElement} */
  const svgNode = (name, attrs) => {
    const node = document.createElementNS(ns, name);
    Object.entries(attrs).forEach(([key, value]) => node.setAttribute(key, String(value)));
    return node;
  };

  /** @param {SVGElement} plot @param {number} width @param {number} height @param {boolean} app @returns {void} */
  const drawSignal = (plot, width, height, app) => {
    const left = app ? 30 : 0;
    const plotWidth = width - left - (app ? 20 : 0);
    const rowHeight = app ? 23 : 36;
    const amplitude = app ? 19 : 31;
    if (app) {
      const defs = svgNode('defs', {});
      const clip = svgNode('clipPath', { id: 'signal-clip' });
      clip.append(svgNode('rect', { id: 'signal-clip-width', x: 0, y: 0, width: 0, height }));
      defs.append(clip);
      plot.append(defs);
    }
    data.record.traces.forEach((trace, index) => {
      const base = (app ? 19 : 24) + rowHeight * index;
      plot.append(svgNode('line', { x1: left, y1: base, x2: width, y2: base, class: 'wave-base' }));
      const d = trace.map(([x, y], point) =>
        `${point === 0 ? 'M' : 'L'}${(left + x * plotWidth).toFixed(2)},${(base - (y - .5) * amplitude).toFixed(2)}`
      ).join(' ');
      const path = svgNode('path', { d, class: 'neural-trace' });
      path.style.setProperty('--order', index);
      if (app) path.setAttribute('clip-path', 'url(#signal-clip)');
      plot.append(path);
    });
    if (app) {
      [0, .25, .5, .75, 1].forEach(fraction => {
        const label = svgNode('text', {
          x: left + fraction * plotWidth, y: height - 3,
          'text-anchor': fraction === 0 ? 'start' : fraction === 1 ? 'end' : 'middle', class: 'axis-label',
        });
        label.textContent = `${(data.record.duration * fraction).toFixed(1)} s`;
        plot.append(label);
      });
    }
  };

  drawSignal(document.querySelector('#problem-wave'), 340, 170, false);
  drawSignal(document.querySelector('#app-wave'), 1050, 128, true);
  document.querySelectorAll('.method-phones span,.paper-word span').forEach((item, index) => {
    const siblings = [...item.parentNode.children].filter(child => child.tagName === 'SPAN');
    item.style.setProperty('--order', siblings.indexOf(item));
  });

  const kept = new Set(Array.from({ length: 20 }, (_, i) => Math.floor(i * 88 / 20)));
  const grid = document.querySelector('.sample-grid');
  for (let index = 0; index < 88; index += 1) {
    const dot = document.createElement('i');
    dot.classList.toggle('kept', kept.has(index));
    dot.style.setProperty('--delay', `${(index % 11) * .025}s`);
    dot.setAttribute('aria-hidden', 'true');
    grid.append(dot);
  }
  document.querySelectorAll('.model-layer').forEach((layer, index) => {
    layer.firstElementChild.style.setProperty('--delay', `${index * .08}s`);
  });
  document.querySelector('.plan-slide').style.setProperty('--budget-fill',
    `${data.plan.estimatedMinutes / data.plan.trainingLimitMinutes * 100}%`);
  document.querySelector('#minute-count').textContent = String(Math.round(data.plan.estimatedMinutes));
  const phoneContainer = document.querySelector('#demo-phones');
  data.phonemes.forEach(phone => {
    const token = document.createElement('span');
    token.textContent = phone;
    phoneContainer.append(token);
  });
  const phoneTokens = [...phoneContainer.children];
  const wordTokens = [...document.querySelectorAll('.demo-word')];
  const referenceTokens = [...document.querySelectorAll('.reference-word')];
  const clipWidth = document.querySelector('#signal-clip-width');
  const playhead = document.querySelector('.received-playhead');
  const processingHead = document.querySelector('.processing-playhead');
  const pendingBand = document.querySelector('.pending-band');
  const pendingTime = document.querySelector('#pending-time');
  const queue = document.querySelector('.pending-queue');
  for (let index = 0; index < 24; index += 1) {
    const block = document.createElement('i');
    block.setAttribute('aria-hidden', 'true');
    queue.append(block);
  }
  const queueBlocks = [...queue.children];
  const streamClock = document.querySelector('#stream-clock');
  const streamStatus = document.querySelector('#stream-status');
  const appWindow = document.querySelector('.app-window');

  /** @returns {void} */
  const stopFrame = () => {
    window.cancelAnimationFrame(frame);
    frame = 0;
  };

  /** @param {HTMLElement} slide @returns {number} */
  const visibleStep = slide => Math.max(-1, ...[...slide.querySelectorAll('.fragment.visible')]
    .map(fragment => Number(fragment.dataset.fragmentIndex)));

  /** @param {number} received @param {number} decoded @param {number} elapsedSeconds @returns {void} */
  const paintDemo = (received, decoded = received, elapsedSeconds = received * data.record.duration) => {
    const active = Math.min(phoneTokens.length - 1, Math.floor(decoded * phoneTokens.length));
    clipWidth.setAttribute('width', String(30 + received * 1000));
    playhead.style.left = `${(30 + received * 1000) / 1050 * 100}%`;
    processingHead.style.left = `${(30 + decoded * 1000) / 1050 * 100}%`;
    pendingBand.style.left = `${(30 + decoded * 1000) / 1050 * 100}%`;
    pendingBand.style.width = `${Math.max(0, received - decoded) * 1000 / 1050 * 100}%`;
    const pendingSeconds = Math.max(0, received - decoded) * data.record.duration;
    pendingTime.textContent = pendingSeconds > .01 ? `${pendingSeconds.toFixed(1).replace('.', ',')} s pendientes` : 'Sin señal pendiente';
    const pendingBlocks = Math.ceil(Math.max(0, received - decoded) * queueBlocks.length);
    queueBlocks.forEach((block, index) => block.classList.toggle('queued', index < pendingBlocks));
    streamClock.textContent = `${elapsedSeconds.toFixed(1).replace('.', ',')} s`;
    streamStatus.textContent = received < 1 ? 'Recibiendo' : decoded < 1 ? 'Decodificando' : 'Terminada';
    if (appScenario === 'slow' && pendingSeconds > .1) {
      streamStatus.textContent = received < 1 ? 'Acumulando retraso' : 'Procesando lo pendiente';
    }
    appWindow.classList.toggle('stream-running', received < 1);
    phoneTokens.forEach((token, index) => {
      token.classList.toggle('phone-complete', index < active || decoded >= 1);
      token.classList.toggle('phone-active', index === active && decoded < 1 && decoded > 0);
      token.setAttribute('aria-hidden', String(index > active || decoded === 0));
    });
    wordTokens.forEach((token, index) => {
      const threshold = [2, 5, 12][index] / phoneTokens.length;
      token.classList.toggle('word-visible', decoded >= threshold);
      token.setAttribute('aria-hidden', String(decoded < threshold));
    });
    referenceTokens.forEach(token => {
      token.classList.toggle('reference-highlight', appScenario === 'error'
        && wordTokens[Number(token.dataset.word)].classList.contains('word-visible'));
    });
  };

  /** Replay received samples; scenario speeds and predictions are illustrative.
   * @param {'normal'|'slow'|'error'} scenario @returns {void} */
  const runDemo = (scenario = appScenario ?? 'normal') => {
    stopFrame();
    const slide = document.querySelector('.app-slide');
    appScenario = scenario;
    slide.classList.toggle('scenario-slow', scenario === 'slow');
    slide.classList.toggle('scenario-error', scenario === 'error');
    slide.classList.toggle('compare-on', scenario === 'error');
    slide.querySelector('.reference-row').setAttribute('aria-hidden', String(scenario !== 'error'));
    slide.querySelector('.app-queue-row').setAttribute('aria-hidden', String(scenario !== 'slow'));
    slide.querySelector('.app-phone-row').setAttribute('aria-hidden', String(scenario === 'slow'));
    document.querySelector('#demo-scenario-label').textContent = scenario === 'slow' ? 'Modelo demasiado lento'
      : scenario === 'error' ? 'No predice bien' : 'Tiempo real simulado';
    phoneTokens.forEach((token, index) => {
      token.textContent = scenario === 'error' ? ({ 3: 'IH', 4: 'S' }[index] ?? data.phonemes[index])
        : index === 5 ? 'P' : data.phonemes[index];
    });
    wordTokens.forEach((token, index) => {
      token.textContent = (scenario === 'error' ? ['Is', 'this', 'solution?'] : ['Is', 'that', 'pollution?'])[index];
      token.classList.toggle('mistake-word', scenario === 'error' && index > 0);
    });
    if (motionOff) {
      paintDemo(1, scenario === 'slow' ? 1 / 3 : 1);
      return;
    }
    paintDemo(0);
    const start = performance.now();
    const duration = data.record.duration * 1000;
    const illustratedDelay = 350;
    const processingFactor = scenario === 'slow' ? 3 : 1;
    const tick = time => {
      const elapsed = time - start;
      const received = Math.min(duration, Math.floor(elapsed / 20) * 20) / duration;
      const processedMillis = Math.floor(Math.max(0, elapsed - illustratedDelay) / processingFactor / 20) * 20;
      const decoded = Math.min(duration, processedMillis) / duration;
      paintDemo(received, decoded, elapsed / 1000);
      if (decoded < 1 && Reveal.getCurrentSlide() === slide) frame = requestAnimationFrame(tick);
    };
    frame = requestAnimationFrame(tick);
  };

  /** @param {boolean} reduced @param {boolean} animate @returns {void} */
  const planCounts = (reduced, animate) => {
    const count = document.querySelector('#trial-count');
    const units = document.querySelector('#unit-count');
    const targetTrials = reduced ? data.plan.selected : data.plan.train;
    const targetUnits = reduced ? data.plan.units : data.plan.fullUnits;
    const startTrials = Number(count.textContent.replaceAll('.', ''));
    const startUnits = Number(units.textContent.replaceAll('.', ''));
    stopFrame();
    if (!animate || motionOff) {
      count.textContent = numberFormat.format(targetTrials);
      units.textContent = numberFormat.format(targetUnits);
      return;
    }
    const started = performance.now();
    const tick = now => {
      const progress = Math.min(1, (now - started) / 950);
      const ease = 1 - (1 - progress) ** 3;
      count.textContent = numberFormat.format(Math.round(startTrials + (targetTrials - startTrials) * ease));
      units.textContent = numberFormat.format(Math.round(startUnits + (targetUnits - startUnits) * ease));
      if (progress < 1) frame = requestAnimationFrame(tick);
    };
    frame = requestAnimationFrame(tick);
  };

  /** @param {boolean} replay @returns {void} */
  const syncScene = (replay = false) => {
    const slide = Reveal.getCurrentSlide();
    const step = visibleStep(slide);
    const scene = slide.dataset.scene;
    if (scene === 'problem') {
      if (replay) {
        slide.classList.remove('decode-on');
        void slide.offsetWidth;
      }
      slide.classList.toggle('decode-on', step >= 0);
    }
    if (scene === 'plan') {
      const reduced = step >= 0;
      if (replay) {
        slide.classList.remove('plan-reduced', 'plan-timing');
        planCounts(false, false);
        void slide.offsetWidth;
      }
      slide.classList.toggle('plan-reduced', reduced);
      slide.classList.toggle('plan-timing', step >= 1);
      slide.querySelector('.time-plan').setAttribute('aria-hidden', String(step < 1));
      if (lastPlanReduced !== reduced || replay) planCounts(reduced, !motionOff);
      lastPlanReduced = reduced;
    }
    if (scene === 'app') {
      const scenario = step >= 1 ? 'error' : step >= 0 ? 'slow' : 'normal';
      if (replay || appScenario !== scenario) runDemo(scenario);
    }
    document.querySelectorAll('.section-nav button').forEach(button => {
      button.setAttribute('aria-current', String(Number(button.dataset.slide) === Reveal.getIndices().h));
    });
  };

  /** @returns {void} */
  const changeScene = () => {
    stopFrame();
    document.querySelector('.problem-slide').classList.remove('decode-on');
    lastPlanReduced = null;
    appScenario = null;
    syncScene(true);
  };

  /** @returns {void} */
  const applyMotion = () => {
    stopFrame();
    document.body.classList.toggle('motion-off', motionOff);
    const button = document.querySelector('#motion-toggle');
    button.setAttribute('aria-pressed', String(motionOff));
    button.setAttribute('aria-label', motionOff ? 'Activar animaciones' : 'Desactivar animaciones');
    Reveal.configure({ transition: motionOff ? 'none' : 'fade' });
    syncScene(true);
  };

  const dialog = document.querySelector('#info-dialog');
  /** @returns {void} */
  const openInfo = () => {
    focusBeforeDialog = document.activeElement;
    dialog.showModal();
    document.querySelector('#info-close').focus();
  };
  /** @returns {void} */
  const closeInfo = () => {
    dialog.close();
    if (focusBeforeDialog instanceof HTMLElement) focusBeforeDialog.focus();
  };
  document.querySelector('#info-open').addEventListener('click', openInfo);
  document.querySelector('#info-close').addEventListener('click', closeInfo);
  dialog.addEventListener('cancel', event => { event.preventDefault(); closeInfo(); });
  document.querySelector('#motion-toggle').addEventListener('click', () => { motionOff = !motionOff; applyMotion(); });
  document.querySelector('#replay-demo').addEventListener('click', () => runDemo());
  document.querySelectorAll('.section-nav button').forEach(button => {
    button.addEventListener('click', () => Reveal.slide(Number(button.dataset.slide), 0, -1));
  });
  reducedMotion.addEventListener('change', event => { motionOff = event.matches; applyMotion(); });
  document.addEventListener('visibilitychange', () => {
    if (document.hidden) {
      stopFrame();
      if (Reveal.getCurrentSlide()?.dataset.scene === 'app') paintDemo(1, appScenario === 'slow' ? 1 / 3 : 1);
    }
  });
  document.addEventListener('keydown', event => {
    if (dialog.open || event.ctrlKey || event.metaKey || event.altKey) return;
    if (event.key.toLowerCase() === 'r') { event.preventDefault(); syncScene(true); }
    if (event.key.toLowerCase() === 'm') { event.preventDefault(); motionOff = !motionOff; applyMotion(); }
    if (event.key.toLowerCase() === 'i') { event.preventDefault(); openInfo(); }
  });

  Reveal.initialize({
    width: 1600, height: 900, margin: 0, minScale: .1, maxScale: 2,
    center: false, hash: true, controls: true, controlsTutorial: false,
    controlsLayout: 'edges', controlsBackArrows: 'visible', progress: true,
    slideNumber: false, transition: motionOff ? 'none' : 'fade', transitionSpeed: 'fast',
    backgroundTransition: 'none', keyboard: true, touch: true,
    autoSlide: 0, navigationMode: 'linear', plugins: [RevealNotes],
    disableLayout: false, overview: true,
  }).then(() => {
    document.body.classList.toggle('motion-off', motionOff);
    document.querySelector('#motion-toggle').setAttribute('aria-pressed', String(motionOff));
    document.querySelector('#motion-toggle').setAttribute('aria-label', motionOff ? 'Activar animaciones' : 'Desactivar animaciones');
    Reveal.on('slidechanged', changeScene);
    Reveal.on('fragmentshown', () => syncScene());
    Reveal.on('fragmenthidden', () => syncScene());
    changeScene();
  });
})();
