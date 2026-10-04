/* Visual examples illustrate the method; no model predictions are generated here. */
(() => {
  'use strict';

  const data = window.T12_DATA;
  const reducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)');
  let motionPaused = reducedMotion.matches;
  let timers = [];
  let countFrame = 0;
  let dialogFocus = null;
  const keyboard = document.querySelector('.virtual-keyboard');
  const keys = new Map();
  const writing = document.querySelector('.writing-path');
  const writingLength = writing.getTotalLength();
  writing.style.strokeDasharray = writingLength;
  writing.style.strokeDashoffset = writingLength;

  for (const letter of 'ABCDEFGHIJKLMNOPQRSTUVWXYZ·') {
    const key = document.createElement('span');
    key.textContent = letter;
    keyboard.append(key);
    keys.set(letter, key);
  }

  data.sessions.forEach((session, index) => {
    const day = document.createElement('span');
    day.className = `day-dot${[0, 11, 23].includes(index) ? ' selected' : ''}`;
    day.textContent = String(index + 1);
    day.title = session.date;
    day.style.setProperty('--delay', `${index * 0.045}s`);
    document.querySelector('.day-rail').append(day);
  });

  const ns = 'http://www.w3.org/2000/svg';
  const svgNode = (name, attributes) => {
    const node = document.createElementNS(ns, name);
    Object.entries(attributes).forEach(([key, value]) => node.setAttribute(key, String(value)));
    return node;
  };
  document.querySelectorAll('[data-traces]').forEach(plot => {
    const record = data.records[Number(plot.dataset.traces)];
    const colors = ['#6de4ea', '#8eb1ff', '#f0a6b8', '#a6cbe8'];
    for (let tick = 0; tick <= 6; tick += 1) {
      const x = 53 + tick * 100;
      plot.append(svgNode('line', { x1: x, y1: 7, x2: x, y2: 205, class: 'grid-line' }));
      const label = svgNode('text', { x, y: 229, 'text-anchor': 'middle' });
      label.textContent = `${(record.duration * tick / 6).toFixed(1)} s`;
      plot.append(label);
    }
    record.traces.forEach((trace, index) => {
      const base = 45 + index * 51;
      const label = svgNode('text', { x: 4, y: base + 3 });
      label.textContent = String([9, 31, 64, 100][index]);
      plot.append(label);
      const points = trace.map(([x, y], point) =>
        `${point === 0 ? 'M' : 'L'}${(53 + x * 600).toFixed(2)},${(base - y * 36).toFixed(2)}`
      ).join(' ');
      plot.append(svgNode('path', { d: points, stroke: colors[index], class: 'trace' }));
    });
  });

  const picker = document.querySelector('#session-picker');
  const dateFormatter = new Intl.DateTimeFormat('es-ES', { day: '2-digit', month: 'short', year: 'numeric', timeZone: 'UTC' });
  data.records.forEach((record, index) => {
    const option = document.createElement('option');
    option.value = String(index);
    option.textContent = `${dateFormatter.format(new Date(record.date.replaceAll('/', '-') + 'T12:00:00Z'))} · ensayo ${record.trial + 1}`;
    picker.append(option);
  });
  const updateRecord = () => {
    const record = data.records[Number(picker.value)];
    const heatmap = document.querySelector('#app-heatmap');
    heatmap.src = `assets/${record.heatmap}`;
    heatmap.alt = `Actividad neuronal real de ${record.session}, ensayo ${record.trial + 1}`;
    document.querySelector('#reference-text').textContent = record.text;
    document.querySelector('#app-duration').textContent = `${record.duration.toFixed(2).replace('.', ',')} s`;
  };
  picker.addEventListener('change', updateRecord);
  updateRecord();

  const clearScene = () => {
    timers.forEach(timer => window.clearTimeout(timer));
    timers = [];
    window.cancelAnimationFrame(countFrame);
  };
  const later = (action, delay) => timers.push(window.setTimeout(action, delay));
  const moveCursor = letter => {
    const key = keys.get(letter);
    const scene = document.querySelector('.cursor-scene');
    const cursor = document.querySelector('.moving-cursor');
    // offsetTop of a key is measured from the nearest positioned scene.
    const keyBox = key.getBoundingClientRect();
    const sceneBox = scene.getBoundingClientRect();
    const scale = Reveal.getScale();
    cursor.style.transform = `translate(${(keyBox.left - sceneBox.left) / scale + key.offsetWidth * 0.58 - 12}px,${(keyBox.top - sceneBox.top) / scale + key.offsetHeight * 0.55 - 125}px)`;
    keys.forEach(node => node.classList.remove('selected'));
    key.classList.add('selected');
  };
  const animateScene = (slide, replay = false) => {
    clearScene();
    if (replay && !motionPaused) {
      slide.getAnimations({ subtree: true }).forEach(animation => {
        if (animation instanceof CSSAnimation) {
          animation.currentTime = 0;
          animation.play();
        }
      });
    }
    if (slide.dataset.scene === 'history') {
      const output = document.querySelector('#cursor-output');
      output.textContent = motionPaused ? 'HELLO' : '';
      keys.forEach(node => node.classList.remove('selected'));
      if (motionPaused) moveCursor('O');
      else [...'HELLO'].forEach((letter, index) => {
        later(() => moveCursor(letter), 400 + index * 650);
        later(() => { output.textContent = 'HELLO'.slice(0, index + 1); }, 750 + index * 650);
      });
    }
    if (slide.dataset.scene === 'speech') {
      const word = document.querySelector('#speech-word');
      word.textContent = motionPaused ? 'hello' : '';
      if (!motionPaused) [...'hello'].forEach((_, index) => {
        later(() => { word.textContent = 'hello'.slice(0, index + 1); }, 1250 + index * 190);
      });
    }
    if (slide.dataset.scene === 'results') {
      const number = slide.querySelector('[data-count]');
      number.textContent = motionPaused ? '62' : '0';
      if (!motionPaused) {
        const start = performance.now();
        const count = time => {
          const fraction = Math.min(1, (time - start) / 1250);
          number.textContent = String(Math.round(62 * (1 - (1 - fraction) ** 3)));
          if (fraction < 1) countFrame = requestAnimationFrame(count);
        };
        countFrame = requestAnimationFrame(count);
      }
    }
  };
  const updateFragments = () => {
    const ctc = document.querySelector('.ctc-collapse');
    document.querySelector('.phonemes').classList.toggle('collapsed', ctc.classList.contains('visible'));
  };
  const updateSlide = event => {
    const slide = event.currentSlide || Reveal.getCurrentSlide();
    document.querySelector('#slide-counter').textContent = `${Reveal.getIndices().h + 1} / ${Reveal.getTotalSlides()}`;
    updateFragments();
    animateScene(slide);
  };
  const setMotion = paused => {
    motionPaused = paused;
    document.body.classList.toggle('motion-paused', paused);
    Reveal.configure({ transition: paused ? 'none' : 'fade', autoAnimate: !paused });
    animateScene(Reveal.getCurrentSlide());
  };
  document.body.classList.toggle('motion-paused', motionPaused);

  const info = document.querySelector('#info-dialog');
  const viewer = document.querySelector('#image-dialog');
  const openDialog = dialog => {
    dialogFocus = document.activeElement;
    dialog.showModal();
  };
  document.querySelector('#info-open').addEventListener('click', () => openDialog(info));
  document.querySelectorAll('.image-zoom').forEach(link => link.addEventListener('click', event => {
    event.preventDefault();
    const expanded = document.querySelector('#expanded-image');
    expanded.src = link.href;
    expanded.alt = link.querySelector('img').alt;
    document.querySelector('#image-caption').textContent = link.dataset.caption;
    openDialog(viewer);
  }));
  document.querySelectorAll('[data-close]').forEach(button => button.addEventListener('click', () => {
    document.getElementById(button.dataset.close).close();
  }));
  [info, viewer].forEach(dialog => {
    dialog.addEventListener('click', event => { if (event.target === dialog) dialog.close(); });
    dialog.addEventListener('close', () => {
      dialogFocus?.focus();
    });
  });
  document.addEventListener('keydown', event => {
    if (info.open || viewer.open) event.stopPropagation();
  }, true);

  Reveal.initialize({
    width: 1280,
    height: 720,
    margin: 0,
    minScale: 0.1,
    maxScale: 2,
    center: false,
    scrollActivationWidth: null,
    hash: true,
    controls: true,
    controlsTutorial: false,
    controlsLayout: 'edges',
    progress: true,
    transition: motionPaused ? 'none' : 'fade',
    transitionSpeed: 'fast',
    backgroundTransition: 'fade',
    autoAnimate: !motionPaused,
    autoAnimateDuration: 0.8,
    slideNumber: false,
    keyboardCondition: () => !info.open && !viewer.open && document.activeElement?.tagName !== 'SELECT',
    plugins: [RevealNotes]
  }).then(() => {
    updateSlide({ currentSlide: Reveal.getCurrentSlide() });
    Reveal.on('slidechanged', updateSlide);
    Reveal.on('fragmentshown', updateFragments);
    Reveal.on('fragmenthidden', updateFragments);
    Reveal.addKeyBinding({ keyCode: 82, key: 'R', description: 'Repetir las animaciones de la diapositiva' }, () => animateScene(Reveal.getCurrentSlide(), true));
    Reveal.addKeyBinding({ keyCode: 77, key: 'M', description: 'Activar o pausar el movimiento' }, () => setMotion(!motionPaused));
    Reveal.addKeyBinding({ keyCode: 73, key: 'I', description: 'Abrir fuentes y controles' }, () => openDialog(info));
    reducedMotion.addEventListener('change', event => setMotion(event.matches));
  });
})();
