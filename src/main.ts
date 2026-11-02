import './style.css';
import tgpu from 'typegpu';
import { createGlitchPipeline } from './shader';

const MAX_DPR = 1.5;

const canvas = document.getElementById('glitch') as HTMLCanvasElement;
const freezeAt = new URLSearchParams(location.search).get('t'); // ?t=2.1 freezes the loop at that time
const reducedMotion = matchMedia('(prefers-reduced-motion: reduce)').matches || freezeAt !== null;

async function start() {
  if (!navigator.gpu) return fallback();

  let root: Awaited<ReturnType<typeof tgpu.init>>;
  try {
    root = await tgpu.init();
  } catch {
    return fallback();
  }

  const format = navigator.gpu.getPreferredCanvasFormat();
  const context = root.configureContext({ canvas, alphaMode: 'premultiplied' });
  const glitch = await createGlitchPipeline(root, format);

  const resize = () => {
    const dpr = Math.min(devicePixelRatio, MAX_DPR);
    canvas.width = Math.max(1, Math.round(canvas.clientWidth * dpr));
    canvas.height = Math.max(1, Math.round(canvas.clientHeight * dpr));
  };

  let running = false;
  let rafId = 0;
  let elapsed = freezeAt ? Number(freezeAt) : 0; // seconds of "visible" time, so the loop resumes where it paused
  let last = performance.now();

  function frame(now: number) {
    if (running) elapsed += Math.min((now - last) / 1000, 0.1);
    last = now;
    glitch.write(elapsed, canvas.width, canvas.height);
    glitch.pipeline.withColorAttachment({ view: context }).draw(3);
    if (running) rafId = requestAnimationFrame(frame);
  }

  new ResizeObserver(() => {
    resize();
    if (!running) frame(performance.now());
  }).observe(canvas);
  resize();

  const play = () => {
    if (running || reducedMotion) return;
    running = true;
    last = performance.now();
    rafId = requestAnimationFrame(frame);
  };
  const pause = () => {
    running = false;
    cancelAnimationFrame(rafId);
  };

  new IntersectionObserver(([e]) => (e.isIntersecting && !document.hidden ? play() : pause())).observe(canvas);
  document.addEventListener('visibilitychange', () => (document.hidden ? pause() : play()));

  frame(performance.now());
  play();

  import.meta.hot?.dispose(() => {
    pause();
    root.destroy();
  });
}

function fallback() {
  document.documentElement.classList.add('no-webgpu');
}

start();
