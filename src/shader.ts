import { common, d, std, tgpu } from 'typegpu';

export const FRAME_URLS = [0, 1, 2, 3, 4, 5].map((i) => `/frames/f${i}.jpg`);
const LAYERS = FRAME_URLS.length;

const LOOP_SECONDS = 4.5;
const LOOP_FRAMES = 270;
const ROWS = 540;
const KEYFRAME_OFFSET = 5; // keyframe k was taken at frame 5 + 45 * k
const KEYFRAME_STEP = LOOP_FRAMES / LAYERS;

const CROP = d.vec4f(0.05, 0.065, 0.95, 0.985);
const CROP_ASPECT = ((CROP.z - CROP.x) * 1920) / ((CROP.w - CROP.y) * 1080);

const rect = (x0: number, y0: number, x1: number, y1: number) =>
  d.vec4f((x0 - 3) / 960, (y0 - 3) / 540, (x1 + 3) / 960, (y1 + 3) / 540);
const labelRects = tgpu.const(d.arrayOf(d.vec4f, 6), [
  rect(88, 104, 144, 122), 
  rect(144, 212, 242, 232), 
  rect(144, 352, 242, 372), 
  rect(714, 94, 812, 113), 
  rect(750, 270, 804, 290), 
  rect(660, 426, 758, 446), 
]);

const GHOST_SHIFT = 17 / 960;

const window = (frame: number, from: number, to: number): number => {
  'use gpu';
  return std.select(d.f32(0), d.f32(1), frame >= from && frame < to + 1);
};

const inRect = (p: d.v2f, r: d.v4f): number => {
  'use gpu';
  return std.step(r.x, p.x) * std.step(p.x, r.z) * std.step(r.y, p.y) * std.step(p.y, r.w);
};

export async function createGlitchPipeline(
  root: Awaited<ReturnType<typeof tgpu.init>>,
  format: GPUTextureFormat,
) {
  const bitmaps = await Promise.all(
    FRAME_URLS.map(async (u) => createImageBitmap(await (await fetch(u)).blob())),
  );

  const texture = root
    .createTexture({
      size: [bitmaps[0].width, bitmaps[0].height, LAYERS],
      format: 'rgba8unorm',
      dimension: '2d',
    })
    .$usage('sampled', 'render');
  texture.write(bitmaps, { fit: 'stretch' });
  bitmaps.forEach((b) => b.close());

  const view = texture.createView(d.texture2dArray(d.f32));
  const sampler = root.createSampler({
    magFilter: 'linear',
    minFilter: 'linear',
    addressModeU: 'clamp-to-edge',
    addressModeV: 'clamp-to-edge',
  });

  const shiftData = new Int8Array(await (await fetch('/shifts.bin')).arrayBuffer());
  const shifts = root.createReadonly(d.arrayOf(d.i32, LOOP_FRAMES * ROWS), Array.from(shiftData));

  const timeU = root.createUniform(d.f32);
  const aspectU = root.createUniform(d.f32);

  const loopFrame = (): number => {
    'use gpu';
    return std.fract(timeU.$ / d.f32(LOOP_SECONDS)) * d.f32(LOOP_FRAMES);
  };

  const background = (p: d.v2f): d.v3f => {
    'use gpu';
    const phase =
      std.fract((loopFrame() - d.f32(KEYFRAME_OFFSET) + d.f32(LOOP_FRAMES)) / d.f32(LOOP_FRAMES)) *
      d.f32(LAYERS);
    const row = std.clamp(d.i32(std.floor(p.y * d.f32(ROWS))), 0, ROWS - 1);
    const frame = std.clamp(d.i32(std.floor(loopFrame())), 0, LOOP_FRAMES - 1);
    // labels are not part of the shifting bar layer
    const rects = labelRects.$;
    const onLabel = std.min(
      d.f32(1),
      inRect(p, rects[0]) +
        inRect(p, rects[1]) +
        inRect(p, rects[2]) +
        inRect(p, rects[3]) +
        inRect(p, rects[4]) +
        inRect(p, rects[5]),
    );
    const shift = d.f32(shifts.$[frame * ROWS + row]) * (1 - onLabel);
    const q = d.vec2f(p.x + shift / 960, p.y);
    const i0 = d.i32(std.floor(phase));
    const i1 = (i0 + 1) % LAYERS;
    const a = std.fract(phase);
    const c0 = std.textureSample(view.$, sampler.$, q, i0).rgb;
    const c1 = std.textureSample(view.$, sampler.$, q, i1).rgb;
    return std.mix(c0, c1, a);
  };

  /** mirrored copy of the label's strip, shifted left, laid over the original */
  const ghost = (base: d.v3f, p: d.v2f, r: d.v4f, amount: number, lower: number): d.v3f => {
    'use gpu';
    const cx = (r.x + r.z) * 0.5;
    const rel = (p.y - r.y) / (r.w - r.y);
    const strip = std.select(
      std.step(rel, d.f32(0.45)),
      std.step(d.f32(0.55), rel),
      lower > 0.5,
    );
    const mirrored = background(d.vec2f(2 * cx - (p.x + d.f32(GHOST_SHIFT)), p.y));
    const m = amount * inRect(p, r) * strip;
    return std.mix(base, std.max(base, mirrored), m);
  };

  const fragment = tgpu.fragmentFn({
    in: { uv: d.vec2f },
    out: d.vec4f,
  })(({ uv }) => {
    'use gpu';
    const aspect = aspectU.$;
    const cropA = d.f32(CROP_ASPECT);
    const scale = d.vec2f(std.min(d.f32(1), aspect / cropA), std.min(d.f32(1), cropA / aspect));
    const c = uv.sub(0.5).mul(scale).add(0.5);
    const p = d.vec2f(CROP.x + c.x * (CROP.z - CROP.x), CROP.y + c.y * (CROP.w - CROP.y));

    const f = loopFrame();
    const rects = labelRects.$;

    const mirrorAll = window(f, 110, 115);

    const pm = d.vec2f(1 - p.x, p.y);
    const labelMask = std.min(
      d.f32(1),
      inRect(p, rects[0]) +
        inRect(p, rects[1]) +
        inRect(p, rects[2]) +
        inRect(p, rects[3]) +
        inRect(p, rects[4]) +
        inRect(p, rects[5]) +
        inRect(pm, rects[0]) +
        inRect(pm, rects[1]) +
        inRect(pm, rects[2]) +
        inRect(pm, rects[3]) +
        inRect(pm, rects[4]) +
        inRect(pm, rects[5]),
    );

    let col = background(p);
    col = std.mix(col, background(pm), mirrorAll * labelMask);

    // strip ghosts on specific labels
    const t1 = window(f, 103, 107);
    const t2 = window(f, 133, 135);
    const t3 = window(f, 160, 164);
    col = ghost(col, p, rects[0], t3, d.f32(0));
    col = ghost(col, p, rects[1], std.max(t1, t3), d.f32(0));
    col = ghost(col, p, rects[2], t2, d.f32(1));
    col = ghost(col, p, rects[3], std.max(window(f, 104, 107), t3), d.f32(0));

    return d.vec4f(col, 1);
  });

  const pipeline = root.createRenderPipeline({
    vertex: common.fullScreenTriangle,
    fragment,
    targets: { format },
  });

  return {
    pipeline,
    write(time: number, w: number, h: number) {
      timeU.write(time);
      aspectU.write(w / h);
    },
  };
}
