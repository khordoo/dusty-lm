// Render film.html to PNG frames with headless Chrome (via playwright-core),
// then encode an MP4 and a GIF with ffmpeg-static.
//
// Run:
//   PW_DIR=/path/to/temp/pw node analysis/opencode-film/render.cjs
//
// PW_DIR must contain node_modules/playwright-core and node_modules/ffmpeg-static.
// Playwright's bundled ffmpeg is VP8/WebM-only, so it cannot make the MP4.

const fs = require('fs');
const path = require('path');
const { spawnSync } = require('child_process');

const here = __dirname;
const pw = process.env.PW_DIR;
if (!pw) throw new Error('set PW_DIR to the directory that has node_modules/playwright-core');
const { chromium } = require(path.join(pw, 'node_modules', 'playwright-core'));

const FPS = 30;
const TOTAL = 12.0;
const N = Math.round(FPS * TOTAL);
const FRAMES = path.join(here, 'frames');

(async () => {
  fs.rmSync(FRAMES, { recursive: true, force: true });
  fs.mkdirSync(FRAMES, { recursive: true });

  const browser = await chromium.launch({
    executablePath: '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',
    headless: true,
    args: ['--hide-scrollbars', '--force-device-scale-factor=1'],
  });
  const page = await browser.newPage({ viewport: { width: 1920, height: 1080 }, deviceScaleFactor: 1 });
  await page.goto('file://' + path.join(here, 'film.html'));
  await page.waitForFunction('window.ready === true');

  for (let i = 0; i < N; i++) {
    await page.evaluate((t) => window.renderFrame(t), i / FPS);
    await page.screenshot({ path: path.join(FRAMES, `frame-${String(i).padStart(4, '0')}.png`) });
    if (i % 30 === 0) console.log(`frame ${i}/${N}`);
  }
  await browser.close();

  const ff = process.env.FFMPEG ||
    path.join(pw, 'node_modules', 'ffmpeg-static', 'ffmpeg');
  const mp4 = path.join(here, 'milo-rename.mp4');
  const gif = path.join(here, 'milo-rename.gif');

  console.log('encoding mp4 ...');
  let r = spawnSync(ff, [
    '-y', '-framerate', String(FPS), '-i', path.join(FRAMES, 'frame-%04d.png'),
    '-c:v', 'libx264', '-preset', 'medium', '-crf', '18',
    '-pix_fmt', 'yuv420p', '-movflags', '+faststart', mp4,
  ], { stdio: 'inherit' });
  if (r.status !== 0) process.exit(r.status || 1);

  console.log('encoding gif ...');
  r = spawnSync(ff, [
    '-y', '-i', mp4,
    '-vf', 'fps=15,scale=960:-1:flags=lanczos,split[s0][s1];[s0]palettegen=max_colors=128[p];[s1][p]paletteuse=dither=bayer',
    gif,
  ], { stdio: 'inherit' });
  if (r.status !== 0) process.exit(r.status || 1);

  console.log('wrote', mp4);
  console.log('wrote', gif);
})();
