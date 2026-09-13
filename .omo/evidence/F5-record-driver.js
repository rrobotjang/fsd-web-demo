/* F5 Recording driver — fresh 90s demo run with REAL Qwen3-VL-4B narration.
 * The vLLM daemon is connected (bridge health_check=True), so ws_stream now
 * flows real narration. We record the full run to video_canon (source for
 * ffmpeg scene clips) and verify narration text is real (not mock/placeholder).
 */
const path = require('path');
const fs = require('fs');
const { chromium } = require('/Users/robotjang/helper-app/node_modules/playwright');

const EVIDENCE = '/Users/robotjang/fsd-web-demo/.omo/evidence';
const FRONTEND_URL = 'http://localhost:5173/';
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

(async () => {
  const browser = await chromium.launch();
  const context = await browser.newContext({
    viewport: { width: 1440, height: 900 },
    recordVideo: { dir: path.join(EVIDENCE, 'video_canon'), size: { width: 1440, height: 900 } },
  });
  const page = await context.newPage();
  const result = { ready: {}, narration: {}, qwen_scene: {}, timer: {}, errors: [] };
  page.on('console', (m) => { if (m.type() === 'error') result.errors.push('console: ' + m.text()); });
  page.on('pageerror', (e) => result.errors.push('pageerror: ' + e.message));

  try {
    await page.goto(FRONTEND_URL, { waitUntil: 'networkidle' });
    await page.waitForSelector('button:has-text("Start Demo")', { timeout: 15000 });

    // ---- capture one narration from the WS as proof the real model speaks ----
    const narrationSamples = [];
    await page.click('button:has-text("Start Demo")');
    await page.waitForSelector('button:has-text("Stop")', { timeout: 10000 });

    // watch the narration panel text for up to 60s; collect distinct non-waiting samples
    const collectStart = Date.now();
    while (Date.now() - collectStart < 60000) {
      const t = await page.evaluate(() => document.body.innerText).catch(() => '');
      const m = t.match(/Situation:\s*([^\n]+)/);
      if (m && m[1]) {
        const s = m[1].trim();
        if (!s.includes('Waiting for AI analysis')) {
          if (!narrationSamples.includes(s)) narrationSamples.push(s);
        }
      }
      if (narrationSamples.length >= 5) break;
      await sleep(1500);
    }
    result.narration = {
      samples: narrationSamples,
      distinct_count: narrationSamples.length,
    };

    // ---- full 90s run to completion ----
    const deadline = Date.now() + 95000;
    let timerReached = 0;
    while (Date.now() < deadline) {
      const t = await page.evaluate(() => document.body.innerText).catch(() => '');
      const m = t.match(/(\d+)s \/ 90s/);
      if (m) timerReached = Math.max(timerReached, parseInt(m[1], 10));
      if (timerReached >= 90) break;
      await sleep(400);
    }
    const autoStopped = await page.evaluate(() =>
      [...document.querySelectorAll('button')].some((b) => b.textContent.includes('Start Demo'))
    ).catch(() => false);
    const endText = await page.evaluate(() => document.body.innerText).catch(() => '');
    result.timer = { timerReached, reached_90s: timerReached >= 90, auto_stopped: autoStopped, destination_shown: endText.includes('Destination') };

    // capture one zoomed-in narration panel shot late in the run for clip QA
    await page.screenshot({ path: path.join(EVIDENCE, 'F5-shot-narration-live.png') });
  } catch (e) {
    result.errors.push('fatal: ' + e.message);
  }

  await sleep(800);
  await browser.close();
  if (page.video()) {
    const vpath = await page.video().path();
    result.video_path = vpath;
  }
  fs.writeFileSync(path.join(EVIDENCE, 'F5-record-result.json'), JSON.stringify(result, null, 2));
  console.log(JSON.stringify(result, null, 2));
})();