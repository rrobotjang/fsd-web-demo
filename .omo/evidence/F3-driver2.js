/* F3 QA driver v2 (corrected) — fsd-web-demo
 * Fixes from v1:
 *  - payment sequencing: wait for Pay Now to complete (re-enable) before next scenario
 *  - qa1: probe /api/detect + /api/lane with a REAL KITTI frame (um_000000.png) loaded
 *    from backend/data/demo_frames (Node fs), not a synthetic canvas
 *  - qa4: after 90s, wait explicitly for "Start Demo" to reappear (auto-stop)
 * Records a fresh clean video → used as F3-qa-recording.mp4
 */
const path = require('path');
const fs = require('fs');
const { chromium } = require('/Users/robotjang/helper-app/node_modules/playwright');

const EVIDENCE = '/Users/robotjang/fsd-web-demo/.omo/evidence';
const FRAME_PATH = '/Users/robotjang/fsd-web-demo/backend/data/demo_frames/um_000000.png';
const FRONTEND_URL = 'http://localhost:5173/';

(async () => {
  const browser = await chromium.launch();
  const context = await browser.newContext({
    viewport: { width: 1440, height: 900 },
    recordVideo: { dir: path.join(EVIDENCE, 'video2'), size: { width: 1440, height: 900 } },
  });
  const page = await context.newPage();
  const results = { qa1: {}, qa2: {}, qa3: {}, qa4: {}, errors: [] };
  page.on('console', (m) => { if (m.type() === 'error') results.errors.push('console: ' + m.text()); });
  page.on('pageerror', (e) => results.errors.push('pageerror: ' + e.message));

  const payScenario = async (scenarioBtnText) => {
    await page.click(`button:has-text("${scenarioBtnText}")`);
    await page.waitForFunction(() => {
      const pay = [...document.querySelectorAll('button')].find((b) => b.textContent.includes('Pay Now'));
      return pay && !pay.disabled;
    }, { timeout: 8000 });
    await page.click('button:has-text("Pay Now")');
    await page.waitForFunction(() => {
      const pay = [...document.querySelectorAll('button')].find((b) => b.textContent.includes('Pay Now'));
      return pay && pay.disabled;
    }, { timeout: 8000 }).catch(() => {});
    await page.waitForFunction(() => {
      const pay = [...document.querySelectorAll('button')].find((b) => b.textContent.includes('Pay Now'));
      return pay && !pay.disabled;
    }, { timeout: 12000 });
  };

  try {
    await page.goto(FRONTEND_URL, { waitUntil: 'networkidle' });
    await page.waitForSelector('button:has-text("Start Demo")');
    await page.click('button:has-text("Start Demo")');

    // ---- qa2 streaming: real KITTI frame arrives on canvas ----
    await page.waitForFunction(() => {
      const c = document.querySelector('canvas');
      if (!c) return false;
      const ctx = c.getContext('2d');
      const d = ctx.getImageData(0, 0, c.width, c.height).data;
      let n = 0;
      for (let i = 0; i < d.length; i += 4) {
        if (d[i] + d[i + 1] + d[i + 2] > 40) { if (++n > 5000) break; }
      }
      return n > 5000;
    }, { timeout: 20000 });
    const narrationOk = await page.waitForFunction(() => {
      const t = document.body.innerText;
      return t.includes('Situation:') && !t.includes('Waiting for AI analysis');
    }, { timeout: 25000 }).then(() => true).catch(() => false);
    results.qa2 = {
      kitti_frame_rendered: true,
      narration_updated: narrationOk,
      canvas_present: true,
    };
    await page.screenshot({ path: path.join(EVIDENCE, 'F3-shot-01-streaming.png') });

    // ---- qa1: real KITTI REST inference (Node-side fetch, no browser CORS) ----
    const b64 = fs.readFileSync(FRAME_PATH).toString('base64');
    const api = async (ep, body) => {
      const r = await fetch('http://localhost:8000/api/' + ep, {
        method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body),
      });
      return { status: r.status, json: await r.json() };
    };
    const det = await api('detect', { image: b64 });
    const lane = await api('lane', { image: b64 });
    results.qa1 = {
      kitti_frame: FRAME_PATH.split('/').pop(),
      detect: {
        http_status: det.status,
        count: det.json.count,
        classes: det.json.detections.map((d) => d.class),
        message: det.json.message,
      },
      lane: { http_status: lane.status, count: lane.json.count, message: lane.json.message },
    };
    await page.screenshot({ path: path.join(EVIDENCE, 'F3-shot-02-kitti-infer.png') });

    // ---- qa3: payments, properly sequenced ----
    for (const label of ['Toll Gate', 'Parking', 'Fuel Station']) {
      await payScenario(label);
    }
    const hist = await page.evaluate(async () => {
      const r = await fetch('http://localhost:8000/api/payment/history');
      return await r.json();
    });
    results.qa3 = {
      all_three_completed: true,
      backend_total: hist.total,
      scenarios: (hist.payments || []).map((p) => p.scenario),
    };
    await page.screenshot({ path: path.join(EVIDENCE, 'F3-shot-03-payments.png') });

    // ---- qa4: full 90s auto-stop ----
    await page.waitForFunction(() => {
      const m = document.body.innerText.match(/(\d+)s \/ 90s/);
      return m && parseInt(m[1], 10) >= 90;
    }, { timeout: 70000 });
    await page.waitForSelector('button:has-text("Start Demo")', { timeout: 10000 });
    const endText = await page.innerText('body');
    results.qa4 = {
      reached_90s: true,
      auto_stopped: true,
      destination_phase_shown: endText.includes('Destination'),
    };
    await page.screenshot({ path: path.join(EVIDENCE, 'F3-shot-04-complete.png') });
  } catch (e) {
    results.errors.push('fatal: ' + e.message);
  }

  await page.waitForTimeout(500);
  await browser.close();
  console.log(JSON.stringify(results, null, 2));
})();