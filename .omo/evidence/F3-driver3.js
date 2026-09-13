/* F3 QA driver v4 (canonical-final) — fsd-web-demo
 * Root cause of v1-v3 payment failures: PaymentSimulator's 1.5s processing
 * setTimeout resets selectedScenario to null; clicking the next scenario
 * inside that window loses the selection (Pay Now stays disabled).
 * FIX: settle-wait 2200ms after each payment completes before next click.
 * Also: per-action try/catch with labeled logging (no more opaque 30s timeouts),
 * WS frame-rate probe from Node (built-in WebSocket), full 90s + auto-stop.
 */
const path = require('path');
const fs = require('fs');
const { chromium } = require('/Users/robotjang/helper-app/node_modules/playwright');

const EVIDENCE = '/Users/robotjang/fsd-web-demo/.omo/evidence';
const FRAME_PATH = '/Users/robotjang/fsd-web-demo/backend/data/demo_frames/um_000000.png';
const FRONTEND_URL = 'http://localhost:5173/';
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

async function historyTotal() {
  const r = await fetch('http://localhost:8000/api/payment/history');
  return (await r.json()).total ?? 0;
}

async function wsFrameProbe() {
  // sample streaming rate: count frames + gaps over ~5s
  return new Promise((resolve) => {
    const out = { frames: 0, intervalsMs: [], errors: [] };
    const ws = new WebSocket('ws://localhost:8000/ws/stream');
    let last = null;
    const timer = setTimeout(() => {
      try { ws.close(); } catch {}
      resolve(out);
    }, 6000);
    ws.onopen = () => ws.send(JSON.stringify({ start: true }));
    ws.onmessage = (ev) => {
      out.frames++;
      const now = Date.now();
      if (last !== null) out.intervalsMs.push(now - last);
      last = now;
    };
    ws.onerror = (e) => { out.errors.push('ws-error'); };
  });
}

(async () => {
  const browser = await chromium.launch();
  const context = await browser.newContext({
    viewport: { width: 1440, height: 900 },
    recordVideo: { dir: path.join(EVIDENCE, 'video_canon'), size: { width: 1440, height: 900 } },
  });
  const page = await context.newPage();
  const results = { qa1: {}, qa2: {}, qa3: {}, qa3b: {}, qa4: {}, ws_probe: {}, steps: [], errors: [] };
  const log = (name, ok, extra) => results.steps.push({ name, ok, ...extra });
  const errs = (label, e) => results.errors.push(`${label}: ${e.message}`);

  page.on('console', (m) => { if (m.type() === 'error') results.errors.push('console: ' + m.text()); });
  page.on('pageerror', (e) => results.errors.push('pageerror: ' + e.message));

  try {
    // ---- qa2: streaming first (demo start → canvas frame + narration) ----
    await page.goto(FRONTEND_URL, { waitUntil: 'networkidle' });
    await page.waitForSelector('button:has-text("Start Demo")', { timeout: 15000 });
    await page.screenshot({ path: path.join(EVIDENCE, 'F3-shot-00-initial.png') });
    await page.click('button:has-text("Start Demo")');
    await page.waitForSelector('button:has-text("Stop")', { timeout: 10000 });

    const t0 = Date.now();
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
    }, { timeout: 25000 });
    const firstFrameMs = Date.now() - t0;
    const narrationOk = await page.waitForFunction(() => {
      const t = document.body.innerText;
      return t.includes('Situation:') && !t.includes('Waiting for AI analysis');
    }, { timeout: 25000 }).then(() => true).catch(() => false);
    results.qa2 = { first_frame_ms: firstFrameMs, narration_updated: narrationOk };
    log('qa2-streaming', narrationOk, { first_frame_ms: firstFrameMs });
    await page.screenshot({ path: path.join(EVIDENCE, 'F3-shot-01-streaming.png') });

    // ---- qa3: payments with settle-wait between scenarios ----
    const payResults = {};
    for (const label of ['Toll Gate', 'Parking', 'Fuel Station']) {
      try {
        const before = await historyTotal();
        await page.click(`button:has-text("${label}")`);
        await page.waitForFunction(() => {
          const pay = [...document.querySelectorAll('button')].find((b) => b.textContent.includes('Pay Now'));
          return pay && !pay.disabled;
        }, { timeout: 15000 });
        await page.click('button:has-text("Pay Now")');
        const deadline = Date.now() + 20000;
        let after = before;
        while (Date.now() < deadline && after <= before) {
          await sleep(400);
          after = await historyTotal();
        }
        payResults[label] = { before, after, ok: after === before + 1 };
        await sleep(2200); // let the 1.5s processing timer fully settle before next click
      } catch (e) {
        payResults[label] = { ok: false, error: e.message };
        errs('qa3-' + label, e);
      }
    }
    const hist = await (await fetch('http://localhost:8000/api/payment/history')).json();
    results.qa3 = { per_scenario: payResults, total: hist.total, scenarios: hist.payments.map((p) => p.scenario) };
    log('qa3-payments', Object.values(payResults).every((r) => r.ok), { total: hist.total });
    await page.screenshot({ path: path.join(EVIDENCE, 'F3-shot-02-payments.png') });

    // ---- qa3b: WS frame-rate probe from Node (while demo still runs) ----
    results.ws_probe = await wsFrameProbe();
    const intervals = results.ws_probe.intervalsMs;
    const avgMs = intervals.length ? intervals.reduce((a, b) => a + b, 0) / intervals.length : 0;
    results.ws_probe.avg_interval_ms = Math.round(avgMs);
    results.ws_probe.est_fps = intervals.length ? +(1000 / avgMs).toFixed(1) : 0;
    log('qa3b-ws-rate', results.ws_probe.frames > 5, { frames: results.ws_probe.frames, est_fps: results.ws_probe.est_fps });

    // ---- qa1: real KITTI REST inference ----
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
      detect: { status: det.status, count: det.json.count, classes: det.json.detections.map((d) => d.class) },
      lane: { status: lane.status, count: lane.json.count },
    };
    log('qa1-inference', det.status === 200 && lane.status === 200, { detect: det.json.count, lane: lane.json.count });

    // ---- qa4: full 90s + auto-stop (Node-side polling; no waitForFunction helper) ----
    const deadline = Date.now() + 95000;
    let timerReached = 0;
    while (Date.now() < deadline) {
      const t = await page.evaluate(() => document.body.innerText);
      const m = t.match(/(\d+)s \/ 90s/);
      if (m) { timerReached = Math.max(timerReached, parseInt(m[1], 10)); }
      if (timerReached >= 90) break;
      await sleep(500);
    }
    const autoStopped = await page.evaluate(() =>
      [...document.querySelectorAll('button')].some((b) => b.textContent.includes('Start Demo'))
    );
    const endText = await page.evaluate(() => document.body.innerText);
    results.qa4 = { timerReached, reached_90s: timerReached >= 90, auto_stopped: autoStopped, destination_shown: endText.includes('Destination') };
    log('qa4-90s', timerReached >= 90 && autoStopped && endText.includes('Destination'), { timerReached, auto_stopped: autoStopped });
    await page.screenshot({ path: path.join(EVIDENCE, 'F3-shot-03-complete.png') });
  } catch (e) {
    errs('fatal', e);
  }

  await sleep(500);
  await browser.close();
  fs.writeFileSync(path.join(EVIDENCE, 'F3-driver3-result.json'), JSON.stringify(results, null, 2));
  console.log(JSON.stringify(results, null, 2));
})();