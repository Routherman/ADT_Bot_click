#!/usr/bin/env node
import puppeteer from 'puppeteer';

// Simple SERP fetcher using headless Chromium
// Usage: node puppeteer_search.js --engine=google --query="site:example.com contact" --results=100 --timeout=20000

function parseArgs() {
  const args = process.argv.slice(2);
  const out = {};
  for (const a of args) {
    const m = a.match(/^--([^=]+)=(.*)$/);
    if (m) out[m[1]] = m[2];
  }
  return out;
}

function buildUrl(engine, query, results) {
  const q = encodeURIComponent(query || '');
  if (engine === 'google') {
    const num = results || '100';
    return `https://www.google.com/search?q=${q}&num=${num}`;
  }
  if (engine === 'yahoo') {
    const n = results || '100';
    return `https://search.yahoo.com/search?p=${q}&n=${n}`;
  }
  throw new Error(`Unsupported engine: ${engine}`);
}

function extractLinks(engine) {
  const anchors = Array.from(document.querySelectorAll('a[href]'));
  const links = anchors.map(a => a.getAttribute('href')).filter(Boolean);
  // Normalize Google redirect links
  const out = [];
  for (let href of links) {
    if (engine === 'google' && href.startsWith('/url?q=')) {
      try {
        const u = new URL(href, location.origin);
        const q = u.searchParams.get('q');
        if (q) href = q;
      } catch {}
    }
    out.push(href);
  }
  return Array.from(new Set(out));
}

(async () => {
  const args = parseArgs();
  const engine = (args.engine || 'google').toLowerCase();
  const query = args.query || '';
  const results = args.results || '100';
  const timeout = Number(args.timeout || 20000);
  const headless = (process.env.PUPPETEER_HEADLESS || '1') !== '0';

  const url = buildUrl(engine, query, results);
  const browser = await puppeteer.launch({ headless });
  const page = await browser.newPage();
  await page.setUserAgent(
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36'
  );
  await page.setViewport({ width: 1366, height: 900 });
  await page.setRequestInterception(true);
  page.on('request', req => {
    const r = req.resourceType();
    // block heavy resources
    if (['image','media','font','stylesheet'].includes(r)) return req.abort();
    req.continue();
  });

  let html = '';
  try {
    await page.goto(url, { waitUntil: 'domcontentloaded', timeout });
    // Scroll a bit to trigger lazy loads
    await page.evaluate(async () => {
      await new Promise(r => {
        let y = 0;
        const id = setInterval(() => {
          window.scrollTo(0, y += 500);
          if (y > 3000) { clearInterval(id); r(); }
        }, 50);
      });
    });
    html = await page.content();
    const links = await page.evaluate(extractLinks, engine);

    console.log(JSON.stringify({ ok: true, engine, query, url, html, links }, null, 2));
  } catch (e) {
    console.log(JSON.stringify({ ok: false, engine, query, url, error: String(e) }, null, 2));
  } finally {
    await browser.close();
  }
})();
