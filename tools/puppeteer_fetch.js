#!/usr/bin/env node
import puppeteer from 'puppeteer';

// Generic page fetcher using headless Chromium
// Usage: node puppeteer_fetch.js --url=https://example.com --timeout=20000

function parseArgs() {
  const args = process.argv.slice(2);
  const out = {};
  for (const a of args) {
    const m = a.match(/^--([^=]+)=(.*)$/);
    if (m) out[m[1]] = m[2];
  }
  return out;
}

(async () => {
  const args = parseArgs();
  const url = args.url || '';
  const timeout = Number(args.timeout || 20000);
  const headless = (process.env.PUPPETEER_HEADLESS || '1') !== '0';
  if (!url) {
    console.log(JSON.stringify({ ok: false, error: 'Missing --url' }));
    process.exit(0);
  }

  const browser = await puppeteer.launch({ headless });
  try {
    const page = await browser.newPage();
    await page.setUserAgent(
      'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36'
    );
    await page.setViewport({ width: 1366, height: 900 });
    await page.setRequestInterception(true);
    page.on('request', req => {
      const r = req.resourceType();
      if (['image','media','font','stylesheet'].includes(r)) return req.abort();
      req.continue();
    });

    await page.goto(url, { waitUntil: 'domcontentloaded', timeout });
    // light scroll
    await page.evaluate(async () => {
      await new Promise(r => {
        let y = 0;
        const id = setInterval(() => {
          window.scrollTo(0, y += 400);
          if (y > 2500) { clearInterval(id); r(); }
        }, 50);
      });
    });
    const html = await page.content();
    const text = await page.evaluate(() => document.body ? document.body.innerText : '');
    const mailtos = await page.evaluate(() => Array.from(document.querySelectorAll('a[href^="mailto:"]')).map(a => (a.getAttribute('href') || '').replace(/^mailto:/i, '')));
    const cfemails = await page.evaluate(() => Array.from(document.querySelectorAll('span.__cf_email__')).map(s => s.getAttribute('data-cfemail')).filter(Boolean));
    const selectorList = [
      'a.email','a.staff-email','a.directory-email','a.contact-email',
      '.email','.staff-email','.directory-email','.contact-email',
      'td.email','li.email','span.email','p.email'
    ];
    const nodeTexts = await page.evaluate((sels) => {
      const out = [];
      for (const sel of sels) {
        for (const el of Array.from(document.querySelectorAll(sel))) {
          const t = el.innerText && el.innerText.trim();
          if (t) out.push(t);
        }
      }
      return Array.from(new Set(out));
    }, selectorList);
    console.log(JSON.stringify({ ok: true, url, html, text, mailtos, cfemails, nodeTexts }, null, 2));
  } catch (e) {
    console.log(JSON.stringify({ ok: false, url, error: String(e) }, null, 2));
  } finally {
    await browser.close();
  }
})();
