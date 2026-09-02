const { chromium } = require(process.env.PLAYWRIGHT_MODULE || '/opt/node22/lib/node_modules/playwright');
(async () => {
  const url = process.argv[2]; const out = process.argv[3];
  const browser = await chromium.launch({ headless: true, channel: 'chromium', proxy: { server: process.env.HTTPS_PROXY }, args: ['--disable-blink-features=AutomationControlled', '--ignore-certificate-errors', '--disable-features=PostQuantumKyber,UseMLKEM,PostQuantumKeyAgreement,EncryptedClientHello', '--ssl-version-max=tls1.2'] });
  const ctx = await browser.newContext({
    userAgent: 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36',
    viewport: { width: 1400, height: 1000 }, locale: 'en-US', ignoreHTTPSErrors: true,
  });
  const page = await ctx.newPage();
  const resp = await page.goto(url, { waitUntil: 'domcontentloaded', timeout: 90000 });
  console.log('status', resp && resp.status());
  for (let i = 0; i < 12; i++) {
    await page.waitForTimeout(2500);
    const title = await page.title();
    const n = await page.evaluate(() => document.querySelectorAll('.deck_segment, .deck-segment, [class*="deck"]').length);
    console.log('t', i, 'title=', title, 'deckish=', n);
    if (!/just a moment/i.test(title) && n > 0) break;
  }
  const html = await page.content();
  require('fs').writeFileSync(out, html);
  await page.screenshot({ path: out + '.png', fullPage: false });
  console.log('html bytes', html.length);
  await browser.close();
})().catch(e => { console.error('ERR', e.message); process.exit(1); });
