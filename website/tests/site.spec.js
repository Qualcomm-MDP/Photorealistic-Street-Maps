// @ts-check
// Every page of the built site loads without errors, both when opened
// directly and when reached by in-app navigation (which is when Docusaurus
// loads each page's JavaScript chunk; a missing chunk shows up as
// "Loading chunk ... failed").

const {test, expect} = require('@playwright/test');

const BASE = '/Photorealistic-Street-Maps/';

/** Collect everything that went wrong on a page: script errors, console
 * errors, failed requests and HTTP errors from this site. */
function watch(page, origin) {
  const problems = [];
  page.on('pageerror', (e) => problems.push(`script error: ${e.message}`));
  page.on('console', (m) => {
    if (m.type() === 'error') problems.push(`console error: ${m.text()}`);
  });
  page.on('requestfailed', (r) =>
    problems.push(`request failed: ${r.url()} (${r.failure()?.errorText})`),
  );
  page.on('response', (r) => {
    if (r.url().startsWith(origin) && r.status() >= 400) {
      problems.push(`HTTP ${r.status()}: ${r.url()}`);
    }
  });
  return problems;
}

/** The site's own pages linked from this page: paths under BASE, without
 * #anchors, ?queries or files such as images. */
async function pageLinks(page) {
  const hrefs = await page.$$eval('a[href]', (as) => as.map((a) => a.href));
  const paths = new Set();
  for (const href of hrefs) {
    const url = new URL(href);
    if (url.origin !== new URL(page.url()).origin) continue;
    if (!url.pathname.startsWith(BASE)) continue;
    if (/\.[a-z0-9]+$/i.test(url.pathname)) continue; // a file, not a page
    paths.add(url.pathname);
  }
  return [...paths];
}

/** Every page reachable from the home page, by following links. */
async function crawl(page) {
  const seen = new Set([BASE]);
  const queue = [BASE];
  while (queue.length) {
    await page.goto(queue.shift());
    for (const path of await pageLinks(page)) {
      if (!seen.has(path)) {
        seen.add(path);
        queue.push(path);
      }
    }
  }
  return [...seen].sort();
}

async function expectRealPage(page, path) {
  await expect(page.locator('article h1').first(), `${path} has a title`).toBeVisible();
  await expect(page.getByText('Page Not Found'), `${path} exists`).toHaveCount(0);
}

let pages = [];

test.beforeAll(async ({browser}) => {
  const page = await browser.newPage();
  pages = await crawl(page);
  await page.close();
});

test('the site has the expected pages', () => {
  // A broken sidebar or a page that failed to build would shrink this.
  expect(pages.length).toBeGreaterThanOrEqual(15);
  for (const path of ['getting-started', 'web-app', 'configuration', 'api/web']) {
    expect(pages).toContain(BASE + path);
  }
});

test('every page loads without errors when opened directly', async ({page, baseURL}) => {
  const problems = watch(page, baseURL);
  for (const path of pages) {
    const before = problems.length;
    await page.goto(path, {waitUntil: 'networkidle'});
    await expectRealPage(page, path);
    expect(problems.slice(before), `problems on ${path}`).toEqual([]);
  }
});

test('every page loads without errors through in-app navigation', async ({page, baseURL}) => {
  const problems = watch(page, baseURL);
  await page.goto(BASE, {waitUntil: 'networkidle'});
  for (const path of pages) {
    const before = problems.length;
    // Navigate the way a click on a link does: Docusaurus's router handles
    // the history change and loads the page's chunk, without a page reload.
    await page.evaluate((to) => {
      window.history.pushState({}, '', to);
      window.dispatchEvent(new PopStateEvent('popstate'));
    }, path);
    await page.waitForLoadState('networkidle');
    await expect(page, `navigated to ${path}`).toHaveURL(new RegExp(`${path}$`));
    await expectRealPage(page, path);
    expect(problems.slice(before), `problems navigating to ${path}`).toEqual([]);
  }
});
