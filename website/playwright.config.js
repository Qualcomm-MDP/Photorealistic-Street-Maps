// @ts-check
// Smoke tests for the built documentation site (tests/). They run against the
// production build, served the way GitHub Pages serves it:
//   npm run build && npm run test:site
// See: https://playwright.dev/docs/test-configuration

const {defineConfig, devices} = require('@playwright/test');

const PORT = 3010;

module.exports = defineConfig({
  testDir: 'tests',
  timeout: 5 * 60 * 1000, // one test visits every page
  forbidOnly: !!process.env.CI,
  reporter: process.env.CI ? [['github'], ['list']] : 'list',
  use: {
    baseURL: `http://localhost:${PORT}`,
  },
  projects: [{name: 'chromium', use: {...devices['Desktop Chrome']}}],
  webServer: {
    command: `npm run serve -- --port ${PORT} --no-open`,
    url: `http://localhost:${PORT}/Photorealistic-Street-Maps/`,
    reuseExistingServer: !process.env.CI,
  },
});
