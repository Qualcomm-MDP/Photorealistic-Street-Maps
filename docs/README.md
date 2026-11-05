# PSM documentation

The documentation site, built with [Docusaurus](https://docusaurus.io/). Needs Node.js 20 or newer.

```bash
npm ci          # install
npm start       # live preview at http://localhost:3000/PSM/
npm run build   # production build into build/
npm run test:site   # open every page of the build; fails on errors or missing chunks
                    # (first time: npx playwright install chromium)
```

- Pages: `docs/` (Markdown). Sidebar: `sidebars.js`. Site settings: `docusaurus.config.js`.
- The API reference pages in `docs/api/` (except `web.md`) are generated from the docstrings in `src/`. Regenerate them from the repo root with `poetry run npm run api --prefix docs`.

See the Contributing page of the docs for more.
