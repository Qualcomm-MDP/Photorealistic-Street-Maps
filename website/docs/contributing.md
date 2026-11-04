# Contributing

Contributions are welcome! This page covers setting up a development environment and the project's conventions.

## Development setup

```bash
git clone https://github.com/Qualcomm-MDP/PSM.git
cd PSM
poetry install
poetry run pre-commit install
```

`poetry install` adds the runtime dependencies and the dev tools (pytest, ruff, pre-commit, httpx for the API tests, griffe for the docs). Add a Mapillary token as described in [Getting Started](getting-started.md#from-source-poetry).

## Running tests

```bash
poetry run pytest
```

The tests don't need a Mapillary token, a GPU or network access: external services and the heavy models are replaced with fakes.

```
tests/
├── conftest.py                 # pytest markers
├── conversions/test_basic.py   # unit and coordinate conversions
├── test_api.py                 # web API: jobs, logs, WebSocket, search, options
├── test_dotenv_builder.py      # interactive .env builder
├── test_joblog.py              # JobReporter and the console reporter
├── test_naip.py                # NAIP stage: roof offset, roof patches, draping
├── test_obstruction_cache.py   # segmentation model caching
├── test_osm_client.py          # Overpass retries and mirrors
├── test_pipeline.py            # PipelineChain: stages, forks, resume, events
├── test_redact.py              # keeping the token out of errors and logs
├── test_settings.py            # Settings: token, env files, defaults
└── test_terrain.py             # terrain stage and buildings on terrain
```

## Code style and pre-commit

Pre-commit runs on every commit (and on pull requests through [pre-commit.ci](https://pre-commit.ci/)):

| Hook                                                 | Checks                                           |
| ---------------------------------------------------- | ------------------------------------------------ |
| ruff, ruff-format                                    | Python linting and formatting                    |
| mypy                                                 | Python types                                     |
| codespell                                            | spelling                                         |
| mdformat                                             | Markdown formatting (not `website/`)             |
| prettier                                             | YAML and TOML formatting                         |
| pyproject-fmt, validate-pyproject                    | `pyproject.toml`                                 |
| trailing-whitespace, end-of-file-fixer, check-yaml, ... | general file hygiene                          |

Run them all by hand with:

```bash
poetry run pre-commit run --all-files
```

## Continuous integration

Every pull request runs (`.github/workflows/ci.yml`):

- the tests on **Linux x86_64**, **Linux ARM64** and **Windows**;
- the tests on **macOS (Apple Silicon)** as a separate job: it still reports on the PR, but doesn't block it, since free macOS runners can be slow to start;
- a native build of the **CPU Docker image** on amd64 and arm64, with a smoke test inside it;
- **build-and-test**, the check required to merge: it passes when the Linux and Windows tests and both Docker builds passed.

The docs workflow (`.github/workflows/docs.yml`) checks that the API pages are up to date, builds this site, opens every page of the build in a browser (`npm run test:site`), and deploys it from `main`.

## Adding a pipeline stage

1. Write the stage as a function `(value, state: PipelineState) -> value` in its own module under `src/`, e.g. `src/my_stage/my_stage.py`. `value` is the previous stage's output; return what the next stage needs.
2. Report progress through the reporter, not `print()`:

   ```python
   from common.reporter import get_reporter

   def my_stage(value, state):
       reporter = get_reporter(state)
       for item in reporter.progress(items, desc="Doing things"):
           ...
       reporter.info("Did things")
       return value
   ```

3. Register it in `build_pipeline()` in `src/pipeline.py`, with a label and a short detail for the web app's timeline:

   ```python
   pipeline.add_stage("my_stage", my_stage, label="My stage", detail="What it does")
   ```

4. If it's optional, read a flag from the metadata and return `value` unchanged when it's off (see `terrain_stage`). Set the flag in `src/main.py` (a CLI option) and in `src/api/jobs.py` (a job option); add it to `OPTIONAL_STAGES` there so the web timeline shows the stage as skipped.

Use `state.get_metadata()` / `state.require_metadata()` to read shared context and `state.set_metadata()` to pass results to later stages.

## Documentation

This site is built with [Docusaurus](https://docusaurus.io/) and lives in `website/`. You need [Node.js](https://nodejs.org/) 20 or newer.

```bash
cd website
npm ci              # install
npm start           # live preview at http://localhost:3000/PSM/
npm run build       # production build into website/build (fails on broken links)
npx playwright install chromium   # once, for the tests below
npm run test:site   # opens every page of the build and fails on errors or missing chunks
```

The pages are Markdown files in `website/docs/`; the sidebar is `website/sidebars.js`.

:::tip["Loading chunk ... failed" in `npm start`]

The live preview rebuilds a page's JavaScript under a new name when you edit it (or restart the server, or switch branches), so a tab that was already open can ask for a file that's gone. Hard-refresh the page (Ctrl+Shift+R / Cmd+Shift+R). If it keeps happening, run `npm run clear` and start again, and make sure only one `npm start` is running. The production build doesn't have this problem; `npm run test:site` checks that.

:::

The **API reference** pages (except Web API) are generated from the docstrings in `src/` by `website/scripts/gen_api.py`. After changing docstrings, regenerate them from the repo root and commit the result:

```bash
poetry run npm run api --prefix website
```

CI fails if they're out of date. Docstrings use the Google style (`Args:`, `Returns:`, `Raises:`).

## Pull requests

1. Create a branch from `main`.
2. Make your changes; make sure the tests and pre-commit pass.
3. Open a pull request against `main` with a clear description.
