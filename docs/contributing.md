# Contributing

Contributions are welcome! This guide covers how to set up your development environment and follow project conventions.

## Development Setup

### 1. Clone and install

```bash
git clone https://github.com/Qualcomm-MDP/Photorealistic-Street-Maps.git
cd Photorealistic-Street-Maps
poetry install
```

This installs both runtime dependencies and dev tools (pytest, ruff, pre-commit, mkdocs).

### 2. Configure environment

```bash
poetry run python dotenv_builder.py
```

Follow the prompts to create your `.env` file with a Mapillary API token.

### 3. Activate pre-commit hooks

```bash
poetry run pre-commit install
```

Pre-commit runs **Ruff** (linting + formatting) automatically on every commit. The project uses [pre-commit.ci](https://pre-commit.ci/) for CI enforcement.

## Running Tests

```bash
poetry run pytest
```

Tests are in the `tests/` directory. The test suite includes unit tests for coordinate conversions, the dotenv builder, and pipeline integration tests.

### Test structure

```
tests/
├── conftest.py              # Shared fixtures
├── conversions/             # Coordinate conversion tests
├── test_dotenv_builder.py   # Environment file builder tests
└── test_pipeline.py         # Pipeline integration tests
```

## Code Style

The project uses **Ruff** for both linting and formatting. Configuration is in `pyproject.toml`.

```bash
# Check for issues
poetry run ruff check src/

# Auto-fix
poetry run ruff check src/ --fix

# Format
poetry run ruff format src/
```

## Building the Documentation

This documentation site is built with MkDocs and the Material theme.

```bash
# Live preview (auto-reloads on changes)
poetry run mkdocs serve

# Build static site
poetry run mkdocs build

# Deploy to GitHub Pages
poetry run mkdocs gh-deploy
```

The docs source lives in the `docs/` directory. API reference pages use `mkdocstrings` to auto-generate documentation from Python docstrings.

## Project Conventions

**Python version:** 3.11 -- 3.13 (see `pyproject.toml` for exact constraints).

**Package manager:** Poetry. Do not use `pip install` directly; add dependencies with `poetry add`.

**Source layout:** All application code lives under `src/`. The pipeline entry point is `src/main.py`. Each pipeline stage has its own subpackage (`data_ingest`, `mesh_builder`, `texturing`, `segmentation`). Shared utilities go in `common/`.

**Type hints:** Use type hints for function signatures. The project uses Pyright for type checking (`tool.pyright` config in `pyproject.toml`).

## Adding a New Pipeline Stage

1. Create a new module under `src/` (e.g., `src/my_stage/my_stage.py`)
2. Implement a stage function with the signature `(value, state: PipelineState) -> Any`
3. Register it in `main.py`:

```python
from my_stage.my_stage import my_stage_fn

run_pipeline.add_stage("my_stage", my_stage_fn)
```

The `value` parameter receives the output of the previous stage. Use `state.require_metadata()` and `state.set_metadata()` to access shared context.

## Pull Request Process

1. Create a feature branch from `main`
2. Make your changes and ensure tests pass (`poetry run pytest`)
3. Ensure linting passes (`poetry run ruff check src/`)
4. Open a PR against `main` with a clear description of the changes
5. Pre-commit.ci will run automatically on the PR
