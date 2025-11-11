# Repository Guidelines

## Project Structure & Module Organization
SuperKeet centers on the `src` package: UI components in `src/ui`, audio capture and playback in `src/audio`, batch flows in `src/batch`, speech recognition adapters under `src/asr`, and cross-cutting helpers in `src/utils` and `src/services`. Configuration is loaded from the repository root (`config.yml`) and sensitive keys belong in `credentials.yml`, which should be copied from the provided `.dist` template. Tests mirror the runtime modules inside `tests/`, and assets such as icons live in `assets/`. Temporary recordings created during debugging are written to `debug_audio/`.

## Build, Test, and Development Commands
Install dependencies with Poetry and run all quality gates through the bundled script:
```bash
poetry install
./checkpython.sh
```
`./checkpython.sh` performs linting, typing, security scanning, and pytest in one pass. During iterative work it is faster to run individual tools: `poetry run python -m superkeet.main` launches the app, `poetry run pytest` executes the test suite with coverage, and `ruff check .`/`ruff format` keep formatting consistent.

## Coding Style & Naming Conventions
Python sources follow the Ruff configuration in `pyproject.toml`: 88 character lines, import sorting, and warnings for unused symbols or naming issues. Prefer explicit typing; `mypy` is configured to reject untyped definitions. Use `snake_case` for functions and variables, `PascalCase` for classes, and align UI resource names with their icon filenames in `assets/`. Keep modules cohesive—batch flows in `src/batch`, hotkey handling in `src/hotkey`, etc.—to preserve the existing structure.

## Testing Guidelines
Pytest is configured through `pyproject.toml` with strict markers and coverage targets that generate HTML and XML reports. Place new tests in `tests/`, naming files `test_<feature>.py`, classes `Test<Feature>`, and functions `test_<behavior>`. Use markers (`@pytest.mark.unit`, `integration`, `slow`) so contributors can filter suites (`poetry run pytest -m "not slow"`). When tests touch audio artifacts, leverage fixtures in `tests/test_audio_*` to avoid flaky I/O.

## Commit & Pull Request Guidelines
Recent history favors concise, Title Case commit messages (for example, `Next gen fixes`). Summaries should describe the outcome and start with an action phrase; elaborate details belong in the body. For pull requests, describe user-visible impact, list validation steps (e.g., `./checkpython.sh`), and link any tracking issues. Include screenshots or recordings when UI behavior changes or new hotkey flows are introduced.

## Configuration & Security Notes
Never commit `credentials.yml`; rely on the `.dist` template and document changes in `SETUP.md`. Audio debugging output may contain sensitive content—purge `debug_audio/` before publishing artifacts. If you modify configuration defaults, reflect the change in `README.md` and verify accessibility permissions remain documented.
