# Repository Guidelines

## Project Structure & Module Organization
- Source code: `src/transcriptor/` (core library and CLI entrypoints).
- Tests: `tests/` mirroring package paths (e.g., `tests/cli/test_run.py`).
- Scripts: `scripts/` for developer utilities (lint, format, local run).
- Assets: `assets/` for small fixtures; avoid committing large binaries.
- Docs: `docs/` for design notes and user guides.

Example layout:
```
src/transcriptor/{__init__.py, cli.py, ...}
tests/{unit, integration}/...
scripts/{lint, format, test, run}
```

## Build, Test, and Development Commands
- Setup: `make venv` (create venv). Optional: `make install-whisper` for local STT.
- Run script: `make run ARGS="..."` (pass flags to scripts/meeting_minutes.py).
- OpenRouter flow (recommended): `make minutes FILE=Session.m4a OUT=minutes.txt`.
- Test: `make test` or `pytest -q` (if tests exist).
- Lint/Format: `ruff check .` and `black .` (if configured).

## Coding Style & Naming Conventions
- Language: Python 3.11+.
- Formatting: Black (line length 88); imports via isort (profile=black); linting with Ruff.
- Naming: modules/packages `snake_case`; classes `PascalCase`; functions/vars `snake_case`; constants `UPPER_SNAKE_CASE`.
- File rules: one public class per file when practical; keep functions <50 lines or extract helpers.

## Testing Guidelines
- Framework: pytest with plain asserts and fixtures.
- Structure: mirror `src/` paths; unit first, selective integration.
- Coverage: target ≥90% on changed lines; add tests with regressions.
- Run: `pytest -q` or `make test`; focused: `pytest tests/unit/test_cli.py::test_main`.

## Commit & Pull Request Guidelines
- Commits: Conventional Commits (e.g., `feat(cli): add transcript command`).
- PRs: concise title, clear description, linked issue, test evidence (output or screenshots), and notes on performance/compat.
- Keep PRs small and scoped; include `docs:` updates when behavior changes.

## Security & Configuration Tips
- Do not commit secrets or large datasets. Use `.env.local` (gitignored) and document keys in `.env.example`.
- Validate and sanitize external inputs (paths, URLs). Prefer streaming for large files.
- Pin dependencies for reproducibility; upgrade in dedicated PRs.

## OpenRouter Usage
- Set `OPENROUTER_API_KEY` in your environment or `.env`.
- Use audio‑capable chat models with `--stt-chat-model` (e.g., `google/gemini-2.5-flash-lite`, `openai/gpt-4o-audio-preview`).
- The tool splits long audio and sends base64 WAV segments via chat/completions using `input_audio` content.
