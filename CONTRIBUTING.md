# Contributing to qmlkit

Thanks for helping make quantum-inspired ML more usable.

## Development setup

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -e ".[dev]"
```

## Checks

Run these before opening a pull request:

```bash
python -m pytest
python -m ruff check .
python -m mypy qmlkit
```

## Contribution guidelines

- Keep the core package lightweight. NumPy is the only required runtime dependency.
- Treat sklearn and PyTorch as optional integrations.
- Prefer practical ML usability over quantum-theory completeness.
- Do not add a circuit DSL or real hardware execution in small feature PRs.
- Add tests for public APIs and behavior changes.

## Release checklist

- Update `CHANGELOG.md`.
- Confirm `python -m pytest` passes.
- Confirm examples run with the relevant optional dependencies installed.
- Tag the release with `vX.Y.Z`.
