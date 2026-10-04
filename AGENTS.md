# Repository context

- Application is not implemented yet. `psxhelper` resolves to `src/psxhelper/__init__.py:main` and only prints a greeting; no FastAPI server exists.
- Working code is `utils/sarmaaya_util.py`; tests import it directly from the repository root. It sits outside the `src/psxhelper` package built by `uv_build`.
- `docs/challenge-demo-plan.md` defines the reduced demo scope; `docs/poc-design.md` is the longer-term design. Proposed files, endpoints, dependencies, and commands in these documents are not implemented features.
- Demo plan selects `llama.cpp`/`llama-server`, unlike the full design's Ollama proposal. Read the relevant plan before implementing either runtime.

## Setup and verification

- Use Python 3.12 (`.python-version`; existing `.venv`), not an unqualified system interpreter.
- Baseline setup: `uv sync --locked`. Manifest and lockfile include only `urllib3`; `pandas` and `requests`, required by the utility and tests, are undeclared.
- From repository root, run current tests with temporary missing dependencies: `uv run --with pandas --with requests python -m unittest discover -s tests -v`.
- Focus one test: `uv run --with pandas --with requests python -m unittest discover -s tests -p 'test_sarmaaya_util.py' -k test_request_arguments -v`.
- Specify `-s tests`: tests have no `__init__.py`. Current suite uses `unittest`, not the planned `pytest`, and mocks HTTP; no live provider or model is required.
- No repository CI, lint, formatter, or typecheck configuration exists; do not present guessed commands as established checks.

## Sarmaaya contracts

- `_get`/`_post` raise HTTP errors, reject `success: false` before unwrapping `response`, and preserve payloads without that key. Falsey response values are valid.
- Retry policy covers GET only. Company details use POST; fundamentals resolve ISIN first and accept only `ANN`/`LTM`.
- Price history normalizes dates to UTC, sorts ascending, and resets its index. Tests pin these behaviors.

## Data and implementation boundaries

- Provider access is not permission to collect or redistribute data. Plans require authorized sources or permitted imports; fictional fixtures use fictional symbols, never real tickers with invented financials.
- Planned financial calculations use Python `Decimal` and decimal-string JSON. Missing data remains missing, not zero; preserve period, statement basis, units, source dates, and coverage.
- Saving design documents does not authorize downloads, account actions, deployment, or publication. Consult demo plan's scope cuts rather than scaffolding the full POC.
