# PSXHelper

**A beginner-friendly Pakistan Stock Exchange research demo, built for a friend.**

My friend has been discussing investing with me for months but hasn't managed to
start. PSXHelper brings company information and plain-language explanations into
one place, helping him research his first stocks with less friction, not choosing
stocks for him.

Built for the [Hacktoberfest Weekend Challenge: Build for a Friend](https://dev.to/challenges/hacktoberfest-weekend-2026-10-01).
This is a demo. Its scope grew during development, and work may continue after October.

## Features

- Provider-validated ticker lookup; data availability varies by company.
- Company snapshot, prices, share counts and price/market-cap history.
- Income statements, balance sheets, cash flow and provider-reported ratios.
- Yearly dividend summaries and payout observations.
- Compact K/M/B figures, chart hover values and keyboard observation selection.
- Cash preview showing affordable whole units and remaining PKR cash.
- Local Gemma explanations with a separate panel of sourced evidence.
- Visible missing-data states, source references and coverage caveats.

## Local AI is core

The explanation layer uses **Ollama and Gemma 4**, translating available research
evidence into beginner-focused explanations. Model-generated prose is kept
separate from financial figures; calculations use Python `Decimal`.

The current implementation calls `gemma4:e4b` at
`http://127.0.0.1:11434/api/chat`. E2B is the lighter deployment option, but model
selection is currently hardcoded in `src/psxhelper/main.py`, not an environment
setting. Using `gemma4:e2b` requires updating the request and reported model identifiers.

Answers use a validated JSON contract and references to supplied evidence. Stale
context, unavailable inference and unsupported output receive explicit errors.
There is no cloud fallback. Qualitative explanations can still be wrong.

## Stack

Python 3.12 · FastAPI · pandas · Pydantic · HTML/CSS/JavaScript · Ollama · Gemma 4

OpenCode was used as the AI coding agent.

## Run locally

Requires Python 3.12, `uv`, Ollama, and access to the private data SDK.
Run application commands from the repository root; a standalone wheel is not supported.

Install application dependencies:

```bash
uv sync --locked
```

Install the current model separately. This command downloads model weights:

```bash
ollama pull gemma4:e4b
```

Ensure Ollama is running on loopback port `11434`. If it is not already managed
by a local service, start it in a separate terminal:

```bash
OLLAMA_HOST=127.0.0.1:11434 ollama serve
```

Start PSXHelper:

```bash
uv run psxhelper
```

- Dashboard: <http://127.0.0.1:8000>
- Example company page: <http://127.0.0.1:8000/stocks/DCR>
- API documentation: <http://127.0.0.1:8000/docs>
- Runtime status: <http://127.0.0.1:8000/health>

The application binds to loopback only. Local inference does not make market-data
fetching offline: provider access still requires network connectivity.

## Data access and permissions

Data currently comes through a **private, unofficial SDK**. The data-fetching APIs
will not be released publicly until full permission is obtained. Public source
availability must not be interpreted as access to that SDK or permission to use,
retain, publicly present or redistribute provider data.

The adapter reuses `utils/sarmaaya_util.py` in the development checkout. No private
SDK distribution instructions or provider credentials are included here.
Datasets remain labeled `remote_unverified`; cached provider data is memory-only.

## Important limitations

- Educational research only, not investment advice, recommendations or trade execution.
- Cash previews assume a one-unit lot and zero fees/taxes, using a dated close.
- Statement accounting basis remains unknown. Only known income monetary rows use
  a labeled, user-assumed million-PKR scale; other units/scales remain unverified.
- Fundamentals are numeric-coerced; original decimal-string precision may be lost.
- Missing information is not zero. Verified EPS may be unavailable, so P/E stays missing.
- Market cap is provider-reported, not reconstructed from current share counts.
- Dividend summaries are not verified trailing yields; ex-dates/lifecycle remain unknown.
- Prices, source accuracy, adjustment basis and share effective dates require verification.

## Tests

```bash
uv run --no-sync python -m unittest discover -s tests -v
```

Tests use mocked provider inputs. Passing them does not establish live-model
quality, financial accuracy or data permissions. No fresh tests were run for this README update.

## What's next

Improve beginner explanations, model evaluation and dashboard usability; collect
my friend's feedback; resolve data permissions before broader distribution.

See [implementation progress](docs/implementation-progress.md),
[deferred data work](docs/deferred-data-work.md) and the
[demo plan](docs/challenge-demo-plan.md) for development notes. Older planning
documents may still describe the earlier `llama.cpp` runtime choice; the current
implementation uses Ollama.
