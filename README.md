# AutoMateQA

AI-powered, self-healing test automation engine built on Playwright. Record browser interactions, add assertions visually, and replay tests that fix their own broken selectors — deterministically first, with LLM as a last resort.

## Features

- **Record & Replay** — Record browser interactions with a single CLI command, replay them with confidence scoring
- **Self-Healing Selectors** — Deterministic fingerprint matching resolves ~70% of broken selectors without LLM; OpenAI fills the gap
- **In-Browser Assertion Authoring** — Click elements to add assertions via a floating overlay (7 assertion types)
- **5-Factor Confidence Scoring** — Every element match is scored across test IDs, ARIA roles, attributes, position, and DOM structure
- **4 Healing Modes** — `disabled`, `strict`, `auto_update`, `debug` — choose how aggressively selectors are repaired
- **Healing Telemetry** — Structured JSON logs for every heal: method, similarity score, duration, token usage
- **Healing Cache** — Healed selectors are cached in-memory and persisted to the test file in `auto_update` mode

## Architecture

```
CLI (cli.py)
  │
  ▼
TestEngine (core.py) ── orchestrator
  │
  ├── SelectorEngine    12 strategies + 5-factor confidence scoring
  ├── AssertionEngine    7 assertion types + healing retry
  ├── HealingEngine      cache → deterministic → LLM (3-layer pipeline)
  ├── StepExecutor       resolve → heal → act → assert → screenshot
  ├── RecorderEngine     captures clicks, inputs, navigation
  └── BrowserManager     Playwright lifecycle + JS injection
          │
          ▼
  Injected JS Assertion Layer (floating button, click-to-assert, Ctrl+Shift+A)
```

### Healing Pipeline

```
Selector fails
  │
  ▼
Check cache ──(hit)──► Validate ──► Return
  │ (miss)
  ▼
Deterministic heal (fingerprint matching against DOM)
  │ score ≥ threshold ──► Validate interactability ──► Return
  │ (below threshold)
  ▼
LLM heal (structured prompt, scoped DOM)
  │
  ▼
Validate existence → Fingerprint similarity → Interactability
  │ (pass) ──► Cache + Return
  │ (fail) ──► Retry with fresh DOM
```

## Project Structure

```
AutoMateQA/
├── cli.py                    # CLI entrypoint (record / execute / inspect)
├── requirements.txt          # playwright, pydantic, openai, click, rich
├── tests/
│   └── test_healer.py        # Unit tests for healing engine
└── engine/
    ├── models.py             # 15+ Pydantic models (enums, fingerprints, steps, config, results)
    ├── browser.py            # BrowserManager – Playwright lifecycle + JS bindings
    ├── recorder.py           # RecorderEngine – captures user actions during recording
    ├── selector.py           # SelectorEngine – 12 strategies + confidence scoring
    ├── assertions.py         # AssertionEngine – 7 types + healing retry fallback
    ├── healer.py             # HealingEngine – cache + deterministic + LLM (3-layer)
    ├── executor.py           # StepExecutor – per-step execution pipeline
    ├── core.py               # TestEngine – high-level orchestrator
    └── js/
        └── assertion_layer.js  # Injected browser overlay for assertion authoring
```

## Setup

```bash
# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Install Playwright browsers
playwright install chromium
```

Set `OPENAI_API_KEY` in your environment to use LLM healing.

## Usage

### Record a test

Opens a headed browser. Interact with the page, add assertions via the in-browser overlay, then close the browser to save.

```bash
python3 cli.py record --url "https://example.com" --output my_test.json
```

**In-browser controls during recording:**

| Control | Action |
|---|---|
| Click the floating button | Toggle assertion mode |
| Ctrl+Shift+A | Toggle assertion mode |
| Click element (in assertion mode) | Add assertion to element |
| ESC | Exit assertion mode |
| Close browser | Stop recording |

### Execute a test

```bash
# Without healing
python3 cli.py execute my_test.json --no-llm

# With LLM healing (strict mode — heal but don't persist)
python3 cli.py execute my_test.json --llm --healing-mode strict

# Auto-update mode — heal and save fixed selectors back to the test file
python3 cli.py execute my_test.json --llm --healing-mode auto_update

# Headless with custom confidence threshold
python3 cli.py execute my_test.json --llm --healing-mode auto_update --headless --confidence 0.8
```

### Inspect a test

Pretty-prints step summary, assertions, and config.

```bash
python3 cli.py inspect my_test.json
```

## Configuration

| Option | Default | Description |
|---|---|---|
| `--llm / --no-llm` | `--no-llm` | Enable OpenAI-backed healing |
| `--healing-mode` | `disabled` | `disabled` / `strict` / `auto_update` / `debug` |
| `--confidence` | `0.75` | Selector confidence threshold (0.0–1.0) |
| `--model` | `gpt-4o` | OpenAI model for healing |
| `--headless` | off | Run browser without GUI |
| `--screenshot-dir` | `screenshots` | Directory for failure screenshots |
| `-v, --verbose` | off | Debug-level logging |

## Healing Modes

| Mode | Behaviour |
|---|---|
| `disabled` | No healing. Steps fail if selector can't resolve. |
| `strict` | Heal at runtime but **do not** update the stored test model. |
| `auto_update` | Heal **and** persist the new selector into the test file. Originals are preserved in `selector_history`. |
| `debug` | Print healing suggestions only; never apply them. |

## Confidence Scoring

Elements are scored using a 5-factor weighted algorithm:

```
Confidence = 0.4×T + 0.2×R + 0.15×A + 0.15×P + 0.1×D
```

| Factor | Weight | Measures |
|---|---|---|
| **T** – Test ID | 0.40 | data-testid, data-cy, id, name attributes |
| **R** – Role | 0.20 | ARIA role + label |
| **A** – Attributes | 0.15 | placeholder, href, classes, other attrs |
| **P** – Position | 0.15 | Parent tag, sibling index |
| **D** – DOM structure | 0.10 | XPath / CSS selector depth |

12 selector strategies are tried in priority order: `data-testid` > `data-cy` > `id` > `name` > `role+name` > `aria-label` > `placeholder` > `text` > `tag+text` > `css` > `xpath`. The highest-confidence candidate wins.

## Assertion Types

| Type | Description |
|---|---|
| `visible` | Element is visible on page |
| `hidden` | Element is not visible |
| `text_equals` | Element text matches exactly |
| `text_contains` | Element text contains substring |
| `matches_pattern` | Element text matches regex |
| `attribute_equals` | Element attribute matches value |
| `exists` | Element exists in DOM |

Assertions support the same healing pipeline: if an assertion target has low confidence, the engine retries with a healed selector before declaring failure. Expected values are **never** auto-healed.

## Design Principles

1. **Deterministic first** — stable selectors (data-testid, id, role) before anything else
2. **Heuristic second** — confidence scoring across 12 strategies with text validation
3. **AI last** — LLM healing only when deterministic + heuristic both fail
4. **Validate everything** — healed selectors must pass interactability (visible, enabled, rendered) and fingerprint similarity checks
5. **Observable** — structured telemetry for every healing attempt
