# AutoMateQA

AI-powered, self-healing test automation engine built on Playwright. Record browser interactions, add assertions visually, and replay tests that fix their own broken selectors — deterministically first, visually second, with LLM as a last resort.

## Features

- **Record & Replay** — Record browser interactions with a single CLI command, replay with confidence scoring and smart retries
- **Playwright-Native Locators** — `get_by_role`, `get_by_label`, `get_by_test_id`, `get_by_placeholder` enrichment at record time via the ARIA accessibility API
- **Self-Healing Selectors** — 4-layer pipeline: cache, deterministic fingerprint matching, visual healing (vision LLM), text-based LLM
- **Multi-Model LLM Support** — OpenAI, Anthropic (Claude), and local models (Ollama/vLLM) for healing
- **In-Browser Assertion Authoring** — Click elements to add assertions via a floating overlay (15 assertion types)
- **18 Selector Strategies** — Including 4 Playwright-native, shadow DOM pierce, and iframe locators
- **Smart Retry with Backoff** — Exponential, linear, or no retry with per-action-type timeouts
- **Cross-Browser Testing** — Chromium, Firefox, WebKit with device emulation, locale, and timezone support
- **Compressed `.aqa` Format** — ~9x smaller test files via gzip compression with transparent load/save
- **Multi-Format Reports** — HTML dashboard, JUnit XML (CI/CD), and JSON reports
- **Drift Detection** — Check selectors for degradation without executing actions
- **Test Suites** — Batch execution with optional parallelism
- **Network-Aware Waits** — Opt-in per-step API call capture for targeted waits instead of generic networkidle
- **HAR / Trace / Video** — Full Playwright diagnostics capture

## Architecture

```
CLI (cli.py) — record | execute | inspect | drift-check | suite
  |
  v
TestEngine (core.py) — orchestrator
  |
  |-- PlaywrightLocatorBuilder   ARIA-based locator enrichment at record time
  |-- SelectorEngine             18 strategies + 5-factor confidence scoring
  |-- AssertionEngine            15 assertion types + healing retry
  |-- HealingEngine              cache -> deterministic -> visual -> LLM (4-layer)
  |-- VisualHealer               screenshot + vision model healing
  |-- StepExecutor               resolve -> heal -> act -> assert (smart retry)
  |-- RecorderEngine             captures clicks, inputs, drag-drop, file uploads
  |-- NetworkCaptureLayer        per-step API call tracking (opt-in)
  |-- DriftDetector              confidence trend tracking
  |-- ReportGenerator            HTML / JUnit XML / JSON reports
  |-- BrowserManager             Playwright lifecycle + multi-browser + device emulation
  |       |
  |       v
  |   Injected JS
  |     |-- assertion_layer.js   floating button, click-to-assert, Ctrl+Shift+A
  |     +-- recorder_v2.js       drag-drop, file upload, right-click, extended keys
  |
  +-- LLM Providers (llm_providers.py)
        |-- OpenAIProvider       GPT-4o, GPT-4o-mini
        |-- AnthropicProvider    Claude Sonnet, Claude Opus
        +-- LocalProvider        Ollama, vLLM, LM Studio
```

### 4-Layer Healing Pipeline

```
Selector fails
  |
  v
Layer 1: Cache ------(hit)-----> Validate --> Return
  | (miss)
  v
Layer 2: Deterministic (fingerprint matching against live DOM)
  | score >= threshold --> Validate interactability --> Return
  | (below threshold)
  v
Layer 3: Visual (screenshot + vision LLM bounding box detection)
  | match --> Validate element at coordinates --> Return
  | (no match)
  v
Layer 4: LLM Text (structured prompt, scoped DOM, multi-model)
  | --> Validate existence --> Fingerprint similarity --> Return
  | (all failed)
  v
Coordinate click fallback (stored click_x/click_y from recording)
```

## Project Structure

```
AutoMateQA/
|-- cli.py                      CLI (record / execute / inspect / drift-check / suite)
|-- requirements.txt            playwright, pydantic, openai, anthropic, click, rich
|-- tests/
|   +-- test_healer.py          Unit tests for healing engine
+-- engine/
    |-- models.py               Pydantic models (enums, fingerprints, steps, config, results)
    |-- core.py                 TestEngine — high-level orchestrator
    |-- browser.py              BrowserManager — multi-browser + device emulation + HAR/trace/video
    |-- recorder.py             RecorderEngine — captures actions + async locator enrichment
    |-- selector.py             SelectorEngine — 18 strategies + confidence scoring
    |-- locator_builder.py      PlaywrightLocatorBuilder — ARIA-based locator enrichment
    |-- assertions.py           AssertionEngine — 15 assertion types + healing retry
    |-- healer.py               HealingEngine — cache + deterministic + LLM healing
    |-- visual_healer.py        VisualHealer — screenshot-based healing via vision models
    |-- executor.py             StepExecutor — smart retry + advanced waits
    |-- llm_providers.py        Multi-model LLM abstraction (OpenAI, Anthropic, local)
    |-- network_layer.py        NetworkCaptureLayer — per-step API call tracking
    |-- drift_detector.py       DriftDetector + FlakinessTracker
    |-- reporter.py             ReportGenerator — HTML / JUnit XML / JSON
    +-- js/
        |-- assertion_layer.js  Injected browser overlay for assertion authoring
        +-- recorder_v2.js      Extended event capture (drag, upload, right-click)
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
# Optional: all browsers for cross-browser testing
playwright install

# Configure environment
cp .env.example .env
# Edit .env with your API keys and preferences
```

The engine reads configuration from `.env` file and environment variables. CLI flags override env values when provided. See `.env.example` for all available settings.

## Usage

### Record a test

Opens a headed browser. Interact with the page, add assertions via the overlay, close the browser to save.

```bash
# Default: saves as test.aqa (compressed)
python3 cli.py record --url "https://example.com"

# Custom name
python3 cli.py record --url "https://example.com" -o login_test

# Save as plain JSON instead
python3 cli.py record --url "https://example.com" -o login_test.json

# With network capture + HAR + trace
python3 cli.py record --url "https://example.com" -o login_test --capture-network --har --trace
```

**In-browser controls during recording:**

| Control | Action |
|---|---|
| Click the floating button | Toggle assertion mode |
| Ctrl+Shift+A | Toggle assertion mode |
| Right-click element (in assertion mode) | Add assertion to element |
| ESC | Exit assertion mode |
| Close browser | Stop recording |

**Captured events:** click, double-click, type, select, check/uncheck, hover, keypress (Enter, Tab, Escape, arrow keys, Backspace, Delete, Space), scroll, drag-and-drop, file upload, right-click.

### Execute a test

```bash
# Basic execution (supports both .aqa and .json)
python3 cli.py execute test.aqa

# With LLM healing
python3 cli.py execute test.aqa --llm --healing-mode auto_update

# With Anthropic instead of OpenAI
python3 cli.py execute test.aqa --llm --provider anthropic --model claude-sonnet-4-20250514

# Cross-browser
python3 cli.py execute test.aqa --browser firefox --headless

# Device emulation
python3 cli.py execute test.aqa --device "iPhone 14" --headless

# With report generation
python3 cli.py execute test.aqa --report html --report-path report.html

# Full options
python3 cli.py execute test.aqa \
  --llm --provider openai --healing-mode auto_update \
  --headless --browser chromium \
  --retry-strategy exponential --max-retries 3 \
  --report junit --report-path results.xml \
  --trace --verbose
```

### Check for selector drift

Navigates to the page and checks all selectors without executing actions.

```bash
python3 cli.py drift-check test.aqa --threshold 0.7
python3 cli.py drift-check tests/*.aqa --threshold 0.6
```

### Run a test suite

```bash
# Sequential
python3 cli.py suite test1.aqa test2.aqa test3.aqa

# Parallel (4 workers)
python3 cli.py suite test1.aqa test2.aqa test3.aqa --parallel --workers 4

# Stop on first failure
python3 cli.py suite tests/*.aqa --stop-on-failure
```

### Inspect a test

```bash
python3 cli.py inspect test.aqa
```

## File Formats

| Extension | Format | Use Case |
|---|---|---|
| `.aqa` | Gzip-compressed JSON | **Default.** ~9x smaller than JSON. |
| `.json` | Plain JSON | Human-readable, git-diffable. |

Both formats are fully interchangeable — the engine detects the extension and handles compression transparently.

## Configuration

### CLI Options

| Option | Default | Description |
|---|---|---|
| `-o, --output` | `test` | Test name (auto-appends `.aqa`) |
| `--llm / --no-llm` | `--no-llm` | Enable LLM-backed healing |
| `--provider` | `openai` | LLM provider: `openai`, `anthropic`, `local` |
| `--model` | `gpt-4o` | LLM model name |
| `--healing-mode` | `disabled` | `disabled` / `strict` / `auto_update` / `debug` |
| `--confidence` | `0.75` | Selector confidence threshold (0.0-1.0) |
| `--browser` | `chromium` | Browser: `chromium`, `firefox`, `webkit` |
| `--device` | — | Device emulation (e.g. `"iPhone 14"`) |
| `--locale` | — | Browser locale (e.g. `"en-US"`) |
| `--headless` | off | Run without GUI |
| `--retry-strategy` | `exponential` | `none`, `linear`, `exponential` |
| `--max-retries` | `3` | Max retries per step |
| `--capture-network` | off | Capture per-step API calls (opt-in) |
| `--har` | off | Record HAR file |
| `--trace` | off | Record Playwright trace |
| `--video` | off | Record session video |
| `--report` | — | Report format: `html`, `junit`, `json` |
| `--report-path` | — | Report output path |
| `--storage-state` | — | Auth state JSON path (cookies + localStorage) |
| `--save-storage-state` | off | Save auth state after execution |
| `-v, --verbose` | off | Debug-level logging |

### Healing Modes

| Mode | Behaviour |
|---|---|
| `disabled` | No healing. Steps fail if selector can't resolve. |
| `strict` | Heal at runtime but **do not** update the stored test. |
| `auto_update` | Heal **and** persist the new selector into the test file. Originals preserved in `selector_history`. |
| `debug` | Print suggestions only; never apply them. |

## Selector Strategies (18 total)

Strategies are tried in priority order. The highest-confidence candidate wins.

| # | Strategy | Confidence | Source |
|---|---|---|---|
| 1 | Precomputed (pw_native) | 0.84-0.97 | Playwright locators from recording |
| 2 | Precomputed (preferred) | 0.95 | CSS/role from recording |
| 3 | `pw-test-id` | 0.97 | `page.get_by_test_id()` |
| 4 | `pw-label` | 0.92 | `page.get_by_label()` |
| 5 | `pw-role` | 0.89 | `page.get_by_role()` + ARIA name |
| 6 | `pw-placeholder` | 0.84 | `page.get_by_placeholder()` |
| 7 | `data-testid` (CSS) | ~0.87 | `[data-testid="..."]` |
| 8 | `data-cy` / `data-test` | ~0.87 | Common test attributes |
| 9 | `shadow-pierce` | 0.82 | Shadow DOM `>>` combinator |
| 10 | `id` | ~0.82 | `#id` (dynamic IDs skipped) |
| 11 | `frame-locator` | 0.80 | iframe `frame_locator()` |
| 12 | `name` | ~0.77 | `[name="..."]` form fields |
| 13 | `placeholder` (CSS) | ~0.77 | CSS attribute selector |
| 14 | `role+name` (CSS) | ~0.73 | CSS role selector |
| 15 | `aria-label` | ~0.73 | `[aria-label="..."]` |
| 16 | `text-exact` | ~0.57 | `get_by_text(exact=True)` |
| 17 | `tag+text` | ~0.54 | `tag:has-text("...")` |
| 18 | `css` / `xpath` | 0.30-0.65 | Deepest fallback |

### Confidence Formula

```
Confidence = 0.4 * T + 0.2 * R + 0.15 * A + 0.15 * P + 0.1 * D
```

| Factor | Weight | Measures |
|---|---|---|
| **T** - Test ID | 0.40 | data-testid, data-cy, id, name attributes |
| **R** - Role | 0.20 | ARIA role + label |
| **A** - Attributes | 0.15 | placeholder, href, classes, other attrs |
| **P** - Position | 0.15 | Parent tag, sibling index |
| **D** - DOM structure | 0.10 | XPath / CSS selector depth |

## Assertion Types (15 total)

| Type | Description |
|---|---|
| `visible` | Element is visible on page |
| `hidden` | Element is not visible |
| `text_equals` | Element text matches exactly |
| `text_contains` | Element text contains substring |
| `matches_pattern` | Element text matches regex |
| `attribute_equals` | Element attribute matches value |
| `exists` | Element exists in DOM |
| `css_property` | Computed CSS property value (e.g. `background-color`) |
| `element_count` | Number of matching elements |
| `url_equals` | Current page URL matches exactly |
| `url_contains` | Current page URL contains substring |
| `console_no_errors` | No `console.error()` calls during step |
| `network_status` | Specific API returned expected HTTP status |
| `js_expression` | Arbitrary JavaScript evaluates to truthy |
| `accessibility` | Basic a11y check (role, aria-label, focusability) |

## CI/CD Integration

### GitHub Actions

```yaml
name: AutoMateQA Tests
on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: '3.12'
      - run: pip install -r requirements.txt && playwright install chromium
      - run: |
          python3 cli.py execute tests/login.aqa \
            --headless --retry-strategy exponential --max-retries 3 \
            --report junit --report-path results.xml
      - uses: dorny/test-reporter@v1
        if: always()
        with:
          name: AutoMateQA
          path: results.xml
          reporter: java-junit

  drift-check:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: '3.12'
      - run: pip install -r requirements.txt && playwright install chromium
      - run: python3 cli.py drift-check tests/*.aqa --threshold 0.6
```

## Design Principles

1. **Deterministic first** — stable selectors (data-testid, Playwright locators, id, role) before anything else
2. **Heuristic second** — 18-strategy confidence scoring with text validation
3. **Visual third** — screenshot + vision model to locate elements by appearance
4. **AI last** — LLM text healing only when all other layers fail
5. **Validate everything** — healed selectors must pass interactability and fingerprint similarity checks
6. **No guessing** — accessible names come from explicit ARIA attributes and standard `<label>` associations only
7. **Lean by default** — compressed `.aqa` format, network capture opt-in, no files created on execution
