"""
TestEngine (core) – High-level orchestrator for the automation engine.

Provides two main modes:
  1. RECORD – Launch browser, record user actions + assertions, save test model.
  2. EXECUTE – Load a saved test model and replay it.
"""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from engine.assertions import AssertionEngine
from engine.browser import BrowserManager
from engine.executor import StepExecutor
from engine.healer import HealingEngine
from engine.models import (
    EngineConfig,
    HealingMode,
    StepStatus,
    TestModel,
    TestResult,
)
from engine.recorder import RecorderEngine
from engine.selector import SelectorEngine

logger = logging.getLogger(__name__)


class TestEngine:
    """
    Top-level engine that wires all components together.

    Usage:
        engine = TestEngine(
            llm_enabled=True,
            healing_mode="strict",
            confidence_threshold=0.75,
        )
        # Record
        await engine.record(url="https://example.com", save_path="test.json")
        # Execute
        result = await engine.execute(test_path="test.json")
    """

    def __init__(
        self,
        llm_enabled: bool = True,
        healing_mode: str = "strict",
        confidence_threshold: float = 0.75,
        max_healing_attempts: int = 2,
        headless: bool = False,
        verbose: bool = False,
        llm_model: str = "gpt-4o",
    ) -> None:
        self._config = EngineConfig(
            llm_enabled=llm_enabled,
            healing_mode=HealingMode(healing_mode),
            confidence_threshold=confidence_threshold,
            max_healing_attempts=max_healing_attempts,
            headless=headless,
            verbose=verbose,
            llm_model=llm_model,
        )

        # Sub-components
        self._selector_engine = SelectorEngine(self._config)
        self._healing_engine = HealingEngine(self._config)
        self._assertion_engine = AssertionEngine(
            self._config, self._selector_engine, self._healing_engine
        )
        self._step_executor = StepExecutor(
            self._config,
            self._selector_engine,
            self._assertion_engine,
            self._healing_engine,
        )

    @property
    def config(self) -> EngineConfig:
        return self._config

    # ------------------------------------------------------------------
    # RECORD mode
    # ------------------------------------------------------------------

    async def record(
        self,
        url: str = "",
        save_path: str = "test.json",
        test_name: str = "Recorded Test",
    ) -> TestModel:
        """
        Launch the browser in headed mode, let the user interact,
        then save the test model to disk.

        The user closes the browser (or presses Ctrl+C) to stop recording.
        """
        model = TestModel(name=test_name, base_url=url, config=self._config)
        browser = BrowserManager(self._config)
        recorder = RecorderEngine(model)

        # Wire assertion callback
        browser.on_assertion(recorder.handle_assertion)

        try:
            page = await browser.launch(url=url)

            # Inject the recording script for user-action capture
            await self._inject_recorder_script(page)

            await recorder.start(page)

            logger.info(
                "\U0001f3ac Recording started. Interact with the browser.\n"
                "   \u2022 Click the \U0001f3af button  \u2192 toggle assertion mode\n"
                "   \u2022 Ctrl+Shift+A        \u2192 toggle assertion mode\n"
                "   \u2022 Right-click in mode  \u2192 add assertion\n"
                "   Close the browser to stop recording."
            )

            # Wait until the browser is closed by the user
            await self._wait_for_browser_close(page)

        except KeyboardInterrupt:
            logger.info("Recording interrupted by user")
        finally:
            final_model = recorder.stop()
            final_model.updated_at = datetime.now(timezone.utc).isoformat()
            await browser.close()

        # Save to disk
        self._save_model(final_model, save_path)
        logger.info("✅ Test saved to %s (%d steps)", save_path, len(final_model.steps))
        return final_model

    # ------------------------------------------------------------------
    # EXECUTE mode
    # ------------------------------------------------------------------

    async def execute(
        self,
        test_path: str = "",
        test_model: Optional[TestModel] = None,
        screenshot_dir: str = "screenshots",
    ) -> TestResult:
        """
        Load a test model and replay it, returning structured results.

        Either provide a file path or a TestModel object directly.
        """
        if test_model is None:
            test_model = self._load_model(test_path)

        # Override config from the engine (allows runtime overrides)
        test_model.config = self._config

        ss_dir = Path(screenshot_dir)
        ss_dir.mkdir(parents=True, exist_ok=True)

        result = TestResult(
            test_id=test_model.test_id,
            test_name=test_model.name,
            started_at=datetime.now(timezone.utc).isoformat(),
            config_used=self._config,
        )

        browser = BrowserManager(self._config)

        try:
            page = await browser.launch(url=test_model.base_url)

            for step in test_model.steps:
                logger.info(
                    "▶ Step %d: %s", step.step_id, step.action.action_type.value
                )
                step_result = await self._step_executor.execute(
                    page, step, screenshot_dir=ss_dir
                )
                result.steps.append(step_result)

                # Log step outcome
                icon = (
                    "✅"
                    if step_result.status == StepStatus.PASSED
                    else "🔧" if step_result.status == StepStatus.HEALED else "❌"
                )
                logger.info(
                    "  %s Step %d: %s (confidence=%.2f, healed=%s)",
                    icon,
                    step_result.step_id,
                    step_result.status.value,
                    step_result.element_confidence,
                    step_result.healed,
                )

                # Abort on failure if not in auto-heal mode
                if step_result.status == StepStatus.FAILED:
                    break

        except Exception as e:
            logger.error("Execution error: %s", e)
        finally:
            await browser.close()

        result.finished_at = datetime.now(timezone.utc).isoformat()

        # Compute overall status
        if all(
            s.status in (StepStatus.PASSED, StepStatus.HEALED) for s in result.steps
        ):
            result.status = StepStatus.PASSED
        else:
            result.status = StepStatus.FAILED

        # Compute total duration
        result.total_duration_ms = sum(s.duration_ms for s in result.steps)

        # Persist test model when AUTO_UPDATE healed any step so next run uses healed selectors
        if (
            test_path
            and self._config.healing_mode == HealingMode.AUTO_UPDATE
            and any(s.healed for s in result.steps)
        ):
            self._save_model(test_model, test_path)
            logger.info(
                "Saved test model to %s (healed selectors persisted)",
                test_path,
            )

        return result

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _save_model(model: TestModel, path: str) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(model.model_dump_json(indent=2), encoding="utf-8")

    @staticmethod
    def _load_model(path: str) -> TestModel:
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Test file not found: {path}")
        raw = p.read_text(encoding="utf-8")
        return TestModel.model_validate_json(raw)

    async def _inject_recorder_script(self, page) -> None:
        """
        Inject a lightweight JS snippet that captures user actions
        (click, input, select) and sends them to the backend via console.
        """
        recorder_js = """
        (function() {
            if (window.__recorderInjected) return;
            window.__recorderInjected = true;

            // ── Patterns ──────────────────────────────────────────────
            var _frameworkAttrRe = /^(data-v-|data-reactid|_ngcontent|_nghost)/;
            var _dynIdRe = /[0-9a-f]{8}-|[0-9a-f]{12}|^f_|^\\d{6,}/;

            // ── Promote to interactive parent (Step 2) ───────────────
            function getInteractiveParent(el) {
                var cur = el;
                var depth = 0;
                while (cur && cur !== document.body && depth < 8) {
                    var tag = cur.tagName;
                    if (tag === 'PATH' || tag === 'SVG') {
                        cur = cur.parentElement;
                        depth++;
                        continue;
                    }
                    if (tag === 'BUTTON' || tag === 'A' || tag === 'INPUT' ||
                        tag === 'SELECT' || tag === 'TEXTAREA' ||
                        cur.getAttribute('role') === 'button' ||
                        cur.getAttribute('role') === 'link' ||
                        cur.getAttribute('role') === 'tab' ||
                        cur.getAttribute('role') === 'menuitem' ||
                        cur.getAttribute('role') === 'checkbox' ||
                        cur.hasAttribute('tabindex')) {
                        return cur;
                    }
                    cur = cur.parentElement;
                    depth++;
                }
                if (el && (el.tagName === 'PATH' || el.tagName === 'SVG'))
                    return el.parentElement || el;
                return el;
            }

            // ── Helpers ──────────────────────────────────────────────
            function getNthOfType(el) {
                if (!el || !el.parentElement) return 0;
                var tag = el.tagName;
                var idx = 0;
                var child = el.parentElement.firstElementChild;
                while (child) {
                    if (child.tagName === tag) {
                        if (child === el) return idx;
                        idx++;
                    }
                    child = child.nextElementSibling;
                }
                return 0;
            }

            function ownText(el) {
                var t = '';
                for (var i = 0; i < el.childNodes.length; i++) {
                    if (el.childNodes[i].nodeType === 3) t += el.childNodes[i].textContent;
                }
                return t.trim();
            }

            // Relative XPath — last 3 nodes only (Step 4)
            function relativeXPath(el) {
                if (!el || el.nodeType !== 1) return '';
                var parts = [];
                var cur = el;
                var depth = 0;
                while (cur && cur.nodeType === 1 && depth < 3) {
                    var tag = cur.tagName.toLowerCase();
                    var idx = 1;
                    var sib = cur.previousElementSibling;
                    while (sib) {
                        if (sib.tagName.toLowerCase() === tag) idx++;
                        sib = sib.previousElementSibling;
                    }
                    parts.unshift(tag + '[' + idx + ']');
                    cur = cur.parentElement;
                    depth++;
                }
                return '//' + parts.join('/');
            }

            function buildCss(el) {
                var tag = el.tagName.toLowerCase();
                if (el.id && !_dynIdRe.test(el.id)) return tag + '#' + el.id;
                var cls = Array.from(el.classList || []);
                if (cls.length) {
                    var css = tag;
                    cls.forEach(function(c) { css += '.' + c; });
                    return css;
                }
                var p = el.parentElement;
                if (p) {
                    var pTag = p.tagName.toLowerCase();
                    var pCss = pTag;
                    if (p.id && !_dynIdRe.test(p.id)) pCss = pTag + '#' + p.id;
                    else {
                        var pCls = Array.from(p.classList || []);
                        if (pCls.length) pCls.forEach(function(c) { pCss += '.' + c; });
                    }
                    if (pCss !== pTag) {
                        return pCss + ' > ' + tag + ':nth-child(' + (Array.from(p.children).indexOf(el) + 1) + ')';
                    }
                }
                return tag;
            }

            // ── Compute ranked selectors at record time (Step 1/9) ──
            function computeSelectors(el) {
                var s = {};
                var tag = el.tagName.toLowerCase();
                var text = (el.textContent || '').trim().slice(0, 60);

                // 1 — data-testid
                var tid = el.getAttribute('data-testid');
                if (tid) s.preferred = '[data-testid="' + tid + '"]';

                // 2 — data-cy / data-test / data-qa
                var cyAttrs = ['data-cy', 'data-test', 'data-qa'];
                for (var i = 0; i < cyAttrs.length; i++) {
                    var cv = el.getAttribute(cyAttrs[i]);
                    if (cv) {
                        var key = s.preferred ? 'data_cy' : 'preferred';
                        s[key] = '[' + cyAttrs[i] + '="' + cv + '"]';
                        break;
                    }
                }

                // 3 — role + accessible name
                var role = el.getAttribute('role') || '';
                if (!role && tag === 'button') role = 'button';
                if (!role && tag === 'a') role = 'link';
                if (!role && tag === 'input') {
                    var tp = el.getAttribute('type') || 'text';
                    if (tp === 'checkbox') role = 'checkbox';
                    else if (tp === 'radio') role = 'radio';
                    else role = 'textbox';
                }
                if (role) {
                    var aname = el.getAttribute('aria-label') || '';
                    if (!aname && text && text.length < 50) aname = text;
                    if (aname) s.role = 'role=' + role + '[name="' + aname.replace(/"/g, '\\"') + '"]';
                }

                // 4 — name attribute
                var nameAttr = el.getAttribute('name');
                if (nameAttr) s.name = tag + '[name="' + nameAttr + '"]';

                // 5 — placeholder
                var ph = el.getAttribute('placeholder');
                if (ph) s.placeholder = '[placeholder="' + ph + '"]';

                // 6 — label association
                if (el.id && !_dynIdRe.test(el.id)) {
                    var label = document.querySelector('label[for="' + el.id + '"]');
                    if (label) {
                        var lt = (label.textContent || '').trim().slice(0, 50);
                        if (lt) s.label = 'label:has-text("' + lt + '") >> ' + tag;
                    }
                }

                // 7 — text (only for short, unique-ish text)
                if (text && text.length <= 40) s.text = tag + ':has-text("' + text.replace(/"/g, '\\"') + '")';

                // 8 — fallback (type attribute or css)
                var typeAttr = el.getAttribute('type');
                if (typeAttr && (tag === 'input' || tag === 'button')) {
                    s.fallback = tag + '[type="' + typeAttr + '"]';
                } else {
                    s.fallback = buildCss(el);
                }

                return s;
            }

            // ── Compute semantic intent (Step 5) ────────────────────
            function computeIntent(el) {
                var tag = el.tagName.toLowerCase();
                var text = (el.textContent || '').trim().slice(0, 60);
                var intent = { action_label: el.getAttribute('aria-label') || text };
                if (tag === 'a' || el.getAttribute('href')) {
                    intent.type = 'navigation';
                    intent.expected_navigation = true;
                } else if (tag === 'button' || el.getAttribute('role') === 'button') {
                    var tp = el.getAttribute('type');
                    if (tp === 'submit' || el.closest('form')) {
                        intent.type = 'submit_form';
                        intent.expected_navigation = true;
                    } else {
                        intent.type = 'action';
                        intent.expected_navigation = false;
                    }
                } else if (tag === 'input' || tag === 'textarea' || tag === 'select') {
                    intent.type = 'input';
                    intent.expected_navigation = false;
                } else {
                    intent.type = 'interaction';
                    intent.expected_navigation = false;
                }
                return intent;
            }

            // ── Build fingerprint ────────────────────────────────────
            function fp(el) {
                if (!el || !el.tagName) return {};
                var attrs = {};
                for (var i = 0; i < (el.attributes || []).length; i++) {
                    var a = el.attributes[i];
                    if (!_frameworkAttrRe.test(a.name)) attrs[a.name] = a.value;
                }
                var direct = ownText(el);
                var full = (el.textContent || '').trim().slice(0, 200);
                return {
                    tag_name: el.tagName.toLowerCase(),
                    element_id: (el.id && !_dynIdRe.test(el.id)) ? el.id : '',
                    class_names: Array.from(el.classList || []),
                    text_content: direct || full,
                    attributes: attrs,
                    css_selector: buildCss(el),
                    xpath: relativeXPath(el),
                    aria_label: el.getAttribute('aria-label') || '',
                    role: el.getAttribute('role') || '',
                    parent_tag: el.parentElement ? el.parentElement.tagName.toLowerCase() : '',
                    sibling_index: el.parentElement ? Array.from(el.parentElement.children).indexOf(el) : 0,
                    nth_of_type: getNthOfType(el),
                    data_testid: el.getAttribute('data-testid') || '',
                    placeholder: el.getAttribute('placeholder') || '',
                    name: el.getAttribute('name') || '',
                    href: el.getAttribute('href') || '',
                    selectors: computeSelectors(el)
                };
            }

            // ── Click capture — promote to interactive parent ────────
            document.addEventListener('click', (e) => {
                if (e.target.closest('#__assertion_menu') ||
                    e.target.closest('#__assertion_fab') ||
                    e.target.id === '__assertion_highlight' ||
                    e.target.id === '__assertion_mode_banner' ||
                    e.target.id === '__assertion_fab' ||
                    e.target.id === '__assertion_menu' ||
                    window.__assertionLayerInjected && window.__assertionMode) return;
                var target = getInteractiveParent(e.target);
                console.log('__RECORDER__:' + JSON.stringify({
                    action: 'click',
                    fingerprint: fp(target),
                    intent: computeIntent(target),
                    url: window.location.href,
                    click_x: Math.round(e.clientX),
                    click_y: Math.round(e.clientY)
                }));
            }, true);

            // Track last type step per element to avoid duplicate from change after paste/input
            var _lastTypeKey = null;
            function recordType(el, value, action) {
                if (!el || (!el.tagName || (el.tagName !== 'INPUT' && el.tagName !== 'TEXTAREA' && el.tagName !== 'SELECT'))) return;
                action = action || 'type';
                if (el.tagName === 'SELECT') action = 'select';
                if (el.type === 'checkbox') action = el.checked ? 'check' : 'uncheck';
                if (el.tagName === 'INPUT' && action === 'type') {
                    var key = (el.id || el.name || el.placeholder || '') + ':' + value;
                    if (_lastTypeKey === key) return;
                    _lastTypeKey = key;
                }
                console.log('__RECORDER__:' + JSON.stringify({
                    action: action,
                    value: value,
                    fingerprint: fp(el),
                    url: window.location.href
                }));
            }

            // Paste: record immediately after paste (change fires only on blur, so paste was missed)
            var _inputDebounce = {};
            document.addEventListener('paste', (e) => {
                var el = e.target;
                if (el.tagName !== 'INPUT' && el.tagName !== 'TEXTAREA') return;
                if (window.__assertionLayerInjected && window.__assertionMode) return;
                var id = el.id || el.name || (el.placeholder && el.placeholder.slice(0, 20)) || ('el_' + Math.random());
                clearTimeout(_inputDebounce[id]);
                _inputDebounce[id] = undefined;
                setTimeout(function() {
                    recordType(el, el.value || '', 'type');
                }, 0);
            }, true);

            // Input (typing): debounce so we get one step per batch; paste is handled above
            document.addEventListener('input', (e) => {
                var el = e.target;
                if (el.tagName !== 'INPUT' && el.tagName !== 'TEXTAREA') return;
                if (window.__assertionLayerInjected && window.__assertionMode) return;
                var id = el.id || el.name || (el.placeholder && el.placeholder.slice(0, 20)) || ('el_' + Math.random());
                if (_inputDebounce[id] !== undefined) clearTimeout(_inputDebounce[id]);
                _inputDebounce[id] = setTimeout(function() {
                    recordType(el, el.value || '', 'type');
                    delete _inputDebounce[id];
                }, 400);
            }, true);

            // Change: only for SELECT and checkbox (text inputs handled by input/paste)
            document.addEventListener('change', (e) => {
                const el = e.target;
                if (el.tagName === 'SELECT') {
                    console.log('__RECORDER__:' + JSON.stringify({
                        action: 'select',
                        value: el.value || '',
                        fingerprint: fp(el),
                        url: window.location.href
                    }));
                    return;
                }
                if (el.type === 'checkbox' || el.type === 'radio') {
                    console.log('__RECORDER__:' + JSON.stringify({
                        action: el.checked ? 'check' : 'uncheck',
                        value: el.value || '',
                        fingerprint: fp(el),
                        url: window.location.href
                    }));
                }
            }, true);

            // Keyboard capture (Enter, Tab, Escape)
            document.addEventListener('keydown', (e) => {
                if (window.__assertionMode) return;
                if (['Enter', 'Tab', 'Escape'].includes(e.key)) {
                    console.log('__RECORDER__:' + JSON.stringify({
                        action: 'keypress',
                        value: e.key,
                        fingerprint: fp(e.target),
                        url: window.location.href
                    }));
                }
            }, true);

            // Scroll capture (debounced — one step per scroll gesture)
            var _scrollTimer = null;
            var _scrollTarget = null;
            window.addEventListener('scroll', (e) => {
                if (window.__assertionLayerInjected && window.__assertionMode) return;
                clearTimeout(_scrollTimer);
                _scrollTarget = e.target === document ? document.documentElement : e.target;
                _scrollTimer = setTimeout(function() {
                    var el = _scrollTarget || document.documentElement;
                    console.log('__RECORDER__:' + JSON.stringify({
                        action: 'scroll',
                        fingerprint: fp(el === document.documentElement ? document.body : el),
                        value: JSON.stringify({
                            scrollX: Math.round(window.scrollX),
                            scrollY: Math.round(window.scrollY)
                        }),
                        url: window.location.href
                    }));
                    _scrollTarget = null;
                }, 300);
            }, true);

            console.log('__RECORDER_READY__');
        })();
        """
        await page.context.add_init_script(recorder_js)
        # Also evaluate on the current page
        await page.evaluate(recorder_js)

    @staticmethod
    async def _wait_for_browser_close(page) -> None:
        """Block until the page / browser is closed by the user."""
        try:
            await page.wait_for_event("close", timeout=0)
        except Exception:
            # Browser closed or disconnected
            pass
