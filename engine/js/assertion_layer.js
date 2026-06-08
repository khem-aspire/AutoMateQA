/**
 * Assertion Layer – injected into every page via Playwright's addInitScript.
 *
 * Provides:
 *  1. Floating assertion toolbar button (always visible, works on all OS).
 *  2. Custom right-click context menu for adding assertions.
 *  3. Ctrl+Shift+A keyboard shortcut (cross-platform; avoids Mac ALT issues).
 *  4. Communication back to Python via the exposed `__assertion_bridge` binding.
 *
 * Mac Note: ALT/Option on Mac produces special characters, so we use
 *           Ctrl+Shift+A instead, plus a floating button as the primary UX.
 */
(function () {
  "use strict";

  // -------------------------------------------------------------------------
  // Guard: only inject once
  // -------------------------------------------------------------------------
  if (window.__assertionLayerInjected) return;
  window.__assertionLayerInjected = true;

  // -------------------------------------------------------------------------
  // Detect macOS
  // -------------------------------------------------------------------------
  const isMac = /Mac|iPhone|iPad|iPod/i.test(navigator.platform || navigator.userAgent);

  // -------------------------------------------------------------------------
  // Assertion types offered to the user
  // -------------------------------------------------------------------------
  const ASSERTION_TYPES = [
    { label: "✅ Visible", value: "visible" },
    { label: "🚫 Hidden", value: "hidden" },
    { label: "📝 Text Equals", value: "text_equals" },
    { label: "🔍 Text Contains", value: "text_contains" },
    { label: "🔣 Matches Pattern", value: "matches_pattern" },
    { label: "🏷️ Attribute Equals", value: "attribute_equals" },
    { label: "📌 Exists", value: "exists" },
  ];

  // -------------------------------------------------------------------------
  // API assertion constants (Phase 1)
  // -------------------------------------------------------------------------
  const API_TARGETS = [
    { value: "status",             label: "Status" },
    { value: "request_jsonpath",   label: "Request Body (JSONPath)" },
    { value: "response_jsonpath",  label: "Response Body (JSONPath)" },
    { value: "request_header",     label: "Request Header" },
    { value: "response_header",    label: "Response Header" },
    { value: "response_schema",    label: "Response Schema" },
    { value: "response_time_ms",   label: "Response Time (ms)" },
  ];
  const API_OPS = [
    "equals","not_equals","contains","not_contains","matches_regex",
    "exists","not_exists","gt","gte","lt","lte",
  ];

  // -------------------------------------------------------------------------
  // State
  // -------------------------------------------------------------------------
  let assertionMode = false;   // true while assertion mode is active
  let menuEl = null;           // the floating context menu
  let targetElement = null;    // element the user right-clicked / clicked

  // -------------------------------------------------------------------------
  // Helpers – build an element fingerprint
  // -------------------------------------------------------------------------
  const _frameworkAttrRe = /^(data-v-|data-reactid|_ngcontent|_nghost)/;
  const _dynIdRe = /[0-9a-f]{8}-|[0-9a-f]{12}|^f_|^\d{6,}/;

  function fingerprint(el) {
    if (!el || el === document || el === document.documentElement) {
      return {};
    }

    const attrs = {};
    for (const attr of el.attributes || []) {
      if (!_frameworkAttrRe.test(attr.name)) attrs[attr.name] = attr.value;
    }

    function buildCss(node) {
      const tag = node.tagName.toLowerCase();
      if (node.id && !_dynIdRe.test(node.id)) return tag + "#" + node.id;
      const cls = Array.from(node.classList || []);
      if (cls.length) {
        let css = tag;
        cls.forEach((c) => (css += "." + c));
        return css;
      }
      const p = node.parentElement;
      if (p) {
        let pCss = p.tagName.toLowerCase();
        if (p.id && !_dynIdRe.test(p.id)) pCss += "#" + p.id;
        else {
          const pCls = Array.from(p.classList || []);
          if (pCls.length) pCls.forEach((c) => (pCss += "." + c));
        }
        if (pCss !== p.tagName.toLowerCase()) {
          return pCss + " > " + tag + ":nth-child(" +
            (Array.from(p.children).indexOf(node) + 1) + ")";
        }
      }
      return tag;
    }

    function relativeXPath(node) {
      if (!node || node.nodeType !== 1) return "";
      const parts = [];
      let cur = node;
      let depth = 0;
      while (cur && cur.nodeType === 1 && depth < 3) {
        const tag = cur.tagName.toLowerCase();
        let idx = 1;
        let sib = cur.previousElementSibling;
        while (sib) {
          if (sib.tagName.toLowerCase() === tag) idx++;
          sib = sib.previousElementSibling;
        }
        parts.unshift(`${tag}[${idx}]`);
        cur = cur.parentElement;
        depth++;
      }
      return "//" + parts.join("/");
    }

    function ownText(node) {
      let t = "";
      for (let i = 0; i < node.childNodes.length; i++) {
        if (node.childNodes[i].nodeType === 3) t += node.childNodes[i].textContent;
      }
      return t.trim();
    }

    function computeSelectors(node) {
      const s = {};
      const tag = node.tagName.toLowerCase();
      const text = (node.textContent || "").trim().slice(0, 60);

      const tid = node.getAttribute("data-testid");
      if (tid) s.preferred = `[data-testid="${tid}"]`;

      for (const ca of ["data-cy", "data-test", "data-qa"]) {
        const cv = node.getAttribute(ca);
        if (cv) { s[s.preferred ? "data_cy" : "preferred"] = `[${ca}="${cv}"]`; break; }
      }

      let role = node.getAttribute("role") || "";
      if (!role && tag === "button") role = "button";
      if (!role && tag === "a") role = "link";
      if (role) {
        const aname = node.getAttribute("aria-label") || (text.length < 50 ? text : "");
        if (aname) s.role = `role=${role}[name="${aname.replace(/"/g, '\\"')}"]`;
      }

      const nameAttr = node.getAttribute("name");
      if (nameAttr) s.name = `${tag}[name="${nameAttr}"]`;

      const ph = node.getAttribute("placeholder");
      if (ph) s.placeholder = `[placeholder="${ph}"]`;

      if (text && text.length <= 40)
        s.text = `${tag}:has-text("${text.replace(/"/g, '\\"')}")`;

      s.fallback = buildCss(node);
      return s;
    }

    const direct = ownText(el);
    const full = (el.textContent || "").trim().slice(0, 200);

    return {
      tag_name: el.tagName.toLowerCase(),
      element_id: (el.id && !_dynIdRe.test(el.id)) ? el.id : "",
      class_names: Array.from(el.classList || []),
      text_content: direct || full,
      attributes: attrs,
      css_selector: buildCss(el),
      xpath: relativeXPath(el),
      aria_label: el.getAttribute("aria-label") || "",
      role: el.getAttribute("role") || "",
      parent_tag: el.parentElement ? el.parentElement.tagName.toLowerCase() : "",
      sibling_index: el.parentElement
        ? Array.from(el.parentElement.children).indexOf(el) : 0,
      nth_of_type: el.parentElement
        ? Array.from(el.parentElement.children)
          .filter((c) => c.tagName === el.tagName).indexOf(el) : 0,
      data_testid: el.getAttribute("data-testid") || "",
      placeholder: el.getAttribute("placeholder") || "",
      name: el.getAttribute("name") || "",
      href: el.getAttribute("href") || "",
      selectors: computeSelectors(el),
    };
  }

  // -------------------------------------------------------------------------
  // Inject global styles once
  // -------------------------------------------------------------------------
  function injectStyles() {
    if (document.getElementById("__assertion_styles")) return;
    const style = document.createElement("style");
    style.id = "__assertion_styles";
    style.textContent = `
      @keyframes assertMenuFadeIn {
        from { opacity: 0; transform: translateY(-4px); }
        to   { opacity: 1; transform: translateY(0); }
      }
      @keyframes assertBtnPulse {
        0%, 100% { box-shadow: 0 2px 12px rgba(30, 144, 255, 0.5); }
        50%      { box-shadow: 0 4px 24px rgba(30, 144, 255, 0.8); }
      }
      #__assertion_fab {
        position: fixed;
        bottom: 20px;
        right: 20px;
        z-index: 2147483647;
        width: 48px;
        height: 48px;
        border-radius: 50%;
        border: none;
        background: linear-gradient(135deg, #1e1e2e, #1e90ff);
        color: #ffffff;
        font-size: 22px;
        cursor: pointer;
        display: flex;
        align-items: center;
        justify-content: center;
        box-shadow: 0 2px 12px rgba(30, 144, 255, 0.4);
        transition: transform 0.15s, box-shadow 0.15s, background 0.3s;
        user-select: none;
        -webkit-user-select: none;
      }
      #__assertion_fab:hover {
        transform: scale(1.1);
        box-shadow: 0 4px 20px rgba(30, 144, 255, 0.6);
      }
      #__assertion_fab.active {
        background: linear-gradient(135deg, #000000, #1e90ff);
        animation: assertBtnPulse 1.5s ease-in-out infinite;
      }
      #__api_assertion_fab {
        transition: transform 0.15s, box-shadow 0.15s;
        user-select: none;
        -webkit-user-select: none;
      }
      #__api_assertion_fab:hover {
        transform: scale(1.1);
        box-shadow: 0 4px 20px rgba(43, 193, 124, 0.6);
      }
    `;
    document.head.appendChild(style);
  }

  // -------------------------------------------------------------------------
  // Floating Assertion Button (FAB)
  // -------------------------------------------------------------------------
  let fabEl = null;
  let apiFabEl = null;

  function createFAB() {
    if (fabEl) return;
    injectStyles();

    fabEl = document.createElement("button");
    fabEl.id = "__assertion_fab";
    fabEl.textContent = "🎯";
    fabEl.title = "Toggle DOM Assertion Mode (Ctrl+Shift+A)";
    fabEl.addEventListener("click", (e) => {
      e.preventDefault(); e.stopPropagation(); toggleAssertionMode();
    });
    document.body.appendChild(fabEl);

    apiFabEl = document.createElement("button");
    apiFabEl.id = "__api_assertion_fab";
    apiFabEl.textContent = "🌐";
    apiFabEl.title = "Open Network Assertion Panel";
    apiFabEl.style.cssText = `
      position: fixed; bottom: 80px; right: 20px; z-index: 2147483647;
      width: 48px; height: 48px; border-radius: 50%; border: none;
      background: linear-gradient(135deg, #1e1e2e, #2bc17c);
      color: #fff; font-size: 20px; cursor: pointer;
      box-shadow: 0 2px 12px rgba(43, 193, 124, 0.4);
    `;
    apiFabEl.addEventListener("click", (e) => {
      e.preventDefault(); e.stopPropagation();
      toggleNetworkPanel();
    });
    document.body.appendChild(apiFabEl);
  }

  // Reliable FAB creation: handle all timing scenarios
  function ensureFAB() {
    if (document.body) {
      createFAB();
    } else {
      // Poll until body exists (handles edge cases in SPAs)
      const interval = setInterval(() => {
        if (document.body) {
          clearInterval(interval);
          createFAB();
        }
      }, 100);
    }
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', ensureFAB);
  } else {
    // DOM already loaded (script injected late or via evaluate)
    ensureFAB();
  }

  // -------------------------------------------------------------------------
  // Toggle assertion mode
  // -------------------------------------------------------------------------
  function toggleAssertionMode() {
    assertionMode = !assertionMode;
    // Expose to window so the recorder script can check it
    window.__assertionMode = assertionMode;
    if (assertionMode) {
      showModeBanner();
      if (fabEl) {
        fabEl.classList.add("active");
        fabEl.textContent = "⏸️";
        fabEl.title = "Exit Assertion Mode (Ctrl+Shift+A or ESC)";
      }
    } else {
      hideModeBanner();
      hideHighlight();
      removeMenu();
      if (fabEl) {
        fabEl.classList.remove("active");
        fabEl.textContent = "🎯";
        fabEl.title = "Toggle Assertion Mode (Ctrl+Shift+A)";
      }
    }
  }

  // -------------------------------------------------------------------------
  // Custom Context Menu
  // -------------------------------------------------------------------------
  function createMenu(x, y) {
    removeMenu();
    injectStyles();

    menuEl = document.createElement("div");
    menuEl.id = "__assertion_menu";
    Object.assign(menuEl.style, {
      position: "fixed",
      left: `${x}px`,
      top: `${y}px`,
      zIndex: "2147483647",
      background: "#1e1e2e",
      color: "#cdd6f4",
      border: "1px solid #45475a",
      borderRadius: "8px",
      padding: "6px 0",
      fontFamily: "'Segoe UI', system-ui, sans-serif",
      fontSize: "13px",
      boxShadow: "0 8px 24px rgba(0,0,0,0.45)",
      minWidth: "200px",
      animation: "assertMenuFadeIn 0.12s ease-out",
    });

    // Ensure menu doesn't go off-screen
    const menuWidth = 220;
    const menuHeight = ASSERTION_TYPES.length * 32 + 50;
    if (x + menuWidth > window.innerWidth) {
      menuEl.style.left = `${window.innerWidth - menuWidth - 10}px`;
    }
    if (y + menuHeight > window.innerHeight) {
      menuEl.style.top = `${window.innerHeight - menuHeight - 10}px`;
    }

    // Header
    const header = document.createElement("div");
    header.textContent = "🎯 Add Assertion";
    Object.assign(header.style, {
      padding: "6px 14px 8px",
      fontWeight: "600",
      fontSize: "12px",
      color: "#a6adc8",
      borderBottom: "1px solid #313244",
      marginBottom: "4px",
      letterSpacing: "0.5px",
      textTransform: "uppercase",
    });
    menuEl.appendChild(header);

    ASSERTION_TYPES.forEach((at) => {
      const item = document.createElement("div");
      item.textContent = at.label;
      Object.assign(item.style, {
        padding: "7px 14px",
        cursor: "pointer",
        transition: "background 0.1s",
      });
      item.addEventListener("mouseenter", () => {
        item.style.background = "#313244";
      });
      item.addEventListener("mouseleave", () => {
        item.style.background = "transparent";
      });
      item.addEventListener("click", (e) => {
        e.stopPropagation();
        e.preventDefault();
        handleAssertionChoice(at.value);
      });
      menuEl.appendChild(item);
    });

    document.body.appendChild(menuEl);
  }

  function removeMenu() {
    if (menuEl && menuEl.parentElement) {
      menuEl.parentElement.removeChild(menuEl);
    }
    menuEl = null;
  }

  // -------------------------------------------------------------------------
  // Check if element is part of the assertion UI
  // -------------------------------------------------------------------------
  function isAssertionUI(el) {
    if (!el) return false;
    return !!(
      el.id === "__assertion_fab" ||
      el.id === "__assertion_menu" ||
      el.id === "__assertion_highlight" ||
      el.id === "__assertion_mode_banner" ||
      el.id === "__assertion_input_modal" ||
      el.closest("#__assertion_menu") ||
      el.closest("#__assertion_fab") ||
      el.closest("#__assertion_input_modal")
    );
  }

  // -------------------------------------------------------------------------
  // Custom in-page input modal (replaces window.prompt which Playwright
  // auto-dismisses, causing prompt-based assertions to be silently lost)
  // -------------------------------------------------------------------------
  let inputModalEl = null;

  function showInputModal(title, prefill, callback) {
    removeInputModal();

    inputModalEl = document.createElement("div");
    inputModalEl.id = "__assertion_input_modal";
    Object.assign(inputModalEl.style, {
      position: "fixed",
      inset: "0",
      zIndex: "2147483647",
      display: "flex",
      alignItems: "center",
      justifyContent: "center",
      background: "rgba(0,0,0,0.5)",
      fontFamily: "'Segoe UI', system-ui, sans-serif",
    });

    const card = document.createElement("div");
    Object.assign(card.style, {
      background: "#1e1e2e",
      border: "1px solid #45475a",
      borderRadius: "12px",
      padding: "20px 24px",
      width: "380px",
      boxShadow: "0 16px 48px rgba(0,0,0,0.5)",
      animation: "assertMenuFadeIn 0.15s ease-out",
    });

    const label = document.createElement("div");
    label.textContent = title;
    Object.assign(label.style, {
      color: "#cdd6f4",
      fontSize: "14px",
      fontWeight: "600",
      marginBottom: "12px",
    });
    card.appendChild(label);

    const input = document.createElement("input");
    input.type = "text";
    input.value = prefill || "";
    Object.assign(input.style, {
      width: "100%",
      boxSizing: "border-box",
      padding: "10px 12px",
      borderRadius: "8px",
      border: "1px solid #45475a",
      background: "#313244",
      color: "#cdd6f4",
      fontSize: "14px",
      outline: "none",
    });
    card.appendChild(input);

    const btnRow = document.createElement("div");
    Object.assign(btnRow.style, {
      display: "flex",
      justifyContent: "flex-end",
      gap: "8px",
      marginTop: "16px",
    });

    function makeBtn(text, primary) {
      const btn = document.createElement("button");
      btn.textContent = text;
      Object.assign(btn.style, {
        padding: "8px 18px",
        borderRadius: "8px",
        border: "none",
        fontSize: "13px",
        fontWeight: "600",
        cursor: "pointer",
        background: primary ? "#89b4fa" : "#45475a",
        color: primary ? "#1e1e2e" : "#cdd6f4",
      });
      return btn;
    }

    const cancelBtn = makeBtn("Cancel", false);
    const okBtn = makeBtn("OK", true);
    btnRow.appendChild(cancelBtn);
    btnRow.appendChild(okBtn);
    card.appendChild(btnRow);

    inputModalEl.appendChild(card);
    document.body.appendChild(inputModalEl);
    input.focus();

    function finish(value) {
      removeInputModal();
      callback(value);
    }

    okBtn.addEventListener("click", (e) => {
      e.stopImmediatePropagation();
      finish(input.value);
    });
    cancelBtn.addEventListener("click", (e) => {
      e.stopImmediatePropagation();
      finish(null);
    });
    input.addEventListener("keydown", (e) => {
      if (e.key === "Enter") { e.stopImmediatePropagation(); finish(input.value); }
      if (e.key === "Escape") { e.stopImmediatePropagation(); finish(null); }
    });
    inputModalEl.addEventListener("click", (e) => {
      if (e.target === inputModalEl) { e.stopImmediatePropagation(); finish(null); }
    });
  }

  function removeInputModal() {
    if (inputModalEl && inputModalEl.parentElement) {
      inputModalEl.parentElement.removeChild(inputModalEl);
    }
    inputModalEl = null;
  }

  // -------------------------------------------------------------------------
  // Handle assertion selection
  // -------------------------------------------------------------------------
  function finishAssertion(assertionType, expectedValue, attributeName) {
    const payload = {
      action: "add_assertion",
      assertion_type: assertionType,
      fingerprint: fingerprint(targetElement),
      value: expectedValue,
      attribute_name: attributeName,
      timestamp: new Date().toISOString(),
    };
    sendToBackend(payload);
    showToast(`✅ Assertion added: ${assertionType}`);
  }

  function handleAssertionChoice(assertionType) {
    removeMenu();

    const needsValue = [
      "text_equals",
      "text_contains",
      "matches_pattern",
      "attribute_equals",
    ];

    if (!needsValue.includes(assertionType)) {
      finishAssertion(assertionType, "", "");
      return;
    }

    if (assertionType === "attribute_equals") {
      showInputModal("Enter attribute name (e.g. href, class)", "", (attrName) => {
        if (attrName === null) return;
        showInputModal(`Enter expected value for "${attrName}"`, "", (val) => {
          if (val === null) return;
          finishAssertion(assertionType, val, attrName);
        });
      });
    } else {
      const labels = {
        text_equals: "Enter expected text",
        text_contains: "Enter text to search for",
        matches_pattern: "Enter regex pattern",
      };
      const el = targetElement;
      const prefill = assertionType === "text_equals"
        ? (el ? (el.textContent || "").trim().slice(0, 200) : "")
        : "";
      showInputModal(labels[assertionType] || "Enter expected value", prefill, (val) => {
        if (val === null) return;
        finishAssertion(assertionType, val, "");
      });
    }
  }

  // -------------------------------------------------------------------------
  // Communication back to Python backend
  // -------------------------------------------------------------------------
  function sendToBackend(payload) {
    const json = JSON.stringify(payload);

    // Always send via console (guaranteed to work)
    console.log("__ASSERTION__:" + json);

    // Also try the exposed binding (faster, more reliable)
    try {
      if (typeof window.__assertion_bridge === "function") {
        window.__assertion_bridge(json);
      }
    } catch (e) {
      // Binding not available – console fallback already sent
    }
  }

  // -------------------------------------------------------------------------
  // Toast notification
  // -------------------------------------------------------------------------
  function showToast(msg) {
    const toast = document.createElement("div");
    toast.textContent = msg;
    Object.assign(toast.style, {
      position: "fixed",
      bottom: "80px",
      right: "24px",
      zIndex: "2147483647",
      background: "#a6e3a1",
      color: "#1e1e2e",
      padding: "10px 20px",
      borderRadius: "8px",
      fontFamily: "'Segoe UI', system-ui, sans-serif",
      fontSize: "13px",
      fontWeight: "600",
      boxShadow: "0 4px 16px rgba(0,0,0,0.3)",
      animation: "assertMenuFadeIn 0.15s ease-out",
    });
    document.body.appendChild(toast);
    setTimeout(() => {
      toast.style.transition = "opacity 0.3s";
      toast.style.opacity = "0";
      setTimeout(() => toast.remove(), 300);
    }, 2000);
  }

  // -------------------------------------------------------------------------
  // Highlight element on hover in assertion mode
  // -------------------------------------------------------------------------
  let highlightOverlay = null;

  function showHighlight(el) {
    if (isAssertionUI(el)) return;

    if (!highlightOverlay) {
      highlightOverlay = document.createElement("div");
      highlightOverlay.id = "__assertion_highlight";
      Object.assign(highlightOverlay.style, {
        position: "fixed",
        zIndex: "2147483646",
        border: "2px solid #89b4fa",
        background: "rgba(137, 180, 250, 0.12)",
        pointerEvents: "none",
        borderRadius: "3px",
        transition: "all 0.08s ease-out",
      });
      document.body.appendChild(highlightOverlay);
    }
    const rect = el.getBoundingClientRect();
    Object.assign(highlightOverlay.style, {
      left: `${rect.left}px`,
      top: `${rect.top}px`,
      width: `${rect.width}px`,
      height: `${rect.height}px`,
      display: "block",
    });
  }

  function hideHighlight() {
    if (highlightOverlay) highlightOverlay.style.display = "none";
  }

  // -------------------------------------------------------------------------
  // Assertion mode indicator banner
  // -------------------------------------------------------------------------
  let modeBanner = null;

  function showModeBanner() {
    if (modeBanner) return;
    modeBanner = document.createElement("div");
    modeBanner.id = "__assertion_mode_banner";
    const shortcut = isMac ? "⌃⇧A" : "Ctrl+Shift+A";
    modeBanner.textContent = `🎯 ASSERTION MODE — Click an element  |  Press ESC or ${shortcut} to exit`;
    Object.assign(modeBanner.style, {
      position: "fixed",
      top: "0",
      left: "0",
      right: "0",
      zIndex: "2147483647",
      background: "linear-gradient(90deg, rgba(0, 0, 0, 0.75), rgba(30, 144, 255, 0.75))",
      color: "#ffffff",
      textAlign: "center",
      padding: "6px",
      fontFamily: "'Segoe UI', system-ui, sans-serif",
      fontSize: "12px",
      fontWeight: "700",
      letterSpacing: "0.5px",
      textShadow: "0 1px 2px rgba(0, 0, 0, 0.8)",
      boxShadow: "0 2px 6px rgba(0, 0, 0, 0.2)",
      pointerEvents: "none",
    });
    document.body.appendChild(modeBanner);
  }

  function hideModeBanner() {
    if (modeBanner) {
      modeBanner.remove();
      modeBanner = null;
    }
  }

  // -------------------------------------------------------------------------
  // Event listeners
  // -------------------------------------------------------------------------

  // Right-click → custom assertion menu (when in assertion mode)
  // stopImmediatePropagation prevents the recorder script (same element,
  // same phase) from seeing assertion-mode interactions.
  document.addEventListener("contextmenu", (e) => {
    if (!assertionMode) return;
    if (isAssertionUI(e.target)) return;

    e.preventDefault();
    e.stopImmediatePropagation();
    targetElement = e.target;
    createMenu(e.clientX, e.clientY);
  }, true);

  // Ctrl+Shift+A → toggle assertion mode (works on Mac and Windows/Linux)
  document.addEventListener("keydown", (e) => {
    if (e.ctrlKey && e.shiftKey && e.key.toLowerCase() === "a") {
      e.preventDefault();
      e.stopImmediatePropagation();
      toggleAssertionMode();
    }
    // ESC closes network panel or exits assertion mode
    if (e.key === "Escape") {
      if (networkPanelEl) {
        e.preventDefault();
        e.stopImmediatePropagation();
        closeNetworkPanel();
        return;
      }
      if (assertionMode || menuEl) {
        e.preventDefault();
        e.stopImmediatePropagation();
        assertionMode = false;
        window.__assertionMode = false;
        hideModeBanner();
        hideHighlight();
        removeMenu();
        if (fabEl) {
          fabEl.classList.remove("active");
          fabEl.textContent = "🎯";
          fabEl.title = "Toggle Assertion Mode (Ctrl+Shift+A)";
        }
      }
    }
  }, true);

  // In assertion mode: highlight on hover
  document.addEventListener("mousemove", (e) => {
    if (!assertionMode) return;
    if (isAssertionUI(e.target)) return;
    showHighlight(e.target);
  }, true);

  // In assertion mode: click an element → open assertion menu
  document.addEventListener("click", (e) => {
    // Close menu if clicking outside
    if (menuEl && !menuEl.contains(e.target)) {
      removeMenu();
    }

    if (!assertionMode) return;
    if (isAssertionUI(e.target)) return;

    e.preventDefault();
    e.stopImmediatePropagation();
    targetElement = e.target;
    createMenu(e.clientX, e.clientY);
  }, true);

  // -------------------------------------------------------------------------
  // Network Assertion Panel
  // -------------------------------------------------------------------------
  let networkPanelEl = null;
  let networkDimEl = null;
  let selectedCallId = null;
  let networkView = "list"; // "list" | "builder"

  function openNetworkPanel() {
    if (networkPanelEl) return;
    // Dim overlay – blocks all page interaction while panel is open
    networkDimEl = document.createElement("div");
    networkDimEl.id = "__aqa_network_dim";
    networkDimEl.style.cssText = `
      position:fixed;inset:0;z-index:2147483640;
      background:rgba(0,0,0,0.45);
      display:flex;align-items:flex-end;justify-content:flex-start;
      padding:16px;box-sizing:border-box;pointer-events:all;
    `;
    networkDimEl.innerHTML = `
      <span style="color:rgba(255,255,255,0.55);font:12px -apple-system,sans-serif;
        background:rgba(0,0,0,0.5);padding:4px 10px;border-radius:20px;user-select:none;">
        ⏸ Recording paused — close panel to resume
      </span>`;
    document.body.appendChild(networkDimEl);
    window.__networkPanelOpen = true;
    renderNetworkPanel();
  }

  function closeNetworkPanel() {
    if (networkPanelEl) { networkPanelEl.remove(); networkPanelEl = null; }
    if (networkDimEl) { networkDimEl.remove(); networkDimEl = null; }
    window.__networkPanelOpen = false;
    networkView = "list";
  }

  function toggleNetworkPanel() {
    if (networkPanelEl) { closeNetworkPanel(); return; }
    openNetworkPanel();
  }

  function showCallListView() {
    networkView = "list";
    selectedCallId = null;
    const search = document.getElementById("__aqa_np_search");
    const calls = document.getElementById("__aqa_np_calls");
    const builder = document.getElementById("__aqa_np_builder");
    if (search) search.style.display = "";
    if (calls) calls.style.display = "";
    if (builder) builder.style.display = "none";
    renderCallList();
  }

  function showBuilderView(call) {
    networkView = "builder";
    const search = document.getElementById("__aqa_np_search");
    const calls = document.getElementById("__aqa_np_calls");
    const builder = document.getElementById("__aqa_np_builder");
    if (search) search.style.display = "none";
    if (calls) calls.style.display = "none";
    if (builder) builder.style.display = "";
    renderBuilder(call);
  }

  function renderNetworkPanel() {
    networkPanelEl = document.createElement("div");
    networkPanelEl.id = "__aqa_network_panel";
    networkPanelEl.style.cssText = `
      position: fixed; top: 60px; right: 20px; width: 520px;
      max-height: 80vh; overflow-y: auto; z-index: 2147483647;
      background: #1e1e2e; color: #eee; border-radius: 10px;
      box-shadow: 0 8px 24px rgba(0,0,0,0.4); padding: 12px;
      font: 13px -apple-system, BlinkMacSystemFont, sans-serif;
    `;
    networkPanelEl.innerHTML = `
      <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:8px;">
        <strong>🌐 Network (this step)</strong>
        <button id="__aqa_np_close" style="background:none;border:none;color:#eee;cursor:pointer;font-size:18px;">×</button>
      </div>
      <input id="__aqa_np_search" placeholder="Search: method or URL" style="width:100%;padding:6px;border-radius:6px;border:1px solid #333;background:#2a2a3e;color:#eee;margin-bottom:8px;">
      <div id="__aqa_np_calls"></div>
      <div id="__aqa_np_builder" style="display:none;"></div>
    `;
    document.body.appendChild(networkPanelEl);

    // Stop events at bubble phase so target handlers still fire inside the
    // panel, but the event never reaches document-level recorder listeners.
    for (const evt of ["click", "dblclick", "input", "change", "keydown", "keyup", "keypress", "scroll", "wheel", "contextmenu", "mousedown", "mouseup"]) {
      networkPanelEl.addEventListener(evt, (e) => e.stopPropagation(), false);
    }

    document.getElementById("__aqa_np_close").onclick = closeNetworkPanel;
    document.getElementById("__aqa_np_search").addEventListener("input", renderCallList);
    renderCallList();
  }

  function renderCallList() {
    const calls = (window.__aqaNetwork && window.__aqaNetwork.callsForCurrentStep()) || [];
    const q = (document.getElementById("__aqa_np_search").value || "").toLowerCase();
    const filtered = calls.filter(c =>
      !q || (`${c.method} ${c.url}`).toLowerCase().includes(q)
    );
    const html = filtered.slice(0, 50).map(c => `
      <div data-call-id="${c.id}" class="__aqa_call_row"
           style="padding:6px 8px;border-bottom:1px solid #2a2a3e;cursor:pointer;${selectedCallId===c.id?'background:#2a2a3e;':''}">
        <span style="color:#2bc17c;">${c.method}</span>
        ${truncate(c.url, 60)}
        <span style="float:right;color:${c.status>=200&&c.status<400?'#2bc17c':'#e74c3c'}">${c.status} · ${c.durationMs}ms</span>
      </div>
    `).join("") || `<div style="padding:12px;color:#888;">No calls captured yet.</div>`;
    document.getElementById("__aqa_np_calls").innerHTML = html;
    for (const row of document.querySelectorAll(".__aqa_call_row")) {
      row.addEventListener("click", () => {
        selectedCallId = Number(row.dataset.callId);
        showBuilderView(calls.find(c => c.id === selectedCallId));
      });
    }
  }

  function truncate(s, n) { return s.length > n ? s.slice(0, n) + "…" : s; }

  // -------------------------------------------------------------------------
  // Network Assertion Builder
  // -------------------------------------------------------------------------
  function normalizeUrlToTemplate(url) {
    try {
      const path = new URL(url, location.href).pathname;
      return path.split("/").map(seg => {
        if (!seg) return seg;
        if (/^\d+$/.test(seg)) return "{id}";
        if (/^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$/.test(seg)) return "{uuid}";
        if (/^[0-9a-fA-F]{24}$/.test(seg)) return "{oid}";
        if (/^\d{4}-\d{2}-\d{2}$/.test(seg)) return "{date}";
        return seg;
      }).join("/");
    } catch (_) { return url; }
  }
  function queryKeysOf(url) {
    try {
      const u = new URL(url, location.href);
      return [...new Set([...u.searchParams.keys()])].sort();
    } catch (_) { return []; }
  }

  function renderBuilder(call) {
    const b = document.getElementById("__aqa_np_builder");
    if (!call) { b.innerHTML = ""; return; }
    const template = normalizeUrlToTemplate(call.url);
    const qKeys = queryKeysOf(call.url);

    b.innerHTML = `
      <div style="display:flex;align-items:center;gap:10px;margin-bottom:10px;">
        <button id="__aqa_np_back" style="background:none;border:1px solid #555;color:#aaa;padding:4px 10px;border-radius:6px;cursor:pointer;font-size:12px;">← Back</button>
        <span style="color:#2bc17c;font-weight:bold;">${escape(call.method)}</span>
        <span style="color:#aaa;font-size:12px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;">${truncate(call.url, 48)} · <span style="color:${call.status>=200&&call.status<400?'#2bc17c':'#e74c3c'}">${call.status}</span> · ${call.durationMs}ms</span>
      </div>
      <div style="margin-bottom:8px;">
        <label style="font-size:12px;color:#aaa;">Path template</label>
        <input id="__aqa_tmpl" value="${escape(template)}" style="width:100%;padding:6px;background:#2a2a3e;color:#eee;border:1px solid #333;border-radius:6px;margin-top:2px;">
      </div>
      <div style="display:flex;gap:6px;flex-wrap:wrap;margin-bottom:8px;">
        ${API_TARGETS.map(t => `<button class="__aqa_tab" data-target="${t.value}" style="padding:4px 8px;background:#2a2a3e;color:#eee;border:1px solid #333;border-radius:6px;cursor:pointer;">${t.label}</button>`).join("")}
      </div>
      <div id="__aqa_tab_body"></div>
      <div style="display:flex;justify-content:flex-end;gap:8px;margin-top:10px;">
        <button id="__aqa_save" style="background:#2bc17c;color:#000;border:none;padding:6px 12px;border-radius:6px;cursor:pointer;">Save Assertion</button>
      </div>
    `;
    document.getElementById("__aqa_np_back").onclick = showCallListView;
    document.querySelectorAll(".__aqa_tab").forEach(btn => btn.onclick = () => renderTab(btn.dataset.target, call, qKeys));
    renderTab("status", call, qKeys);
    document.getElementById("__aqa_save").onclick = () => saveFromBuilder(call, qKeys);
  }

  function escape(s) { return String(s).replace(/[&<>"]/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;","\"":"&quot;"}[c])); }

  let currentBuilder = { target: "status", op: "equals", expected: "", jsonpath: "", headerName: "", schema: null };
  const NO_EXPECTED_OPS = new Set(["exists", "not_exists"]);

  function renderTab(target, call, qKeys) {
    currentBuilder.target = target;
    currentBuilder.jsonpath = "";
    currentBuilder.headerName = "";
    currentBuilder.schema = null;
    const body = document.getElementById("__aqa_tab_body");
    if (target === "status") {
      body.innerHTML = `
        <div>Status: <strong>${call.status}</strong></div>
        ${renderOpAndExpected(["equals","not_equals"], "equals", String(call.status))}
      `;
      bindOpExpected();
    } else if (target === "response_jsonpath" || target === "request_jsonpath") {
      const src = target === "response_jsonpath" ? call.responseBody : call.requestBody;
      body.innerHTML = `
        <div style="max-height:220px;overflow:auto;background:#151521;padding:6px;border-radius:6px;">${renderJsonTree(src, "$")}</div>
        <div style="margin-top:6px;">JSONPath: <input id="__aqa_jp" style="width:70%;background:#2a2a3e;color:#eee;border:1px solid #333;padding:4px;"></div>
        ${renderOpAndExpected(API_OPS, "equals", "")}
      `;
      bindOpExpected();
      document.querySelectorAll(".__aqa_json_leaf").forEach(l => l.onclick = () => {
        document.getElementById("__aqa_jp").value = l.dataset.path;
        const inp = document.getElementById("__aqa_expected");
        if (inp && !NO_EXPECTED_OPS.has(currentBuilder.op)) inp.value = l.dataset.value;
        currentBuilder.jsonpath = l.dataset.path;
        if (!NO_EXPECTED_OPS.has(currentBuilder.op)) currentBuilder.expected = l.dataset.value;
      });
      document.getElementById("__aqa_jp").addEventListener("input", e => currentBuilder.jsonpath = e.target.value);
    } else if (target === "response_header" || target === "request_header") {
      const src = target === "response_header" ? call.responseHeaders : call.requestHeaders;
      const options = Object.keys(src || {}).map(h => `<option value="${escape(h)}">${escape(h)}</option>`).join("");
      body.innerHTML = `
        <div>Header name: <select id="__aqa_hdr" style="background:#2a2a3e;color:#eee;padding:4px;">${options}</select></div>
        ${renderOpAndExpected(API_OPS, "equals", "")}
      `;
      currentBuilder.headerName = Object.keys(src || {})[0] || "";
      document.getElementById("__aqa_hdr").addEventListener("change", e => currentBuilder.headerName = e.target.value);
      bindOpExpected();
    } else if (target === "response_schema") {
      const schema = deriveSchema(call.responseBody);
      currentBuilder.schema = schema;
      currentBuilder.op = "response_schema";
      body.innerHTML = `
        <div>Schema derived from response:</div>
        <textarea id="__aqa_schema" style="width:100%;height:180px;background:#151521;color:#eee;border:1px solid #333;border-radius:6px;">${escape(JSON.stringify(schema, null, 2))}</textarea>
      `;
      document.getElementById("__aqa_schema").addEventListener("input", e => {
        try { currentBuilder.schema = JSON.parse(e.target.value); } catch (_) {}
      });
    } else if (target === "response_time_ms") {
      body.innerHTML = `
        <div>Observed: <strong>${call.durationMs}ms</strong></div>
        ${renderOpAndExpected(["lt","lte","gt","gte","equals"], "lt", String(call.durationMs + 100))}
      `;
      bindOpExpected();
    }
  }

  function renderOpAndExpected(ops, defaultOp, defaultExpected) {
    currentBuilder.op = defaultOp || ops[0];
    currentBuilder.expected = defaultExpected || "";
    const noExp = NO_EXPECTED_OPS.has(currentBuilder.op);
    const opItems = ops.map(o =>
      `<div class="__aqa_op_item" data-op="${o}" style="padding:5px 10px;cursor:pointer;border-radius:4px;${o===currentBuilder.op?"background:#3a3a5e;color:#2bc17c;font-weight:bold;":""}">${o}</div>`
    ).join("");
    return `
      <div style="margin-top:8px;display:flex;align-items:center;gap:10px;flex-wrap:wrap;">
        <div style="display:flex;align-items:center;gap:6px;">
          <label style="color:#aaa;font-size:12px;">Op</label>
          <div id="__aqa_op_wrap" style="position:relative;display:inline-block;">
            <div id="__aqa_op" data-value="${escape(currentBuilder.op)}"
              style="background:#2a2a3e;color:#eee;border:1px solid #444;padding:4px 28px 4px 8px;border-radius:6px;cursor:pointer;min-width:110px;font-size:12px;user-select:none;position:relative;">
              ${escape(currentBuilder.op)}
              <span style="position:absolute;right:7px;top:50%;transform:translateY(-50%);color:#aaa;font-size:10px;">▼</span>
            </div>
            <div id="__aqa_op_menu" style="display:none;position:absolute;top:calc(100% + 2px);left:0;min-width:140px;background:#1e1e2e;border:1px solid #444;border-radius:6px;z-index:2147483647;box-shadow:0 4px 16px rgba(0,0,0,0.5);padding:4px 0;max-height:220px;overflow-y:auto;">
              ${opItems}
            </div>
          </div>
        </div>
        <div style="display:flex;align-items:center;gap:6px;flex:1;">
          <label style="color:#aaa;font-size:12px;">Expected</label>
          <input id="__aqa_expected" value="${escape(currentBuilder.expected)}"
            ${noExp ? "disabled" : ""}
            style="flex:1;background:#2a2a3e;color:#eee;border:1px solid #333;padding:4px 6px;border-radius:6px;font-size:12px;${noExp ? "opacity:0.4;" : ""}">
        </div>
      </div>`;
  }

  function _applyOp(op) {
    currentBuilder.op = op;
    const trigger = document.getElementById("__aqa_op");
    if (trigger) {
      trigger.dataset.value = op;
      trigger.childNodes[0].textContent = op + " ";
    }
    const menu = document.getElementById("__aqa_op_menu");
    if (menu) {
      menu.querySelectorAll(".__aqa_op_item").forEach(el => {
        const active = el.dataset.op === op;
        el.style.background = active ? "#3a3a5e" : "";
        el.style.color = active ? "#2bc17c" : "#eee";
        el.style.fontWeight = active ? "bold" : "";
      });
      menu.style.display = "none";
    }
    const ex = document.getElementById("__aqa_expected");
    if (ex) {
      const noExp = NO_EXPECTED_OPS.has(op);
      ex.disabled = noExp;
      ex.style.opacity = noExp ? "0.4" : "1";
      if (noExp) { ex.value = ""; currentBuilder.expected = ""; }
    }
  }

  let _opMenuCloseHandler = null;

  function bindOpExpected() {
    const trigger = document.getElementById("__aqa_op");
    const menu = document.getElementById("__aqa_op_menu");
    const exEl = document.getElementById("__aqa_expected");

    if (_opMenuCloseHandler) {
      document.removeEventListener("click", _opMenuCloseHandler, false);
      _opMenuCloseHandler = null;
    }

    if (trigger && menu) {
      trigger.addEventListener("click", e => {
        e.stopPropagation();
        menu.style.display = menu.style.display === "none" ? "block" : "none";
      });
      menu.querySelectorAll(".__aqa_op_item").forEach(item => {
        item.addEventListener("mouseenter", () => { if (item.dataset.op !== currentBuilder.op) item.style.background = "#2a2a3e"; });
        item.addEventListener("mouseleave", () => { if (item.dataset.op !== currentBuilder.op) item.style.background = ""; });
        item.addEventListener("click", e => { e.stopPropagation(); _applyOp(item.dataset.op); });
      });
      _opMenuCloseHandler = () => { menu.style.display = "none"; };
      document.addEventListener("click", _opMenuCloseHandler, false);
    }
    if (exEl) {
      exEl.addEventListener("input", e => { currentBuilder.expected = e.target.value; });
    }
  }

  function renderJsonTree(value, path) {
    if (value === null || typeof value !== "object") {
      return `<span class="__aqa_json_leaf" data-path="${escape(path)}" data-value="${escape(String(value))}" style="cursor:pointer;color:#8ec07c;">${escape(JSON.stringify(value))}</span>`;
    }
    if (Array.isArray(value)) {
      return "[<div style='margin-left:12px;'>" +
        value.map((v, i) => `${i}: ${renderJsonTree(v, `${path}[${i}]`)}`).join("<br>") +
        "</div>]";
    }
    const entries = Object.entries(value);
    return "{<div style='margin-left:12px;'>" +
      entries.map(([k, v]) => `<span style="color:#83a598;">"${escape(k)}"</span>: ${renderJsonTree(v, `${path}.${k}`)}`).join("<br>") +
      "</div>}";
  }

  function deriveSchema(v) {
    if (v === null) return { type: "null" };
    if (typeof v === "string") return { type: "string" };
    if (typeof v === "number") return Number.isInteger(v) ? { type: "integer" } : { type: "number" };
    if (typeof v === "boolean") return { type: "boolean" };
    if (Array.isArray(v)) return { type: "array", items: v.length ? deriveSchema(v[0]) : {} };
    if (typeof v === "object") {
      const props = {}; const required = [];
      for (const [k, val] of Object.entries(v)) { props[k] = deriveSchema(val); if (val !== null) required.push(k); }
      return { type: "object", properties: props, required };
    }
    return {};
  }

  function saveFromBuilder(call, qKeys) {
    const template = document.getElementById("__aqa_tmpl").value;
    const payload = {
      assertion_type: "api_call",
      fingerprint: {},
      api_spec: {
        method: call.method,
        path_template: template,
        query_keys_present: qKeys,
        target: currentBuilder.target,
        op: currentBuilder.op,
        expected: currentBuilder.expected || "",
        jsonpath: currentBuilder.jsonpath || "",
        header_name: currentBuilder.headerName || "",
        expected_schema: currentBuilder.schema || {},
      },
      timestamp: String(Date.now()),
    };
    console.log("__ASSERTION__:" + JSON.stringify(payload));
    if (typeof window.__assertion_bridge === "function") {
      window.__assertion_bridge(JSON.stringify(payload));
    }
    // Visual feedback
    const s = document.getElementById("__aqa_save");
    if (s) { s.textContent = "Saved ✓"; setTimeout(() => s.textContent = "Save Assertion", 1200); }
  }

  // -------------------------------------------------------------------------
  // Ready signal
  // -------------------------------------------------------------------------
  console.log("__ASSERTION_LAYER_READY__");
})();
