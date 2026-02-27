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
    `;
    document.head.appendChild(style);
  }

  // -------------------------------------------------------------------------
  // Floating Assertion Button (FAB)
  // -------------------------------------------------------------------------
  let fabEl = null;

  function createFAB() {
    if (fabEl) return;
    injectStyles();

    fabEl = document.createElement("button");
    fabEl.id = "__assertion_fab";
    fabEl.textContent = "🎯";
    fabEl.title = "Toggle Assertion Mode (Ctrl+Shift+A)";

    fabEl.addEventListener("click", (e) => {
      e.preventDefault();
      e.stopPropagation();
      toggleAssertionMode();
    });

    document.body.appendChild(fabEl);
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
    // ESC exits assertion mode and closes menu
    if (e.key === "Escape") {
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
  // Ready signal
  // -------------------------------------------------------------------------
  console.log("__ASSERTION_LAYER_READY__");
})();
