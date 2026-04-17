/**
 * recorder_v2.js — Extended event capture for AutoMateQA.
 *
 * Adds: drag-and-drop, file upload, right-click, extended keyboard,
 * form submit capture.
 *
 * Injected AFTER the base recorder script. Uses the same fp() and
 * console.log("__RECORDER__:...") protocol.
 */
(function () {
    if (window.__recorderV2Injected) return;
    window.__recorderV2Injected = true;

    // Reuse base recorder's fp() when available (exposed on window by base script).
    // Falls back to minimal fingerprint if base hasn't loaded yet.
    function getFp(el) {
        if (window.__recorderFp) return window.__recorderFp(el);
        if (!el || !el.tagName) return {};
        var attrs = {};
        for (var i = 0; i < (el.attributes || []).length; i++) {
            var a = el.attributes[i];
            attrs[a.name] = a.value;
        }
        return {
            tag_name: el.tagName.toLowerCase(),
            element_id: el.id || '',
            class_names: Array.from(el.classList || []),
            text_content: (el.textContent || '').trim().slice(0, 200),
            attributes: attrs,
            css_selector: el.tagName.toLowerCase(),
            xpath: '',
            aria_label: el.getAttribute('aria-label') || '',
            role: el.getAttribute('role') || '',
            parent_tag: el.parentElement ? el.parentElement.tagName.toLowerCase() : '',
            sibling_index: 0,
            nth_of_type: 0,
            data_testid: el.getAttribute('data-testid') || '',
            placeholder: el.getAttribute('placeholder') || '',
            name: el.getAttribute('name') || '',
            href: el.getAttribute('href') || '',
            selectors: {}
        };
    }

    // ── Drag & Drop ──
    var _dragSource = null;
    document.addEventListener('dragstart', function (e) {
        _dragSource = e.target;
    }, true);

    document.addEventListener('drop', function (e) {
        if (!_dragSource) return;
        if (window.__assertionMode) return;
        e.preventDefault();
        console.log('__RECORDER__:' + JSON.stringify({
            action: 'drag_and_drop',
            fingerprint: getFp(_dragSource),
            value: JSON.stringify({
                drop_x: Math.round(e.clientX),
                drop_y: Math.round(e.clientY)
            }),
            intent: { type: 'drag_and_drop' },
            url: window.location.href
        }));
        _dragSource = null;
    }, true);

    // ── File Upload ──
    document.addEventListener('change', function (e) {
        var el = e.target;
        if (el.tagName !== 'INPUT' || el.type !== 'file') return;
        if (!el.files || el.files.length === 0) return;
        var names = Array.from(el.files).map(function (f) { return f.name; });
        console.log('__RECORDER__:' + JSON.stringify({
            action: 'file_upload',
            value: JSON.stringify(names),
            fingerprint: getFp(el),
            url: window.location.href
        }));
    }, true);

    // ── Right-Click (contextmenu) ──
    document.addEventListener('contextmenu', function (e) {
        if (window.__assertionMode) return;
        if (e.target.closest && (
            e.target.closest('#__assertion_menu') ||
            e.target.closest('#__assertion_fab'))) return;
        console.log('__RECORDER__:' + JSON.stringify({
            action: 'right_click',
            fingerprint: getFp(e.target),
            click_x: Math.round(e.clientX),
            click_y: Math.round(e.clientY),
            url: window.location.href
        }));
    }, true);

    // ── Extended Keyboard ──
    var _extendedKeys = [
        'ArrowUp', 'ArrowDown', 'ArrowLeft', 'ArrowRight',
        'Backspace', 'Delete', 'Space', ' '
    ];
    document.addEventListener('keydown', function (e) {
        if (window.__assertionMode) return;
        // Base recorder handles Enter, Tab, Escape
        if (['Enter', 'Tab', 'Escape'].includes(e.key)) return;
        if (!_extendedKeys.includes(e.key)) return;
        console.log('__RECORDER__:' + JSON.stringify({
            action: 'keypress',
            value: e.key === ' ' ? 'Space' : e.key,
            fingerprint: getFp(e.target),
            url: window.location.href
        }));
    }, true);

    console.log('__RECORDER_V2_READY__');
})();
