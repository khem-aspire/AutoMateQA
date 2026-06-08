/**
 * network_recorder.js — shadows fetch + XMLHttpRequest during recording
 * so the assertion UI can read request/response bodies, headers, and status.
 *
 * MUST be injected BEFORE any app code. BrowserManager uses addInitScript
 * for this.
 *
 * Exposes window.__aqaNetwork with:
 *   callsForCurrentStep()   -> array of call records (most recent first)
 *   clearCurrentStep()      -> empty the buffer
 *   on(cb)                  -> subscribe to new calls (returns unsubscribe fn)
 *
 * Each call record:
 *   {
 *     id, method, url, status, startedAt, durationMs,
 *     requestHeaders, requestBody (string | object | null),
 *     responseHeaders, responseBody (string | object | null),
 *     responseBodyTruncated (bool),
 *   }
 *
 * Redaction applied to headers + JSON bodies for display only:
 *   - keys matching /password|token|apikey|api_key|authorization|secret|bearer/i
 *     have their values replaced with "***".
 */
(function () {
  "use strict";
  if (window.__aqaNetworkInstalled) return;
  window.__aqaNetworkInstalled = true;

  const BODY_CAP_BYTES = 256 * 1024;
  const REDACT_RE = /^(password|token|apikey|api_key|authorization|secret|bearer|cookie|set-cookie)$/i;

  let buffer = [];
  const subscribers = new Set();
  let nextId = 1;

  function notify(record) {
    for (const cb of subscribers) {
      try { cb(record); } catch (_) {}
    }
  }

  function redactObject(value) {
    if (Array.isArray(value)) return value.map(redactObject);
    if (value && typeof value === "object") {
      const out = {};
      for (const [k, v] of Object.entries(value)) {
        out[k] = REDACT_RE.test(k) ? "***" : redactObject(v);
      }
      return out;
    }
    return value;
  }

  function redactHeaders(hdrs) {
    const out = {};
    for (const [k, v] of Object.entries(hdrs || {})) {
      out[k] = REDACT_RE.test(k) ? "***" : v;
    }
    return out;
  }

  function capBody(text) {
    if (text == null) return { body: null, truncated: false };
    const bytes = new Blob([text]).size;
    if (bytes <= BODY_CAP_BYTES) return { body: text, truncated: false };
    return { body: text.slice(0, BODY_CAP_BYTES), truncated: true };
  }

  function tryParseJson(text) {
    if (typeof text !== "string" || !text) return text;
    try { return JSON.parse(text); } catch (_) { return text; }
  }

  function parseHeaderString(raw) {
    const out = {};
    if (!raw) return out;
    for (const line of raw.trim().split(/[\r\n]+/)) {
      const idx = line.indexOf(":");
      if (idx > -1) out[line.slice(0, idx).trim()] = line.slice(idx + 1).trim();
    }
    return out;
  }

  // -------------------- fetch shim --------------------
  const origFetch = window.fetch;
  if (typeof origFetch === "function") {
    window.fetch = async function (input, init) {
      const req = input instanceof Request ? input : new Request(input, init || {});
      const startedAt = Date.now();

      let reqBody = null;
      if (init && init.body != null) {
        if (typeof init.body === "string") reqBody = init.body;
        else if (init.body instanceof FormData || init.body instanceof URLSearchParams) {
          reqBody = init.body.toString();
        }
      }

      const reqHeaders = {};
      req.headers.forEach((v, k) => { reqHeaders[k] = v; });

      const record = {
        id: nextId++,
        method: req.method,
        url: req.url,
        status: 0,
        startedAt,
        durationMs: 0,
        requestHeaders: redactHeaders(reqHeaders),
        requestBody: tryParseJson(redactBodyText(reqBody)),
        responseHeaders: {},
        responseBody: null,
        responseBodyTruncated: false,
      };
      buffer.unshift(record);

      let response;
      try {
        response = await origFetch.call(this, input, init);
      } catch (err) {
        record.durationMs = Date.now() - startedAt;
        record.status = 0;
        notify(record);
        throw err;
      }

      record.status = response.status;
      record.durationMs = Date.now() - startedAt;
      const rHdrs = {};
      response.headers.forEach((v, k) => { rHdrs[k] = v; });
      record.responseHeaders = redactHeaders(rHdrs);

      // Read the body without consuming the original stream.
      try {
        const clone = response.clone();
        const text = await clone.text();
        const capped = capBody(text);
        record.responseBody = tryParseJson(redactBodyText(capped.body));
        record.responseBodyTruncated = capped.truncated;
      } catch (_) {
        record.responseBody = null;
      }

      notify(record);
      return response;
    };
  }

  // -------------------- XMLHttpRequest shim --------------------
  const OrigXHR = window.XMLHttpRequest;
  if (typeof OrigXHR === "function") {
    function ShimXHR() {
      const xhr = new OrigXHR();
      let method = "GET";
      let url = "";
      let reqHeaders = {};
      let reqBody = null;
      let startedAt = 0;
      let record = null;

      const origOpen = xhr.open;
      xhr.open = function (m, u) {
        method = m;
        url = u;
        return origOpen.apply(xhr, arguments);
      };

      const origSetHdr = xhr.setRequestHeader;
      xhr.setRequestHeader = function (k, v) {
        reqHeaders[k] = v;
        return origSetHdr.apply(xhr, arguments);
      };

      const origSend = xhr.send;
      xhr.send = function (body) {
        startedAt = Date.now();
        if (body != null) {
          if (typeof body === "string") reqBody = body;
          else if (body instanceof FormData || body instanceof URLSearchParams) {
            reqBody = body.toString();
          }
        }
        record = {
          id: nextId++,
          method,
          url,
          status: 0,
          startedAt,
          durationMs: 0,
          requestHeaders: redactHeaders(reqHeaders),
          requestBody: tryParseJson(redactBodyText(reqBody)),
          responseHeaders: {},
          responseBody: null,
          responseBodyTruncated: false,
        };
        buffer.unshift(record);

        xhr.addEventListener("loadend", () => {
          record.durationMs = Date.now() - startedAt;
          record.status = xhr.status;
          record.responseHeaders = redactHeaders(
            parseHeaderString(xhr.getAllResponseHeaders())
          );
          try {
            const capped = capBody(xhr.responseText || "");
            record.responseBody = tryParseJson(redactBodyText(capped.body));
            record.responseBodyTruncated = capped.truncated;
          } catch (_) {
            record.responseBody = null;
          }
          notify(record);
        });
        return origSend.apply(xhr, arguments);
      };

      return xhr;
    }
    ShimXHR.prototype = OrigXHR.prototype;
    window.XMLHttpRequest = ShimXHR;
  }

  // Heuristic redaction for string bodies: mask common JWT / Bearer patterns.
  function redactBodyText(text) {
    if (typeof text !== "string") return text;
    return text
      .replace(/Bearer\s+[A-Za-z0-9._\-]+/gi, "Bearer ***")
      .replace(/eyJ[A-Za-z0-9._\-]{20,}/g, "eyJ***");
  }

  window.__aqaNetwork = {
    callsForCurrentStep() { return buffer.slice(); },
    clearCurrentStep() { buffer = []; },
    on(cb) {
      subscribers.add(cb);
      return () => subscribers.delete(cb);
    },
  };
})();
