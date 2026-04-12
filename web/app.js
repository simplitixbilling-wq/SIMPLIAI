/* ═══════════════════════════════════════════════════════════════
   SIMPLE_AI — Frontend Logic (pywebview bridge)
   ═══════════════════════════════════════════════════════════════ */

// ── Wait for pywebview API ────────────────────────────────────
let api = null;

const REQUIRED_API_METHODS = [
  'get_app_info',
  'get_model_status',
  'get_chats',
  'get_models',
  'new_chat',
  'start_monitor',
];

function hasRequiredApiMethods(candidate) {
  if (!candidate) return false;
  return REQUIRED_API_METHODS.every((name) => typeof candidate[name] === 'function');
}

async function waitForApi() {
  if (hasRequiredApiMethods(window.pywebview?.api)) {
    api = window.pywebview.api;
    return;
  }

  await new Promise((resolve, reject) => {
    const timeoutMs = 30000;
    const started = Date.now();
    let readyEventFired = false;

    const finish = () => {
      clearInterval(timer);
      resolve();
    };

    const timer = setInterval(() => {
      const candidate = window.pywebview?.api;
      if (hasRequiredApiMethods(candidate)) {
        api = candidate;
        finish();
        return;
      }
      // Once pywebviewready has fired, allow a small grace period for API method binding.
      if (readyEventFired && candidate && Date.now() - started > 5000) {
        const methods = Object.keys(candidate || {}).sort().join(', ');
        clearInterval(timer);
        reject(new Error(`pywebview API methods not fully bound. Found: [${methods}]`));
        return;
      }
      if (Date.now() - started > timeoutMs) {
        clearInterval(timer);
        reject(new Error('Timed out waiting for pywebview API'));
      }
    }, 50);

    window.addEventListener('pywebviewready', () => {
      readyEventFired = true;
    }, { once: true });
  });

  // Final guard in case methods changed between resolve and use.
  if (!hasRequiredApiMethods(api)) {
    throw new Error('pywebview API is available but required methods are missing');
  }
}

// ── State ─────────────────────────────────────────────────────
let currentChatId = null;
let sidebarVisible = true;
let attachedImageName = null;
let attachedDocName = null;
let contextLimitTokens = 0;
let actualContext = { used: 0, total: 0 };

function normalizeReplyTokenLimit(value) {
  const n = Number.parseInt(String(value ?? ''), 10);
  if (Number.isFinite(n) && n > 0) return n;
  return 0;
}

function applyReplyTokenLimitUi(value) {
  const normalized = normalizeReplyTokenLimit(value);
  const asString = String(normalized);
  const chatSelect = $('#chat-max-tokens');
  const settingsSelect = $('#s-max-tokens');

  if (chatSelect) {
    if (!Array.from(chatSelect.options).some(o => o.value === asString)) {
      chatSelect.value = '0';
    } else {
      chatSelect.value = asString;
    }
  }

  if (settingsSelect) {
    if (!Array.from(settingsSelect.options).some(o => o.value === asString)) {
      settingsSelect.value = normalized > 0 ? '512' : '0';
    } else {
      settingsSelect.value = asString;
    }
  }
}
let pendingExportTarget = null;
let isSpeaking = false;
let fullAccessGranted = true;

const FALLBACK_PIPER_VOICES = [
  // English US
  { id: 'en_US-lessac-medium', name: 'English US - Lessac', language: 'en_US', quality: 'medium' },
  { id: 'en_US-amy-medium', name: 'English US - Amy', language: 'en_US', quality: 'medium' },
  { id: 'en_US-bryce-medium', name: 'English US - Bryce', language: 'en_US', quality: 'medium' },
  { id: 'en_US-danny-low', name: 'English US - Danny (low)', language: 'en_US', quality: 'low' },
  { id: 'en_US-hfc_female-medium', name: 'English US - HFC Female', language: 'en_US', quality: 'medium' },
  { id: 'en_US-hfc_male-medium', name: 'English US - HFC Male', language: 'en_US', quality: 'medium' },
  { id: 'en_US-joe-medium', name: 'English US - Joe', language: 'en_US', quality: 'medium' },
  { id: 'en_US-john-medium', name: 'English US - John', language: 'en_US', quality: 'medium' },
  { id: 'en_US-kathleen-low', name: 'English US - Kathleen (low)', language: 'en_US', quality: 'low' },
  { id: 'en_US-kristin-medium', name: 'English US - Kristin', language: 'en_US', quality: 'medium' },
  { id: 'en_US-kusal-medium', name: 'English US - Kusal', language: 'en_US', quality: 'medium' },
  { id: 'en_US-ljspeech-medium', name: 'English US - LJSpeech', language: 'en_US', quality: 'medium' },
  { id: 'en_US-norman-medium', name: 'English US - Norman', language: 'en_US', quality: 'medium' },
  { id: 'en_US-ryan-medium', name: 'English US - Ryan', language: 'en_US', quality: 'medium' },
  // English UK
  { id: 'en_GB-alan-medium', name: 'English UK - Alan', language: 'en_GB', quality: 'medium' },
  { id: 'en_GB-alba-medium', name: 'English UK - Alba', language: 'en_GB', quality: 'medium' },
  { id: 'en_GB-aru-medium', name: 'English UK - Aru', language: 'en_GB', quality: 'medium' },
  { id: 'en_GB-cori-medium', name: 'English UK - Cori', language: 'en_GB', quality: 'medium' },
  { id: 'en_GB-jenny_dioco-medium', name: 'English UK - Jenny Dioco', language: 'en_GB', quality: 'medium' },
  { id: 'en_GB-northern_english_male-medium', name: 'English UK - Northern Male', language: 'en_GB', quality: 'medium' },
  { id: 'en_GB-southern_english_female-medium', name: 'English UK - Southern Female', language: 'en_GB', quality: 'medium' },
  // German
  { id: 'de_DE-thorsten-medium', name: 'German - Thorsten', language: 'de_DE', quality: 'medium' },
  { id: 'de_DE-thorsten_emotional-medium', name: 'German - Thorsten Emotional', language: 'de_DE', quality: 'medium' },
  { id: 'de_DE-eva_k-medium', name: 'German - Eva K', language: 'de_DE', quality: 'medium' },
  { id: 'de_DE-karlsson-low', name: 'German - Karlsson (low)', language: 'de_DE', quality: 'low' },
  { id: 'de_DE-kerstin-low', name: 'German - Kerstin (low)', language: 'de_DE', quality: 'low' },
  { id: 'de_DE-ramona-low', name: 'German - Ramona (low)', language: 'de_DE', quality: 'low' },
  { id: 'de_DE-pavoque-low', name: 'German - Pavoque (low)', language: 'de_DE', quality: 'low' },
  // French
  { id: 'fr_FR-siwis-medium', name: 'French - Siwis', language: 'fr_FR', quality: 'medium' },
  { id: 'fr_FR-gilles-low', name: 'French - Gilles (low)', language: 'fr_FR', quality: 'low' },
  { id: 'fr_FR-tom-medium', name: 'French - Tom', language: 'fr_FR', quality: 'medium' },
  // Spanish
  { id: 'es_ES-carlfm-medium', name: 'Spanish - Carlfm', language: 'es_ES', quality: 'medium' },
  { id: 'es_ES-davefx-medium', name: 'Spanish - Davefx', language: 'es_ES', quality: 'medium' },
  { id: 'es_ES-sharvard-medium', name: 'Spanish - Sharvard', language: 'es_ES', quality: 'medium' },
  // Hindi
  { id: 'hi_IN-pratham-medium', name: 'Hindi India - Pratham', language: 'hi_IN', quality: 'medium' },
  // Tamil
  { id: 'ta_IN-kani-medium', name: 'Tamil India - Kani', language: 'ta_IN', quality: 'medium' },
];

async function waitForApiMethod(methodName, timeoutMs = 4000) {
  const started = Date.now();
  while (Date.now() - started < timeoutMs) {
    const candidate = window.pywebview?.api;
    if (candidate && typeof candidate[methodName] === 'function') {
      api = candidate;
      return true;
    }
    await new Promise((resolve) => setTimeout(resolve, 80));
  }
  return false;
}

async function populateVoiceOptions(selectedFromSettings = '') {
  const sel = $('#s-voice');
  const help = $('#s-voice-help');
  if (!sel) return;
  sel.innerHTML = '<option value="">Default system voice</option>';

  let selected = selectedFromSettings || '';
  try {
    const r = await api.list_tts_voices();
    const voices = Array.isArray(r?.voices) ? r.voices : [];
    if (!selected && r?.selected_voice_id) selected = String(r.selected_voice_id);

    voices.forEach((v) => {
      const id = String(v.id || '');
      if (!id) return;
      const opt = document.createElement('option');
      opt.value = id;
      opt.textContent = v.label || v.name || id;
      sel.appendChild(opt);
    });

    if (selected) sel.value = selected;
    if (help) help.textContent = voices.length
      ? `Detected ${voices.length} voice(s). Selected voice is used for Speak and WAV export.`
      : 'No voices detected. Using default system voice.';
  } catch (e) {
    if (help) help.textContent = 'Voice list unavailable. Using default system voice.';
  }
}

async function populatePiperCatalog() {
  const sel = $('#s-piper-voice-catalog');
  const status = $('#s-piper-download-status');
  if (!sel) return;
  sel.innerHTML = '<option value="">Select a Piper voice...</option>';
  sel.disabled = true;

  const applyCatalog = (voices) => {
    voices.forEach((v) => {
      const id = String(v.id || '');
      if (!id) return;
      const opt = document.createElement('option');
      opt.value = id;
      opt.textContent = `${v.name || id} [${v.language || ''} / ${v.quality || ''}]`;
      sel.appendChild(opt);
    });
  };

  try {
    let voices = [];
    const methodReady = await waitForApiMethod('list_free_piper_voices', 5000);
    if (methodReady) {
      const r = await api.list_free_piper_voices();
      voices = Array.isArray(r?.voices) ? r.voices : [];
    }
    if (!voices.length) {
      voices = FALLBACK_PIPER_VOICES;
      if (status) status.textContent = `Available: ${voices.length} free voices (offline fallback list)`;
    } else if (status) {
      status.textContent = `Available: ${voices.length} free voices`;
    }
    applyCatalog(voices);
  } catch (e) {
    applyCatalog(FALLBACK_PIPER_VOICES);
    if (status) status.textContent = `Available: ${FALLBACK_PIPER_VOICES.length} free voices (fallback list)`;
  } finally {
    sel.disabled = false;
  }
}

// ── Development Mode (set to false before production) ──────────
const DEVELOPMENT_MODE = true;  // Set to false to hide reload button

// ── Helpers ───────────────────────────────────────────────────

function $(sel) { return document.querySelector(sel); }
function $$(sel) { return document.querySelectorAll(sel); }

function showToast(msg, kind = 'info', duration = 4000) {
  const container = $('#toast-container');
  const el = document.createElement('div');
  el.className = `toast ${kind}`;
  el.textContent = msg;
  container.appendChild(el);
  setTimeout(() => el.remove(), duration);
}

function openModal(id) { document.getElementById(id).classList.remove('hidden'); }
function closeModal(id) { document.getElementById(id).classList.add('hidden'); }
// Make closeModal global for onclick handlers in HTML
window.closeModal = closeModal;

/**
 * Styled confirm dialog — replaces browser confirm().
 * Returns a Promise<boolean>.
 */
function showPrompt(title = 'Rename', defaultValue = '') {
  return new Promise((resolve) => {
    const modal = document.getElementById('modal-prompt');
    document.getElementById('prompt-title').textContent = title;
    const input = document.getElementById('prompt-input');
    input.value = defaultValue;
    modal.classList.remove('hidden');
    input.focus();
    input.select();
    function cleanup() {
      modal.classList.add('hidden');
      document.getElementById('prompt-ok').removeEventListener('click', onOk);
      document.getElementById('prompt-cancel').removeEventListener('click', onCancel);
      input.removeEventListener('keydown', onKey);
      modal.removeEventListener('click', onBackdrop);
    }
    function onOk() { cleanup(); resolve(input.value); }
    function onCancel() { cleanup(); resolve(null); }
    function onKey(e) { if (e.key === 'Enter') onOk(); else if (e.key === 'Escape') onCancel(); }
    function onBackdrop(e) { if (e.target === modal) { cleanup(); resolve(null); } }
    document.getElementById('prompt-ok').addEventListener('click', onOk);
    document.getElementById('prompt-cancel').addEventListener('click', onCancel);
    input.addEventListener('keydown', onKey);
    modal.addEventListener('click', onBackdrop);
  });
}

function applyAccessLockUi(locked, statusText = '') {
  fullAccessGranted = !locked;

  const input = $('#user-input');
  const send = $('#btn-send');
  const agentInput = $('#agent-input');
  const agentSend = $('#btn-agent-send');

  if (input) input.disabled = locked;
  if (agentInput) agentInput.disabled = locked;
  if (send) send.disabled = locked;
  if (agentSend) agentSend.disabled = locked;

  if (locked && statusText) {
    setStatus(statusText, true);
  }

  updateSendButton();
}

async function ensureActivationGate(forcePrompt = false) {
  if (!api || typeof api.get_activation_status !== 'function') {
    return true;
  }

  let status = null;
  try {
    status = await api.get_activation_status();
    updateTrialPill(status);
  } catch (_) {
    return true;
  }

  if (!status?.requires_passkey) {
    applyAccessLockUi(false);
    return true;
  }

  if (!forcePrompt) {
    applyAccessLockUi(true, `Trial expired after ${status.days_used || 30} days. Activation required.`);
    return false;
  }

  let unlocked = false;
  let attempts = 0;
  while (!unlocked && attempts < 3) {
    const entered = await showPrompt('Trial expired - Enter passkey for full access', '');
    if (entered == null) break;

    try {
      const result = await api.activate_full_access(entered);
      if (result?.ok) {
        unlocked = true;
        applyAccessLockUi(false);
        updateTrialPill(result?.status || null);
        showToast('Activation successful. Full access unlocked.', 'success', 5000);
        break;
      }
      showToast(result?.error || 'Invalid passkey', 'error', 4500);
    } catch (e) {
      showToast(`Activation failed: ${e?.message || e}`, 'error', 4500);
    }
    attempts += 1;
  }

  if (!unlocked) {
    applyAccessLockUi(true, `Trial expired after ${status.days_used || 30} days. Activation required.`);
  }
  return unlocked;
}

function showConfirm(message, title = 'Confirm', okLabel = 'Delete') {
  return new Promise((resolve) => {
    const modal = document.getElementById('modal-confirm');
    document.getElementById('confirm-title').textContent = title;
    document.getElementById('confirm-message').textContent = message;
    const okBtn = document.getElementById('confirm-ok');
    okBtn.textContent = okLabel;
    // Destructive actions get red button
    const isDestructive = /delete|remove|clear/i.test(okLabel);
    okBtn.style.background = isDestructive ? 'var(--error, #e74c3c)' : '';
    modal.classList.remove('hidden');
    function cleanup() {
      modal.classList.add('hidden');
      okBtn.removeEventListener('click', onOk);
      document.getElementById('confirm-cancel').removeEventListener('click', onCancel);
      modal.removeEventListener('click', onBackdrop);
    }
    function onOk() { cleanup(); resolve(true); }
    function onCancel() { cleanup(); resolve(false); }
    function onBackdrop(e) { if (e.target === modal) { cleanup(); resolve(false); } }
    okBtn.addEventListener('click', onOk);
    document.getElementById('confirm-cancel').addEventListener('click', onCancel);
    modal.addEventListener('click', onBackdrop);
  });
}

function setStatus(text, active) {
  const el = $('#status-text');
  const pill = $('#status-bar');
  if (el) el.textContent = text;
  if (pill) pill.classList.toggle('active', !!active);
}

function updateTrialPill(status) {
  const pill = $('#trial-pill');
  const text = $('#trial-pill-text');
  if (!pill || !text) return;

  pill.classList.remove('warn', 'expired', 'activated');

  if (!status || typeof status !== 'object') {
    text.textContent = 'Trial: unknown';
    return;
  }

  if (status.is_activated) {
    pill.classList.add('activated');
    text.textContent = 'Activated';
    return;
  }

  const daysLeft = Number(status.days_left ?? 0);
  if (status.requires_passkey || daysLeft <= 0) {
    pill.classList.add('expired');
    text.textContent = 'Trial expired';
    return;
  }

  if (daysLeft <= 7) {
    pill.classList.add('warn');
  }
  text.textContent = `Free trial: ${daysLeft} day${daysLeft === 1 ? '' : 's'} left`;
}

/** Switch highlight.js stylesheet for dark/light theme */
function switchHljsTheme(theme) {
  const dark = document.getElementById('hljs-theme-dark');
  const light = document.getElementById('hljs-theme-light');
  if (!dark || !light) return;
  if (theme === 'light') {
    dark.disabled = true;
    light.disabled = false;
  } else {
    dark.disabled = false;
    light.disabled = true;
  }
}

function estimateTokens(text) {
  return Math.ceil((text || '').length / 4);
}

function updateContextBar() {
  const container = $('#context-bar-container');
  const fill = $('#context-fill');
  const labelEl = $('#context-label-text');
  if (!container || !fill || !labelEl) return;

  const details = $('#context-details');
  if (actualContext.total > 0) {
    const pct = Math.max(0, Math.min(100, Math.round((actualContext.used / actualContext.total) * 100)));
    container.style.display = '';
    labelEl.textContent = `${actualContext.used}/${actualContext.total} (${pct}%)`;
    fill.style.width = `${pct}%`;
    fill.style.background = pct >= 90
      ? 'var(--danger)'
      : (pct >= 75 ? 'var(--warning)' : 'var(--success)');
    if (details) {
      details.style.display = 'none';
    }
    return;
  }

  if (!contextLimitTokens || contextLimitTokens <= 0) {
    container.style.display = 'none';
    labelEl.textContent = '0/0 (0%)';
    fill.style.width = '0%';
    if (details) {
      details.style.display = 'none';
    }
    return;
  }

  let transcript = '';
  $$('#messages-container .message').forEach((el) => {
    let t = el.innerText || '';
    t = t.replace(/Copy|Regenerate|Branch|Speak|Export/g, '').trim();
    if (t) transcript += t + '\n';
  });

  const inputTokens = estimateTokens($('#user-input')?.value?.trim() || '');
  const usedTokens = Math.max(0, estimateTokens(transcript) + inputTokens);
  const pct = Math.max(0, Math.min(100, Math.round((usedTokens / contextLimitTokens) * 100)));

  container.style.display = '';
  labelEl.textContent = `${pct}%`;
  fill.style.width = `${pct}%`;
  fill.style.background = pct >= 90
    ? 'var(--danger)'
    : (pct >= 75 ? 'var(--warning)' : 'var(--success)');
}

function escapeHtml(text) {
  const d = document.createElement('div');
  d.textContent = text;
  return d.innerHTML;
}

function assistantActionsHtml() {
  return `
    <button onclick="copyMessage(this)">Copy</button>
    <button onclick="regenerateMessage()">Regenerate</button>
    <button onclick="branchConversation()">Branch</button>
    <button onclick="speakMessage(this)">Speak</button>
    <button onclick="exportMessage(this)">Export</button>
  `;
}

function exportMessage(btn) {
  const msgDiv = btn.closest('.message[data-msg-index]');
  if (!msgDiv || !currentChatId) {
    showToast('Cannot identify message to export', 'warning');
    return;
  }
  const idx = parseInt(msgDiv.getAttribute('data-msg-index'), 10);
  openExportModal({
    chatId: currentChatId,
    messageIndex: idx,
    label: `Export response #${idx}`,
  });
}

async function exportMessageWav(btn) {
  const msgDiv = btn.closest('.message[data-msg-index]');
  if (!msgDiv || !currentChatId) {
    showToast('Cannot identify message to export', 'warning');
    return;
  }
  const idx = parseInt(msgDiv.getAttribute('data-msg-index'), 10);
  if (Number.isNaN(idx)) {
    showToast('Invalid message index', 'warning');
    return;
  }

  const response = await api.export_assistant_message_wav(currentChatId, idx);
  if (response?.error) {
    showToast(response.error, 'error');
    return;
  }
  if (response?.path) {
    showToast(`✓ WAV exported to ${response.path}`, 'success', 5000);
  }
}

function openExportModal(target) {
  pendingExportTarget = target;
  const label = $('#export-target-label');
  if (label) {
    label.textContent = target?.label || 'Export target';
  }
  openModal('modal-export');
}

function preprocessRagCitations(text) {
  if (!text) return '';
  let out = text;

  // Convert one-line list into per-line markdown links.
  out = out.replace(/📄 Sources:\s*([^\n]+)/g, (_m, raw) => {
    const parts = raw.split(',').map(s => s.trim()).filter(Boolean);
    if (!parts.length) return '📄 Sources:';
    return '📄 Sources:\n' + parts.map(name => `- [${name}](rag://source/${encodeURIComponent(name)})`).join('\n');
  });

  // Convert bullet source lines into markdown links.
  out = out.replace(/(^|\n)-\s+([^\n]+)/g, (m, pfx, name) => {
    const trimmed = (name || '').trim();
    if (!trimmed) return m;
    if (trimmed.startsWith('[') && trimmed.includes('](rag://source/')) return m;
    return `${pfx}- [${trimmed}](rag://source/${encodeURIComponent(trimmed)})`;
  });

  // Single source fallback.
  out = out.replace(/📄 Source:\s*([^\n]+)/g, (_m, name) => {
    const trimmed = (name || '').trim();
    if (!trimmed) return '📄 Source:';
    return `📄 Source: [${trimmed}](rag://source/${encodeURIComponent(trimmed)})`;
  });

  return out;
}

/** Build horizontal scrollable source cards HTML */
function buildSourceCardsHtml(sources) {
  if (!sources || !sources.length) return '';
  let cards = '<div class="web-source-cards">';
  sources.forEach((s, i) => {
    let domain = '';
    try { domain = new URL(s.url).hostname.replace(/^www\./, ''); } catch (_) { domain = s.url; }
    const faviconUrl = `https://www.google.com/s2/favicons?sz=32&domain=${encodeURIComponent(domain)}`;
    cards += `<a class="web-source-card" href="${escapeHtml(s.url)}" target="_blank" rel="noopener noreferrer" title="${escapeHtml(s.title)}">` +
      `<div class="src-head">` +
        `<img class="src-favicon" src="${faviconUrl}" alt="" onerror="this.style.display='none'">` +
        `<span class="src-domain">${escapeHtml(domain)}</span>` +
        `<span class="src-index">${i + 1}</span>` +
      `</div>` +
      `<div class="src-title">${escapeHtml(s.title || 'Untitled')}</div>` +
      `<div class="src-snippet">${escapeHtml(s.snippet || '')}</div>` +
    `</a>`;
  });
  cards += '</div>';
  return cards;
}

/** Render markdown with optional web citation badges */
function renderMarkdownWithCitations(text, sources) {
  let html = renderMarkdown(text);
  if (sources && sources.length) {
    // Convert [1], [2] etc. into clickable citation badges
    html = html.replace(/\[(\d+)\]/g, (match, num) => {
      const idx = parseInt(num, 10) - 1;
      if (idx >= 0 && idx < sources.length) {
        const s = sources[idx];
        return `<a class="web-cite" href="${escapeHtml(s.url)}" target="_blank" rel="noopener noreferrer" title="${escapeHtml(s.title)}">${num}</a>`;
      }
      return match;
    });
  }
  return html;
}

/** Simple markdown → HTML (covers common patterns) */
function renderMarkdown(text) {
  // Use marked.js if available, otherwise fall back to manual rendering
  if (typeof marked !== 'undefined') {
    try {
      return renderMarkdownMarked(text);
    } catch (_e) {
      // Fall back to manual renderer on any error
    }
  }
  return renderMarkdownManual(text);
}

/** marked.js renderer with highlight.js code highlighting */
function renderMarkdownMarked(text) {
  const preprocessed = preprocessRagCitations(text);

  // Use marked.use() — the v12-recommended API
  marked.use({
    breaks: true,
    gfm: true,
    renderer: {
      // Override link rendering to handle rag:// and external links
      // marked v12 passes positional args: (href, title, text)
      link(href, title, linkText) {
        href = href || '';
        if (href.startsWith('rag://source/')) {
          const sourceName = decodeURIComponent(href.substring('rag://source/'.length));
          return `<a href="#" class="rag-source-link" data-source="${escapeHtml(sourceName)}">${linkText}</a>`;
        }
        if (/^https?:\/\//i.test(href)) {
          const titleAttr = title ? ` title="${escapeHtml(title)}"` : '';
          return `<a href="${escapeHtml(href)}" target="_blank" rel="noopener noreferrer"${titleAttr}>${linkText}</a>`;
        }
        return linkText || '';
      },

      // Override code blocks to add language header + copy button
      // marked v12 passes positional args: (text, lang, escaped)
      code(codeText, lang) {
        const raw = (codeText != null) ? String(codeText) : '';
        const language = (lang || '').split(/\s/)[0] || '';  // strip info string extras
        let highlighted = escapeHtml(raw);
        if (raw && typeof hljs !== 'undefined') {
          try {
            if (language && hljs.getLanguage(language)) {
              highlighted = hljs.highlight(raw, { language }).value;
            } else {
              highlighted = hljs.highlightAuto(raw).value;
            }
          } catch (_) { /* use escaped text */ }
        }
        const langLabel = language || 'code';
        return `<div class="code-block-wrapper">` +
          `<div class="code-block-header"><span class="code-lang">${escapeHtml(langLabel)}</span>` +
          `<button class="code-copy-btn" onclick="copyCodeBlock(this)">Copy</button></div>` +
          `<pre><code class="hljs lang-${escapeHtml(language)}">${highlighted}</code></pre></div>`;
      },
    },
  });

  return marked.parse(preprocessed);
}

/** Copy code block content to clipboard */
function copyCodeBlock(btn) {
  const wrapper = btn.closest('.code-block-wrapper');
  if (!wrapper) return;
  const code = wrapper.querySelector('code');
  if (!code) return;
  navigator.clipboard.writeText(code.textContent).then(() => {
    btn.textContent = 'Copied!';
    setTimeout(() => { btn.textContent = 'Copy'; }, 1500);
  }).catch(() => {
    btn.textContent = 'Failed';
    setTimeout(() => { btn.textContent = 'Copy'; }, 1500);
  });
}
window.copyCodeBlock = copyCodeBlock;

/** Fallback manual markdown → HTML (original renderer) */
function renderMarkdownManual(text) {
  let html = escapeHtml(preprocessRagCitations(text));

  // Convert markdown tables: match lines with pipes
  html = html.replace(/(\|[^\n]*\|(?:\n|\<br\>))+/g, (tableBlock) => {
    const textLines = tableBlock.replace(/\<br\>/g, '\n').split('\n').map(s => s.trim()).filter(s => s);
    
    if (textLines.length < 2) return tableBlock;
    
    try {
      let headerIdx = 0;
      let dataStart = 1;
      // Check if row 1 is separator (dashes)
      if (/^[\|\s\-:]+$/.test(textLines[1])) dataStart = 2;
      
      const getCells = (line) => line.split('|').map(c => c.trim()).filter(c => c && !c.match(/^-+$/));
      
      const headers = getCells(textLines[headerIdx]);
      if (headers.length === 0) return tableBlock;
      
      const rows = [];
      for (let i = dataStart; i < textLines.length && i < textLines.length; i++) {
        const cells = getCells(textLines[i]);
        if (cells.length > 0) rows.push(cells);
      }
      
      if (rows.length === 0) return tableBlock;
      
      let out = '<table class="md-table"><thead><tr>';
      headers.forEach(h => out += `<th>${h}</th>`);
      out += '</tr></thead><tbody>';
      rows.forEach(cells => {
        out += '<tr>';
        cells.forEach(c => out += `<td>${c}</td>`);
        out += '</tr>';
      });
      out += '</tbody></table>';
      return out;
    } catch (e) {
      return tableBlock;
    }
  });

  // Code blocks ``` ... ```
  html = html.replace(/```(\w*)\n([\s\S]*?)```/g, (_, lang, code) =>
    `<pre><code class="lang-${lang}">${code}</code></pre>`);

  // Inline code
  html = html.replace(/`([^`]+)`/g, '<code>$1</code>');

  // Headers
  html = html.replace(/^### (.+)$/gm, '<h3>$1</h3>');
  html = html.replace(/^## (.+)$/gm, '<h2>$1</h2>');
  html = html.replace(/^# (.+)$/gm, '<h1>$1</h1>');

  // Bold + italic
  html = html.replace(/\*\*\*(.+?)\*\*\*/g, '<strong><em>$1</em></strong>');
  html = html.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');
  html = html.replace(/\*(.+?)\*/g, '<em>$1</em>');

  // Markdown links
  html = html.replace(/\[([^\]]+)\]\(([^)]+)\)/g, (_m, label, href) => {
    if (href.startsWith('rag://source/')) {
      const sourceName = decodeURIComponent(href.substring('rag://source/'.length));
      return `<a href="#" class="rag-source-link" data-source="${escapeHtml(sourceName)}">${label}</a>`;
    }
    if (/^https?:\/\//i.test(href)) {
      return `<a href="${href}" target="_blank" rel="noopener noreferrer">${label}</a>`;
    }
    return label;
  });

  // Blockquote
  html = html.replace(/^&gt; (.+)$/gm, '<blockquote>$1</blockquote>');

  // Bullet lists
  html = html.replace(/^[-•] (.+)$/gm, '<li>$1</li>');
  html = html.replace(/(<li>.*<\/li>)/gs, '<ul>$1</ul>');
  // Collapse nested <ul>
  html = html.replace(/<\/ul>\s*<ul>/g, '');

  // Line breaks (but not inside pre)
  html = html.replace(/\n/g, '<br>');
  // Fix breaks inside pre
  html = html.replace(/<pre>([\s\S]*?)<\/pre>/g, (_, inner) =>
    '<pre>' + inner.replace(/<br>/g, '\n') + '</pre>');

  return html;
}

// ── Auto-resize textarea ──────────────────────────────────────
function autoResize(ta) {
  ta.style.height = 'auto';
  ta.style.height = Math.min(ta.scrollHeight, 150) + 'px';
}

// ═══ INITIALIZATION ═══════════════════════════════════════════

async function init() {
  await waitForApi();

  const safeStep = async (label, fn) => {
    try {
      return await fn();
    } catch (e) {
      console.error(`Init step failed: ${label}`, e);
      showToast(`${label} failed: ${e?.message || e}`, 'warn', 5000);
      return null;
    }
  };

  // Load app info first (needed for theme)
  const info = await safeStep('Loading app info', () => api.get_app_info()) || {
    config: { mode: 'READY' },
    system_ram: '?',
    gpu: { vram: '?' },
    theme: 'Dark',
  };
  updateTrialPill(info.activation || null);
  setStatus(`${info.config?.mode || 'READY'} | RAM: ${info.system_ram}GB | VRAM: ${info.gpu?.vram ?? '?'}GB`);
  document.documentElement.setAttribute('data-theme', String(info.theme || 'Dark').toLowerCase());
  switchHljsTheme(String(info.theme || 'Dark').toLowerCase());

  const startupSettings = await safeStep('Loading app settings', () => api.get_app_settings()) || {};
  applyReplyTokenLimitUi(startupSettings.max_response_tokens);

  const activationOk = await ensureActivationGate(true);
  if (!activationOk) {
    showToast('Trial expired. Enter passkey to use full features.', 'warn', 6000);
  }

  // Run independent tasks in parallel for faster load
  const [_, ms, chats] = await Promise.all([
    safeStep('Loading models', refreshModels),
    safeStep('Loading model status', () => api.get_model_status()),
    safeStep('Reading chats', () => api.get_chats()),
  ]);

  // Apply model status
  if (ms?.loaded) {
    contextLimitTokens = ms.n_ctx || 0;
    $('#loaded-model-label').textContent = `Loaded: ${ms.name || 'model'}`;
    $('#btn-load-model').textContent = 'Unload';
    $('#btn-load-model').disabled = false;
    $('#context-bar-container').style.display = '';
  } else {
    contextLimitTokens = 0;
    $('#context-bar-container').style.display = 'none';
  }

  // Load chats and select first
  await safeStep('Loading chats', refreshChats);
  let chatList = Array.isArray(chats) ? chats : [];
  if (chatList.length === 0) {
    const created = await safeStep('Creating initial chat', () => api.new_chat());
    currentChatId = created?.chat_id || null;
    await safeStep('Refreshing chats', refreshChats);
    if (currentChatId) {
      await safeStep('Selecting initial chat', () => selectChat(currentChatId));
    }
  } else {
    await safeStep('Selecting first chat', () => selectChat(chatList[0]));
  }

  setStatus(`${info.config?.mode || 'READY'} | RAM: ${info.system_ram}GB | VRAM: ${info.gpu?.vram ?? '?'}GB`);
  updateSendButton();
  updateContextBar();

  // Defer non-critical tasks (don't block UI)
  safeStep('Loading knowledge bases', refreshRag);
  safeStep('Starting system monitor', () => api.start_monitor());

  // Setup reload button (development mode)
  const reloadBtn = $('#btn-reload-app');
  if (reloadBtn && DEVELOPMENT_MODE) {
    reloadBtn.classList.remove('hidden');
    reloadBtn.onclick = () => location.reload();
  }

  // Setup mode toggle buttons for Chat vs Agent
  const chatModeBtn = $('#btn-chat-mode');
  const agentModeBtn = $('#btn-agent-mode');
  if (chatModeBtn && agentModeBtn) {
    chatModeBtn.addEventListener('click', () => {
      $('#chat-area').classList.remove('hidden');
      $('#agent-area').classList.add('hidden');
      chatModeBtn.classList.add('active');
      agentModeBtn.classList.remove('active');
    });
    agentModeBtn.addEventListener('click', () => {
      $('#chat-area').classList.add('hidden');
      $('#agent-area').classList.remove('hidden');
      chatModeBtn.classList.remove('active');
      agentModeBtn.classList.add('active');
      refreshInstructionList();
      refreshProcessedFiles();
    });
    chatModeBtn.classList.add('active');
    agentModeBtn.classList.remove('active');
  }
}

// ═══ MODELS ═══════════════════════════════════════════════════

async function refreshModels() {
  const models = await api.get_models();
  const sel = $('#model-select');
  sel.innerHTML = '';
  if (!Array.isArray(models) || models.length === 0) {
    sel.innerHTML = '<option value="">No models found</option>';
    $('#btn-load-model').disabled = true;
    return;
  }

  // Group: local GGUF first, then Ollama
  const localModels = models.filter(m => m.backend !== 'ollama');
  const ollamaModels = models.filter(m => m.backend === 'ollama');

  if (localModels.length) {
    const grp1 = document.createElement('optgroup');
    grp1.label = 'Local (GGUF)';
    for (const m of localModels) {
      const opt = document.createElement('option');
      opt.value = m.label;
      opt.textContent = m.label;
      grp1.appendChild(opt);
    }
    sel.appendChild(grp1);
  }
  if (ollamaModels.length) {
    const grp2 = document.createElement('optgroup');
    grp2.label = 'Ollama';
    for (const m of ollamaModels) {
      const opt = document.createElement('option');
      opt.value = m.label;
      opt.textContent = m.label;
      grp2.appendChild(opt);
    }
    sel.appendChild(grp2);
  }
  if (!localModels.length && !ollamaModels.length) {
    sel.innerHTML = '<option value="">No models found</option>';
    $('#btn-load-model').disabled = true;
    return;
  }

  // Select first and tell bridge
  await api.select_model(models[0].label);
  $('#btn-load-model').disabled = false;

  // Keep compare-model selectors in sync
  const a = $('#compare-model-a');
  const b = $('#compare-model-b');
  if (a && b) {
    a.innerHTML = '';
    b.innerHTML = '';
    for (const m of models) {
      const oa = document.createElement('option');
      oa.value = m.label;
      oa.textContent = m.label;
      a.appendChild(oa);
      const ob = document.createElement('option');
      ob.value = m.label;
      ob.textContent = m.label;
      b.appendChild(ob);
    }
    if (models.length > 1) b.selectedIndex = 1;
  }
}

$('#model-select').addEventListener('change', async function() {
  await api.select_model(this.value);
});

$('#btn-load-model').addEventListener('click', async function() {
  const status = await api.get_model_status();
  if (status.loaded) {
    // Unload first
    await api.unload_model();
    contextLimitTokens = 0;
    $('#loaded-model-label').textContent = 'No model loaded';
    $('#context-bar-container').style.display = 'none';
    updateContextBar();
    showToast('Model unloaded', 'info');
    this.textContent = 'Load';
  }
  const r = await api.load_model();
  if (r.error) {
    showToast(r.error, 'error');
    return;
  }
  this.disabled = true;
  $('#progress-bar-container').style.display = 'block';
});

// ═══ CHATS ════════════════════════════════════════════════════

async function refreshChats() {
  const chatIds = await api.get_chats();
  const list = $('#chat-list');
  list.innerHTML = '';
  for (const id of chatIds) {
    const encodedId = encodeURIComponent(id);
    const row = document.createElement('div');
    row.className = 'chat-row' + (id === currentChatId ? ' active' : '');
    row.dataset.chatId = id;
    row.innerHTML = `
      <span class="chat-name">${escapeHtml(id)}</span>
      <span class="chat-actions">
        <button class="btn-icon" onclick="event.stopPropagation(); exportChatById('${encodedId}')" title="Export">⤓</button>
        <button class="btn-icon" onclick="event.stopPropagation(); renameChatById('${encodedId}')" title="Rename">✎</button>
        <button class="btn-icon" onclick="event.stopPropagation(); deleteChatById('${encodedId}')" title="Delete">✕</button>
      </span>
    `;
    row.addEventListener('click', () => selectChat(id));
    list.appendChild(row);
  }
}

function decodeChatId(encodedId) {
  try {
    return decodeURIComponent(encodedId);
  } catch (_e) {
    return encodedId;
  }
}

async function exportChatById(encodedId) {
  const chatId = decodeChatId(encodedId);
  openExportModal({
    chatId,
    messageIndex: null,
    label: `Export chat: ${chatId}`,
  });
}

async function deleteChatById(encodedId) {
  return deleteChat(decodeChatId(encodedId));
}

async function renameChatById(encodedId) {
  return renameChat(decodeChatId(encodedId));
}

async function selectChat(chatId) {
  const r = await api.load_chat(chatId);
  if (r.error) { showToast(r.error, 'error'); return; }
  currentChatId = chatId;
  renderMessages(r.messages);
  highlightActiveChat();
  // Restore per-chat file attachment state
  syncUploadButton(r.attached_file || null);
  // Load system prompt
  const sp = await api.get_system_prompt();
  $('#system-prompt-input').value = sp || '';
  updateSendButton();
  updateContextBar();
}

function highlightActiveChat() {
  $$('.chat-row').forEach(r => {
    r.classList.toggle('active', r.dataset.chatId === currentChatId);
  });
}

function renderMessages(messages) {
  const container = $('#messages-container');
  container.innerHTML = '';
  for (let i = 0; i < messages.length; i++) {
    const m = messages[i];
    appendMessage(m.role, m.content, false, i);
  }
  updateContextBar();
  scrollToBottom();
}

/** Format a timestamp for display */
function formatTimestamp(date) {
  const h = date.getHours();
  const m = date.getMinutes();
  const ampm = h >= 12 ? 'PM' : 'AM';
  const h12 = h % 12 || 12;
  return `${h12}:${String(m).padStart(2, '0')} ${ampm}`;
}

function appendMessage(role, content, streaming = false, messageIndex = null) {
  const container = $('#messages-container');

  // Remove typing indicator if present
  const existingTyping = container.querySelector('.typing-indicator');
  if (existingTyping) existingTyping.remove();

  // Create wrapper with avatar
  const wrapper = document.createElement('div');
  wrapper.className = `message-wrapper ${role}`;

  const avatar = document.createElement('div');
  avatar.className = 'message-avatar';
  avatar.textContent = role === 'user' ? '🧑' : '🤖';

  const bubble = document.createElement('div');
  bubble.className = `message ${role}`;
  bubble.setAttribute('data-role', role);
  if (messageIndex !== null && messageIndex !== undefined) {
    bubble.setAttribute('data-msg-index', String(messageIndex));
  }
  if (streaming) {
    bubble.classList.add('streaming');
  }

  if (role === 'assistant') {
    bubble.innerHTML = renderMarkdown(content);
    if (!streaming) {
      const actions = document.createElement('div');
      actions.className = 'msg-actions';
      actions.innerHTML = assistantActionsHtml();
      bubble.appendChild(actions);
    }
  } else {
    bubble.innerHTML = renderMarkdown(content);
    if (!streaming) {
      bubble.classList.add('editable');
      bubble.title = 'Double-click to edit and regenerate';
      bubble.addEventListener('dblclick', () => editUserMessage(bubble));
    }
  }

  const meta = document.createElement('div');
  meta.className = 'message-meta';
  meta.textContent = formatTimestamp(new Date());

  const contentCol = document.createElement('div');
  contentCol.style.minWidth = '0';
  contentCol.style.flex = '1';
  contentCol.appendChild(bubble);
  contentCol.appendChild(meta);

  wrapper.appendChild(avatar);
  wrapper.appendChild(contentCol);
  container.appendChild(wrapper);
  scrollToBottom();
  return bubble;
}

document.addEventListener('click', (e) => {
  const target = e.target;
  if (!(target instanceof Element)) return;
  const link = target.closest('.rag-source-link');
  if (!link) return;
  e.preventDefault();
  const source = (link.getAttribute('data-source') || '').trim();
  if (!source) return;

  const input = $('#user-input');
  const current = (input.value || '').trim();
  const suffix = ` in ${source}`;
  input.value = current ? `${current}${suffix}` : `in ${source}`;
  autoResize(input);
  input.focus();
  setStatus(`Source selected: ${source}`);
  showToast(`Selected source: ${source}`, 'info', 2200);
});

function scrollToBottom() {
  const container = $('#messages-container');
  container.scrollTo({ top: container.scrollHeight, behavior: 'smooth' });
}

/** Show animated typing indicator (3 dots) */
function showTypingIndicator() {
  const container = $('#messages-container');
  // Don't duplicate
  if (container.querySelector('.typing-indicator')) return;
  const el = document.createElement('div');
  el.className = 'typing-indicator';
  el.innerHTML = `<div class="message-avatar">🤖</div>` +
    `<div class="typing-dots"><span></span><span></span><span></span></div>`;
  container.appendChild(el);
  scrollToBottom();
}

/** Remove typing indicator */
function removeTypingIndicator() {
  const el = $('#messages-container .typing-indicator');
  if (el) el.remove();
}

$('#btn-new-chat').addEventListener('click', async () => {
  const r = await api.new_chat();
  currentChatId = r.chat_id;
  await refreshChats();
  renderMessages([]);
  // New chat has no file attached
  syncUploadButton(null);
  updateSendButton();
});

async function deleteChat(chatId) {
  if (!await showConfirm(`Delete "${chatId}"?`, 'Delete Chat')) return;
  await api.delete_chat(chatId);
  if (currentChatId === chatId) {
    currentChatId = null;
    renderMessages([]);
  }
  await refreshChats();
  showToast('Chat deleted', 'info');
}

async function renameChat(chatId) {
  const newName = await showPrompt('Rename chat', chatId);
  if (!newName || newName.trim() === '' || newName.trim() === chatId) return;
  const r = await api.rename_chat(chatId, newName.trim());
  if (r.error) { showToast(r.error, 'error'); return; }
  if (currentChatId === chatId) currentChatId = newName.trim();
  await refreshChats();
}

// Chat search filter
$('#chat-search').addEventListener('input', function() {
  const q = this.value.toLowerCase();
  $$('.chat-row').forEach(row => {
    const name = row.querySelector('.chat-name').textContent.toLowerCase();
    row.style.display = name.includes(q) ? '' : 'none';
  });
});

// ═══ SENDING MESSAGES ═════════════════════════════════════════

const inputEl = $('#user-input');
const sendBtn = $('#btn-send');
const stopBtn = $('#btn-stop');

inputEl.addEventListener('input', function() {
  autoResize(this);
  const len = this.value.trim().length;
  updateSendButton();
  // Token estimate
  const tokens = Math.ceil(len / 4);
  $('#token-counter').textContent = `~${tokens} tokens`;
  updateContextBar();
});

inputEl.addEventListener('keydown', function(e) {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault();
    if (!sendBtn.disabled) sendMessage();
  }
});

sendBtn.addEventListener('click', sendMessage);

// ── @RAG mention autocomplete ─────────────────────────────────
(function setupRagMention() {
  const dropdown = document.createElement('div');
  dropdown.id = 'rag-mention-dropdown';
  dropdown.style.cssText = 'position:absolute;display:none;background:var(--bg-secondary);border:1px solid var(--border);border-radius:8px;box-shadow:0 4px 16px rgba(0,0,0,.35);z-index:2000;max-height:180px;overflow-y:auto;min-width:160px;max-width:min(420px, calc(100vw - 16px));overflow-x:hidden;';
  document.body.appendChild(dropdown);

  let activeInput = null;
  let mentionStart = -1;
  let cachedDbs = [];

  async function fetchDbs() {
    try { cachedDbs = (await api.get_rag_databases()) || []; } catch { cachedDbs = []; }
  }

  function hide() { dropdown.style.display = 'none'; activeInput = null; mentionStart = -1; }

  function show(input) {
    const text = input.value;
    const cursor = input.selectionStart;
    // Find the @ that triggered this
    let atPos = -1;
    for (let i = cursor - 1; i >= 0; i--) {
      if (text[i] === '@') { atPos = i; break; }
      if (/\s/.test(text[i]) && i < cursor - 1) break;
    }
    if (atPos < 0) { hide(); return; }

    const fragment = text.slice(atPos + 1, cursor).toLowerCase();
    const matches = cachedDbs.filter(db => db.name.toLowerCase().includes(fragment));
    if (matches.length === 0) { hide(); return; }

    activeInput = input;
    mentionStart = atPos;

    dropdown.innerHTML = '';
    matches.forEach(db => {
      const item = document.createElement('div');
      item.textContent = db.name;
      item.style.cssText = 'padding:7px 12px;cursor:pointer;font-size:13px;color:var(--text-primary);white-space:nowrap;overflow:hidden;text-overflow:ellipsis;';
      item.title = db.name;
      item.addEventListener('mouseenter', () => item.style.background = 'var(--accent)');
      item.addEventListener('mouseleave', () => item.style.background = '');
      item.addEventListener('mousedown', (e) => {
        e.preventDefault();
        insertMention(db.name);
      });
      dropdown.appendChild(item);
    });

    // Position near input and clamp to viewport so popup never goes off-screen.
    const rect = input.getBoundingClientRect();
    const margin = 8;
    const viewW = window.innerWidth || document.documentElement.clientWidth || 0;
    const viewH = window.innerHeight || document.documentElement.clientHeight || 0;

    // Show first so dimensions are measurable.
    dropdown.style.display = 'block';

    const maxByInput = Math.max(160, Math.min(420, Math.floor(rect.width)));
    const maxByViewport = Math.max(160, viewW - margin * 2);
    const targetWidth = Math.min(maxByInput, maxByViewport);
    dropdown.style.width = targetWidth + 'px';

    const box = dropdown.getBoundingClientRect();
    let left = rect.left;
    if (left + box.width > viewW - margin) {
      left = viewW - margin - box.width;
    }
    if (left < margin) left = margin;

    let top = rect.bottom + 4;
    if (top + box.height > viewH - margin) {
      const above = rect.top - box.height - 4;
      top = above >= margin ? above : Math.max(margin, viewH - margin - box.height);
    }

    dropdown.style.left = Math.round(left) + 'px';
    dropdown.style.top = Math.round(top) + 'px';
  }

  function insertMention(name) {
    if (!activeInput || mentionStart < 0) return;
    const before = activeInput.value.slice(0, mentionStart);
    const after = activeInput.value.slice(activeInput.selectionStart);
    const mention = name.includes(' ') ? `@"${name}" ` : `@${name} `;
    activeInput.value = before + mention + after;
    const newCursorPos = before.length + mention.length;
    activeInput.selectionStart = activeInput.selectionEnd = newCursorPos;
    activeInput.focus();
    hide();
  }

  function onInput(e) {
    const input = e.target;
    const text = input.value;
    const cursor = input.selectionStart;
    // Check if user just typed @ or is typing after @
    if (cursor > 0 && text[cursor - 1] === '@') {
      fetchDbs().then(() => show(input));
    } else if (dropdown.style.display !== 'none') {
      show(input);
    }
  }

  function onKeydown(e) {
    if (dropdown.style.display === 'none') return;
    const items = dropdown.children;
    if (!items.length) return;
    let sel = [...items].findIndex(el => el.style.background && el.style.background !== '');
    if (e.key === 'ArrowDown') {
      e.preventDefault();
      if (sel >= 0) items[sel].style.background = '';
      sel = (sel + 1) % items.length;
      items[sel].style.background = 'var(--accent)';
    } else if (e.key === 'ArrowUp') {
      e.preventDefault();
      if (sel >= 0) items[sel].style.background = '';
      sel = sel <= 0 ? items.length - 1 : sel - 1;
      items[sel].style.background = 'var(--accent)';
    } else if ((e.key === 'Enter' || e.key === 'Tab') && sel >= 0) {
      e.preventDefault();
      e.stopPropagation();
      insertMention(items[sel].textContent);
    } else if (e.key === 'Escape') {
      e.preventDefault();
      hide();
    }
  }

  function onBlur() { setTimeout(hide, 200); }

  // Attach to both chat and agent inputs
  const chatInput = $('#user-input');
  if (chatInput) {
    chatInput.addEventListener('input', onInput);
    chatInput.addEventListener('keydown', onKeydown);
    chatInput.addEventListener('blur', onBlur);
  }
  const agentInput = $('#agent-input');
  if (agentInput) {
    agentInput.addEventListener('input', onInput);
    agentInput.addEventListener('keydown', onKeydown);
    agentInput.addEventListener('blur', onBlur);
  }
})();
stopBtn.addEventListener('click', async () => {
  stopBtn.disabled = true;
  setStatus('Stopping generation...', true);
  await api.stop_generation();
});

function updateSendButton() {
  const hasText = inputEl.value.trim().length > 0;
  sendBtn.disabled = !hasText || !fullAccessGranted;
}

let streamingBubble = null;
let isGenerating = false;

async function sendMessage() {
  if (!fullAccessGranted) {
    const unlocked = await ensureActivationGate(true);
    if (!unlocked) {
      showToast('Activation required to continue.', 'warn');
      return;
    }
  }

  const text = inputEl.value.trim();
  if (!text) return;

  inputEl.value = '';
  autoResize(inputEl);
  updateSendButton();

  // If no chat, create one
  if (!currentChatId) {
    const r = await api.new_chat();
    currentChatId = r.chat_id;
    await refreshChats();
  }

  // Show user message immediately
  appendMessage('user', text);
  updateContextBar();

  // Show typing indicator, then create streaming bubble
  showTypingIndicator();
  streamingBubble = appendMessage('assistant', '', true);

  // Send to backend
  sendBtn.style.display = 'none';
  stopBtn.style.display = '';
  stopBtn.disabled = false;
  try {
    const r = await api.send_message(text);
    if (r?.error) {
      if (streamingBubble) {
        streamingBubble.classList.remove('streaming');
        streamingBubble.innerHTML = renderMarkdown(`⚠ ${r.error}`);
        streamingBubble = null;
      }
      isGenerating = false;
      sendBtn.style.display = '';
      stopBtn.style.display = 'none';
      stopBtn.disabled = false;
    }
  } catch (err) {
    if (streamingBubble) {
      streamingBubble.classList.remove('streaming');
      streamingBubble.innerHTML = renderMarkdown(`⚠ ${err?.message || 'Failed to send message'}`);
      streamingBubble = null;
    }
    isGenerating = false;
    sendBtn.style.display = '';
    stopBtn.style.display = 'none';
    stopBtn.disabled = false;
    showToast(`Send failed: ${err?.message || 'Unknown error'}`, 'error');
  }
}

// ═══ BACKEND EVENTS ═══════════════════════════════════════════

let pendingWebSources = null;       // sources from web_sources event

window.addEventListener('app_status', (e) => {
  const text = e?.detail?.text;
  if (text) setStatus(text, true);
});

// Web search events
window.addEventListener('web_search_start', (e) => {
  const query = e?.detail?.query || '';
  pendingWebSources = null;
  if (!streamingBubble) {
    streamingBubble = appendMessage('assistant', '', true);
  }
  streamingBubble.innerHTML =
    `<div class="web-search-anim">🔍 Searching the web for "<strong>${escapeHtml(query)}</strong>"` +
    `<span class="search-dots"><span></span><span></span><span></span></span></div>`;
  scrollToBottom();
});

window.addEventListener('web_sources', (e) => {
  const sources = e?.detail?.sources;
  if (!Array.isArray(sources) || !sources.length) return;
  pendingWebSources = sources;
  // Show cards immediately in streaming bubble while waiting for LLM
  if (streamingBubble) {
    streamingBubble.innerHTML = buildSourceCardsHtml(sources) +
      `<div class="web-search-anim">Generating answer` +
      `<span class="search-dots"><span></span><span></span><span></span></span></div>`;
    scrollToBottom();
  }
});

// Generation events
window.addEventListener('generation_start', () => {
  isGenerating = true;
  if (!streamingBubble) {
    showTypingIndicator();
    streamingBubble = appendMessage('assistant', '', true);
  }
});

window.addEventListener('generation_token', (e) => {
  removeTypingIndicator();
  if (streamingBubble) {
    const cardsHtml = pendingWebSources ? buildSourceCardsHtml(pendingWebSources) : '';
    streamingBubble.innerHTML = cardsHtml + renderMarkdownWithCitations(e.detail.text, pendingWebSources);
    scrollToBottom();
  }
});

window.addEventListener('generation_done', (e) => {
  removeTypingIndicator();
  isGenerating = false;
  const sources = e?.detail?.web_sources || pendingWebSources || null;
  if (streamingBubble) {
    streamingBubble.classList.remove('streaming');
    streamingBubble.setAttribute(
      'data-msg-index',
      String(document.querySelectorAll('#messages-container .message').length - 1)
    );
    const cardsHtml = sources && sources.length ? buildSourceCardsHtml(sources) : '';
    streamingBubble.innerHTML = cardsHtml + renderMarkdownWithCitations(e.detail.text, sources);
    // Add action buttons
    const actions = document.createElement('div');
    actions.className = 'msg-actions';
    actions.innerHTML = assistantActionsHtml();
    streamingBubble.appendChild(actions);
    streamingBubble = null;
  }
  pendingWebSources = null;
  sendBtn.style.display = '';
  stopBtn.style.display = 'none';
  stopBtn.disabled = false;
  attachedImageName = null;
  $('#btn-image').classList.remove('active');
  $('#btn-image').textContent = '▦';
  updateSendButton();
  updateContextBar();
  refreshChats(); // update chat list order
  scrollToBottom();
});

window.addEventListener('generation_error', (e) => {
  removeTypingIndicator();
  isGenerating = false;
  pendingWebSources = null;
  if (streamingBubble) {
    streamingBubble.classList.remove('streaming');
    streamingBubble.innerHTML = `<span style="color:var(--danger)">Error: ${escapeHtml(e.detail.error)}</span>`;
    streamingBubble = null;
  }
  sendBtn.style.display = '';
  stopBtn.style.display = 'none';
  stopBtn.disabled = false;
  updateContextBar();
  showToast(`Generation error: ${e.detail.error}`, 'error');
});

window.addEventListener('context_usage', (e) => {
  const detail = e.detail || {};
  actualContext.used = Number(detail.prompt_tokens || 0);
  actualContext.total = Number(detail.effective_window || 0);
  updateContextBar();
});

window.addEventListener('generation_stopped', (e) => {
  isGenerating = false;
  if (streamingBubble) {
    streamingBubble.classList.remove('streaming');
    const partial = (e?.detail?.text || '').trim();
    const finalText = partial ? `${partial}\n\n⏹ Stopped` : '⏹ Stopped';
    streamingBubble.innerHTML = renderMarkdown(finalText);
    streamingBubble = null;
  }
  pendingWebSources = null;
  sendBtn.style.display = '';
  stopBtn.style.display = 'none';
  stopBtn.disabled = false;
  setStatus('Generation stopped');
  updateContextBar();
});

window.addEventListener('chat_branched', async (e) => {
  const chatId = e?.detail?.chat_id;
  if (!chatId) return;
  await refreshChats();
  await selectChat(chatId);
  showToast(`Branched to ${chatId}`, 'info');
});

window.addEventListener('tts_error', (e) => {
  showToast(`TTS error: ${e.detail.error}`, 'error');
});

// Message added from backend (e.g. web search without model)
window.addEventListener('message_added', (e) => {
  const role = e?.detail?.role;
  const content = e?.detail?.content;
  if (!content || role !== 'assistant') return;

  if (streamingBubble) {
    streamingBubble.classList.remove('streaming');
    streamingBubble.innerHTML = renderMarkdown(content);
    const actions = document.createElement('div');
    actions.className = 'msg-actions';
    actions.innerHTML = assistantActionsHtml();
    streamingBubble.appendChild(actions);
    streamingBubble = null;
  } else {
    appendMessage('assistant', content);
  }

  isGenerating = false;
  pendingWebSources = null;
  sendBtn.style.display = '';
  stopBtn.style.display = 'none';
  stopBtn.disabled = false;
  updateContextBar();
  scrollToBottom();
});

// Model events
window.addEventListener('model_status', (e) => {
  setStatus(e.detail.text, true);
  const pct = e.detail.progress || 0;
  $('#progress-bar').style.width = pct + '%';
  $('#progress-bar-container').style.display = 'block';
});

window.addEventListener('model_loaded', (e) => {
  const d = e.detail;
  const backend = d.backend || 'llama_cpp';
  const backendTag = backend === 'ollama' ? ' [Ollama]' : '';
  contextLimitTokens = d.n_ctx || contextLimitTokens;
  actualContext.used = 0;
  actualContext.total = 0;
  $('#loaded-model-label').textContent = `Loaded: ${d.name}${backendTag}`;
  $('#progress-bar').style.width = '100%';
  setTimeout(() => {
    $('#progress-bar-container').style.display = 'none';
    $('#progress-bar').style.width = '0%';
  }, 1500);
  setStatus(`Loaded in ${d.load_time}s | ctx: ${d.n_ctx} | ${backend}`);
  $('#btn-load-model').textContent = 'Unload';
  $('#btn-load-model').disabled = false;
  $('#context-bar-container').style.display = '';
  updateContextBar();
  showToast(`Model loaded in ${d.load_time}s (${backend})`, 'info');
  updateSendButton();
});

window.addEventListener('model_error', (e) => {
  setStatus(`Error: ${e.detail.error}`);
  $('#progress-bar-container').style.display = 'none';
  $('#btn-load-model').disabled = false;
  $('#btn-load-model').textContent = 'Load';
  showToast(e.detail.error, 'error');
});

// System monitor
window.addEventListener('system_stats', (e) => {
  $('#system-monitor').textContent = e.detail.text;
});

// Download events
window.addEventListener('download_status', (e) => {
  $('#hf-dl-status').textContent = e.detail.text;
  $('#hf-dl-progress').style.width = e.detail.progress + '%';
  $('#hf-download-area').style.display = '';
});

window.addEventListener('download_done', async (e) => {
  showToast(`Downloaded: ${e.detail.filename}`, 'info');
  await refreshModels();
  $('#hf-download-area').style.display = 'none';
});

window.addEventListener('download_error', (e) => {
  showToast(`Download failed: ${e.detail.error}`, 'error');
  $('#hf-download-area').style.display = 'none';
});

// ═══ MESSAGE ACTIONS ══════════════════════════════════════════

function copyMessage(btn) {
  const msgEl = btn.closest('.message');
  const text = msgEl.innerText
    .replace(/Copy|Regenerate|Branch|Speak|Export/g, '')
    .trim();
  navigator.clipboard.writeText(text).then(() => showToast('Copied!', 'info'));
}

async function regenerateMessage() {
  // Prevent multiple simultaneous regenerate requests
  if (isGenerating) {
    showToast('Generation already in progress', 'warning');
    return;
  }
  
  // Remove last assistant message and re-send last user message
  const container = $('#messages-container');
  const msgs = container.querySelectorAll('.message');
  if (msgs.length < 2) return;

  // Find last user message
  let lastUserText = '';
  for (let i = msgs.length - 1; i >= 0; i--) {
    if (msgs[i].getAttribute('data-role') === 'user') {
      lastUserText = msgs[i].innerText.trim();
      break;
    }
  }
  if (!lastUserText) return;

  // Remove last assistant
  const last = msgs[msgs.length - 1];
  if (last.getAttribute('data-role') === 'assistant') {
    last.remove();
  }

  // Re-send
  isGenerating = true;
  streamingBubble = appendMessage('assistant', '...', true);
  sendBtn.style.display = 'none';
  stopBtn.style.display = '';
  stopBtn.disabled = false;
  await api.send_message(lastUserText);
}

async function branchConversation() {
  const r = await api.branch_chat();
  if (r?.error) {
    showToast(r.error, 'error');
    return;
  }
  await refreshChats();
  await selectChat(r.chat_id);
  showToast(`Branched to ${r.chat_id}`, 'info');
}

async function editUserMessage(messageEl) {
  const idxRaw = messageEl.getAttribute('data-msg-index');
  if (idxRaw == null) return;
  const idx = parseInt(idxRaw, 10);
  if (Number.isNaN(idx)) return;

  const oldText = messageEl.innerText.trim();
  const edited = await showPrompt('Edit message', oldText);
  if (edited == null) return;
  const newText = edited.trim();
  if (!newText || newText === oldText) return;

  const r = await api.edit_user_message(idx, newText);
  if (r?.error) {
    showToast(r.error, 'error');
    return;
  }

  renderMessages(r.messages || []);

  if (r.regenerating) {
    streamingBubble = appendMessage('assistant', '..... loading', true);
    sendBtn.style.display = 'none';
    stopBtn.style.display = '';
  } else {
    sendBtn.style.display = '';
    stopBtn.style.display = 'none';
  }

  showToast('Message updated and regenerated', 'info');
}

async function speakMessage(btn) {
  const msgEl = btn.closest('.message');
  if (!msgEl) return;
  if (isSpeaking) {
    const stopResp = await api.stop_speaking();
    if (stopResp?.error) {
      showToast(stopResp.error, 'error');
      return;
    }
    isSpeaking = false;
    $$('.msg-actions button').forEach((b) => {
      if (b.textContent?.trim() === 'Stop') b.textContent = 'Speak';
    });
    showToast('Speech stopped', 'info', 1200);
    return;
  }

  const text = msgEl.innerText
    .replace(/Copy|Regenerate|Branch|Speak|Export/g, '')
    .trim();
  if (!text) return;
  const r = await api.speak_text(text);
  if (r?.error) {
    showToast(r.error, 'error');
  } else {
    isSpeaking = true;
    btn.textContent = 'Stop';
    showToast('Speaking...', 'info', 1500);
  }
}

window.addEventListener('tts_done', () => {
  isSpeaking = false;
  $$('.msg-actions button').forEach((b) => {
    if (b.textContent?.trim() === 'Stop') b.textContent = 'Speak';
  });
});

window.addEventListener('tts_stopped', () => {
  isSpeaking = false;
  $$('.msg-actions button').forEach((b) => {
    if (b.textContent?.trim() === 'Stop') b.textContent = 'Speak';
  });
});

async function exportCurrentChat() {
  if (!currentChatId) {
    showToast('No chat selected', 'warning');
    return;
  }
  openExportModal({
    chatId: currentChatId,
    messageIndex: null,
    label: `Export chat: ${currentChatId}`,
  });
}

$('#btn-confirm-export').addEventListener('click', async () => {
  if (!pendingExportTarget?.chatId) {
    showToast('Nothing selected to export', 'warning');
    return;
  }
  const format = $('#export-format-select').value;
  console.log('[EXPORT] Starting export');
  
  // Close modal IMMEDIATELY
  closeModal('modal-export');
  
  // Small delay to ensure modal closes
  await new Promise(r => setTimeout(r, 100));
  
  try {
    console.log('[EXPORT] Calling api.export_chat_dialog...');
    const response = await api.export_chat_dialog(
      pendingExportTarget.chatId,
      format,
      pendingExportTarget.messageIndex
    );
    
    console.log('[EXPORT] Got response:', response);
    console.log('[EXPORT] Response type:', typeof response);
    console.log('[EXPORT] Response.ok:', response?.ok);
    console.log('[EXPORT] Response.path:', response?.path);
    console.log('[EXPORT] Response.error:', response?.error);
    
    // Show visual indicator
    const indicator = document.createElement('div');
    indicator.id = 'export-indicator';
    indicator.style.position = 'fixed';
    indicator.style.top = '20px';
    indicator.style.right = '20px';
    indicator.style.background = '#333';
    indicator.style.color = '#fff';
    indicator.style.padding = '10px 20px';
    indicator.style.zIndex = '9999';
    indicator.textContent = `Export response: ${JSON.stringify(response)}`;
    document.body.appendChild(indicator);
    
    if (response?.ok && response?.path) {
      showToast(`✓ Exported to ${response.path}`, 'success');
      setTimeout(() => indicator.remove(), 3000);
      return;
    }
    
    if (response?.error) {
      showToast(`✗ ${response.error}`, 'error');
      setTimeout(() => indicator.remove(), 3000);
      return;
    }
    
    if (response?.selected === false) {
      showToast('Export cancelled', 'info');
      setTimeout(() => indicator.remove(), 3000);
      return;
    }
    
    showToast('Export completed', 'success');
    setTimeout(() => indicator.remove(), 3000);
  } catch (err) {
    console.error('[EXPORT] Caught error:', err);
    showToast(`Export error: ${err.message}`, 'error');
  }
});

// ═══ WEB SEARCH TOGGLE ═══════════════════════════════════════

$('#btn-web-search').addEventListener('click', async () => {
  const r = await api.toggle_web_search();
  const btn = $('#btn-web-search');
  btn.classList.toggle('active', r.enabled);
  btn.title = r.enabled ? 'Web search ON' : 'Web search OFF';
  // Keep agent button in sync
  agentWebSearch = !!r.enabled;
  const agentBtn = $('#btn-agent-web');
  if (agentBtn) {
    agentBtn.classList.toggle('active', agentWebSearch);
    agentBtn.style.background = agentWebSearch ? 'var(--accent)' : '';
    agentBtn.style.color = agentWebSearch ? 'white' : '';
  }
  showToast(r.enabled ? 'Web search enabled' : 'Web search disabled', 'info');
  updateSendButton();
});

// ═══ RAG MANAGEMENT ═══════════════════════════════════════════

async function refreshRag(retryCount = 0) {
  let dbs;
  try {
    dbs = await api.get_rag_databases();
  } catch (e) {
    if (retryCount < 3) {
      setTimeout(() => refreshRag(retryCount + 1), 1000 * (retryCount + 1));
    }
    return;
  }
  const list = $('#rag-list');
  list.innerHTML = '';
  for (const db of dbs) {
    const row = document.createElement('div');
    row.className = 'rag-row' + (db.selected ? ' selected' : '');
    row.innerHTML = `
      <span class="rag-name">${escapeHtml(db.name)}</span>
      <span class="rag-count">${db.chunks} chunks</span>
      <button class="btn-icon" onclick="selectRag('${escapeHtml(db.name)}')" title="Select">✓</button>
      <button class="btn-icon" onclick="reindexRag('${escapeHtml(db.name)}')" title="Reindex">↺</button>
      <button class="btn-icon" onclick="deleteRag('${escapeHtml(db.name)}')" title="Delete">✕</button>
    `;
    list.appendChild(row);
  }
}

async function selectRag(name) {
  const r = await api.select_rag(name);
  showToast(r.selected ? `RAG: ${r.selected}` : 'RAG deselected', 'info');
  await refreshRag();
}

async function deleteRag(name) {
  if (!await showConfirm(`Delete knowledge base "${name}"?`, 'Delete Knowledge Base')) return;
  const r = await api.delete_rag(name);
  if (r.error) { showToast(r.error, 'error'); return; }
  showToast('Knowledge base deleted', 'info');
  await refreshRag();
}

async function reindexRag(name) {
  // Show progress bar for reindex
  showRagProgress(name);
  const r = await api.reindex_rag(name);
  if (r?.error) {
    showToast(r.error, 'error');
    hideRagProgress();
    return;
  }
  showToast(`Reindexed ${name}`, 'info');
  await refreshRag();
}

// ── RAG Progress Bar ──
function showRagProgress(name) {
  const bar = $('#rag-progress-bar');
  const text = $('#rag-progress-text');
  const fill = $('#rag-progress-fill');
  bar.classList.remove('hidden');
  fill.classList.remove('error');
  fill.style.width = '0%';
  text.textContent = `Processing "${name}"...`;
}
function hideRagProgress() {
  const bar = $('#rag-progress-bar');
  setTimeout(() => bar.classList.add('hidden'), 1500);
}

window.addEventListener('rag_progress', (e) => {
  const d = e.detail;
  const bar = $('#rag-progress-bar');
  const text = $('#rag-progress-text');
  const fill = $('#rag-progress-fill');

  bar.classList.remove('hidden');

  if (d.error) {
    fill.classList.add('error');
    text.textContent = `Failed: ${d.name}`;
    hideRagProgress();
    refreshRag();
    return;
  }
  if (d.done) {
    fill.style.width = '100%';
    text.textContent = `✓ "${d.name}" complete`;
    hideRagProgress();
    refreshRag();
    return;
  }

  fill.classList.remove('error');
  const pct = d.percent || 0;
  fill.style.width = pct + '%';
  const fileLabel = d.file ? d.file : '';
  const count = d.total_files ? ` (${d.file_index + 1}/${d.total_files})` : '';
  text.textContent = `${fileLabel}${count}`;
});

function updateRagSourceUi() {
  const sel = $('#rag-source-type');
  const label = $('#rag-path-label');
  const pathInput = $('#rag-path');
  if (!sel || !label || !pathInput) return;

  if (sel.value === 'url') {
    label.textContent = 'URL';
    pathInput.placeholder = 'https://example.com/article';
  } else {
    label.textContent = 'Folder path';
    pathInput.placeholder = 'C:\\Documents\\notes';
  }
}

$('#btn-add-rag').addEventListener('click', () => {
  updateRagSourceUi();
  openModal('modal-rag-create');
});

$('#rag-source-type')?.addEventListener('change', updateRagSourceUi);

// RAG sliders
$('#rag-chunk-size').addEventListener('input', function() {
  $('#rag-chunk-val').textContent = this.value;
});
$('#rag-overlap').addEventListener('input', function() {
  $('#rag-overlap-val').textContent = this.value;
});

$('#btn-create-rag').addEventListener('click', async () => {
  const createBtn = $('#btn-create-rag');
  const statusEl = $('#rag-create-status');
  const sourceType = $('#rag-source-type')?.value || 'folder';
  const name = $('#rag-name').value.trim();
  const path = $('#rag-path').value.trim();
  statusEl.textContent = '';
  statusEl.style.color = 'var(--text-muted)';

  if (!name || !path) {
    statusEl.textContent = 'Enter both name and path/URL.';
    statusEl.style.color = 'var(--danger)';
    showToast('Name and path required', 'warn');
    return;
  }
  // Validate name
  if (/[\\/:*?"<>|]/.test(name)) {
    statusEl.textContent = 'Invalid characters in database name.';
    statusEl.style.color = 'var(--danger)';
    showToast('Invalid characters in name', 'warn');
    return;
  }

  createBtn.disabled = true;
  const oldText = createBtn.textContent;
  createBtn.textContent = 'Creating...';
  statusEl.textContent = 'Creating knowledge base, please wait...';
  showRagProgress(name);

  const chunk = parseInt($('#rag-chunk-size').value);
  const overlap = parseInt($('#rag-overlap').value);
  const pathLooksUrl = /^https?:\/\//i.test(path);

  if (sourceType === 'url' && !pathLooksUrl) {
    statusEl.textContent = 'URL mode requires a valid http/https URL.';
    statusEl.style.color = 'var(--danger)';
    showToast('Enter a valid URL (http/https)', 'warn');
    createBtn.disabled = false;
    createBtn.textContent = oldText;
    hideRagProgress();
    return;
  }

  const isUrl = sourceType === 'url' || (sourceType !== 'folder' && pathLooksUrl);
  try {
    const r = isUrl
      ? await api.create_rag_from_url(path, name, chunk, overlap)
      : await api.create_rag_from_folder(path, name, chunk, overlap);

    if (r?.error) {
      statusEl.textContent = r.error;
      statusEl.style.color = 'var(--danger)';
      showToast(r.error, 'error', 6000);
      return;
    }

    statusEl.textContent = `Created \"${name}\" with ${r.chunks} chunks.`;
    statusEl.style.color = 'var(--success)';
    showToast(`Created "${name}" with ${r.chunks} chunks`, 'info');
    await refreshRag();
    await selectRag(name);

    // Clear fields and close only on success
    $('#rag-name').value = '';
    $('#rag-path').value = '';
    $('#rag-source-type').value = 'folder';
    updateRagSourceUi();
    closeModal('modal-rag-create');
  } finally {
    createBtn.disabled = false;
    createBtn.textContent = oldText;
  }
});

// ═══ SETTINGS ═════════════════════════════════════════════════

$('#btn-settings').addEventListener('click', async () => {
  const info = await api.get_app_info();
  const settings = await api.get_app_settings();

  // Populate
  $('#setting-theme').value = info.theme.toLowerCase();
  $('#s-temperature').value = Math.round((settings.temperature || 0.25) * 100);
  $('#s-temp-val').textContent = (settings.temperature || 0.25).toFixed(2);
  $('#s-top-p').value = Math.round((settings.top_p || 0.9) * 100);
  $('#s-topp-val').textContent = (settings.top_p || 0.9).toFixed(2);
  $('#s-repeat-penalty').value = Math.round((settings.repeat_penalty || 1.1) * 100);
  $('#s-reppen-val').textContent = (settings.repeat_penalty || 1.1).toFixed(2);
  applyReplyTokenLimitUi(settings.max_response_tokens);
  $('#s-brave-api-key').value = settings.brave_api_key || '';
  await populateVoiceOptions(settings.tts_voice_id || '');
  await populatePiperCatalog();

  const gpuType = info.gpu.type || 'CPU';
  const vram = info.gpu.vram || 0;
  $('#s-gpu-info').textContent = `${gpuType} (${vram}GB VRAM) — ${info.config.mode}`;
  $('#s-system-info').textContent = `RAM: ${info.system_ram}GB`;

  openModal('modal-settings');
});

$('#btn-download-piper-voice').addEventListener('click', async () => {
  const select = $('#s-piper-voice-catalog');
  const status = $('#s-piper-download-status');
  const btn = $('#btn-download-piper-voice');
  const voiceId = (select?.value || '').trim();
  if (!voiceId) {
    showToast('Select a Piper voice first', 'warning');
    return;
  }

  btn.disabled = true;
  const oldText = btn.textContent;
  btn.textContent = 'Downloading...';
  if (status) status.textContent = 'Downloading...';
  try {
    const r = await api.download_free_piper_voice(voiceId);
    if (r?.error) {
      if (status) status.textContent = r.error;
      showToast(r.error, 'error', 5000);
      return;
    }
    if (status) status.textContent = `Saved to ${r.folder}`;
    showToast(`Downloaded Piper voice to ${r.folder}`, 'success', 5000);
  } finally {
    btn.disabled = false;
    btn.textContent = oldText;
  }
});

// Settings slider live updates
$('#s-temperature').addEventListener('input', function() {
  $('#s-temp-val').textContent = (this.value / 100).toFixed(2);
});
$('#s-top-p').addEventListener('input', function() {
  $('#s-topp-val').textContent = (this.value / 100).toFixed(2);
});
$('#s-repeat-penalty').addEventListener('input', function() {
  $('#s-reppen-val').textContent = (this.value / 100).toFixed(2);
});

$('#btn-save-settings').addEventListener('click', async () => {
  const theme = $('#setting-theme').value;
  document.documentElement.setAttribute('data-theme', theme);
  switchHljsTheme(theme);
  await api.set_theme(theme.charAt(0).toUpperCase() + theme.slice(1));

  const maxResponseTokens = normalizeReplyTokenLimit($('#s-max-tokens').value);
  await api.save_app_settings({
    theme: theme.charAt(0).toUpperCase() + theme.slice(1),
    temperature: parseFloat(($('#s-temperature').value / 100).toFixed(3)),
    top_p: parseFloat(($('#s-top-p').value / 100).toFixed(3)),
    repeat_penalty: parseFloat(($('#s-repeat-penalty').value / 100).toFixed(3)),
    max_response_tokens: maxResponseTokens,
    brave_api_key: ($('#s-brave-api-key').value || '').trim(),
    tts_voice_id: ($('#s-voice')?.value || '').trim(),
  });

  applyReplyTokenLimitUi(maxResponseTokens);

  closeModal('modal-settings');
  showToast('Settings saved', 'info');
});

$('#chat-max-tokens')?.addEventListener('change', async function() {
  const maxResponseTokens = normalizeReplyTokenLimit(this.value);
  await api.save_app_settings({ max_response_tokens: maxResponseTokens });
  applyReplyTokenLimitUi(maxResponseTokens);
  showToast(
    maxResponseTokens > 0
      ? `Reply length limit set to ${maxResponseTokens} tokens`
      : 'Reply length set to Auto (context-based)',
    'info'
  );
});

window.addEventListener('export_ready', (e) => {
  const path = e?.detail?.path;
  const format = (e?.detail?.format || '').toUpperCase();
  if (!path) return;
  showToast(`Saved ${format || 'file'} to ${path}`, 'info', 6000);
});

// ═══ SYSTEM PROMPT ════════════════════════════════════════════

$('#btn-system-prompt').addEventListener('click', async () => {
  const sp = await api.get_system_prompt();
  $('#system-prompt-input').value = sp || '';
  openModal('modal-system-prompt');
});

$('#btn-save-system-prompt').addEventListener('click', async () => {
  const text = $('#system-prompt-input').value;
  await api.set_system_prompt(text);
  closeModal('modal-system-prompt');
  showToast('System prompt saved', 'info');
});

// ═══ PER-MODEL SETTINGS ══════════════════════════════════════

$('#btn-model-settings').addEventListener('click', async () => {
  const status = await api.get_model_status();
  const sel = $('#model-select').value;
  $('#model-settings-name').textContent = sel || 'No model selected';

  const s = await api.get_per_model_settings();
  $('#ms-temperature').value = Math.round((s.temperature || 0.25) * 100);
  $('#ms-temp-val').textContent = (s.temperature || 0.25).toFixed(2);
  if (s.n_ctx) { $('#ms-n-ctx').value = String(s.n_ctx); }
  else { $('#ms-n-ctx').value = 'default'; }
  $('#ms-threads').value = s.n_threads || 4;
  $('#ms-threads-val').textContent = s.n_threads || 4;

  openModal('modal-model-settings');
});

$('#ms-temperature').addEventListener('input', function() {
  $('#ms-temp-val').textContent = (this.value / 100).toFixed(2);
});
$('#ms-threads').addEventListener('input', function() {
  $('#ms-threads-val').textContent = this.value;
});

$('#btn-save-model-settings').addEventListener('click', async () => {
  const cfg = {
    temperature: parseFloat(($('#ms-temperature').value / 100).toFixed(3)),
    n_threads: parseInt($('#ms-threads').value),
  };
  const nCtx = $('#ms-n-ctx').value;
  if (nCtx !== 'default') cfg.n_ctx = parseInt(nCtx);
  await api.save_per_model_settings(cfg);
  closeModal('modal-model-settings');
  showToast('Model settings saved', 'info');
});

$('#btn-reset-model-settings').addEventListener('click', async () => {
  await api.save_per_model_settings({});
  closeModal('modal-model-settings');
  showToast('Model settings reset', 'info');
});

// ═══ HF DOWNLOADER ════════════════════════════════════════════

$('#btn-hf-download').addEventListener('click', async () => {
  const info = await api.get_app_info();
  const gpu = info.gpu;
  const gpuLabel = gpu.type === 'CPU' ? 'CPU-only' :
    `${gpu.type} (${gpu.vram}GB VRAM)`;
  $('#hf-system-info').textContent =
    `System: ${info.system_ram}GB RAM | ${gpuLabel} | ✅ fits  ⚠ may be slow`;
  $('#hf-results').innerHTML = '';
  openModal('modal-hf');
});

// ═══ EXPORT / COMPARE / PLUGINS ═════════════════════════════

async function openCompareModal() {
  const models = await api.get_models();
  if (!models || models.length < 2) {
    showToast('Need at least 2 models for comparison', 'warn');
    return;
  }
  await refreshModels();
  $('#compare-prompt').value = '';
  $('#compare-res-a').value = '';
  $('#compare-res-b').value = '';
  closeModal('modal-settings');
  openModal('modal-compare');
}

$('#btn-run-compare').addEventListener('click', async () => {
  const prompt = $('#compare-prompt').value.trim();
  if (!prompt) {
    showToast('Enter a prompt first', 'warn');
    return;
  }
  const a = $('#compare-model-a').value;
  const b = $('#compare-model-b').value;
  if (!a || !b) {
    showToast('Select both models', 'warn');
    return;
  }
  $('#compare-res-a').value = 'Running...';
  $('#compare-res-b').value = 'Running...';
  const r = await api.compare_models(a, b, prompt);
  if (r?.error) {
    $('#compare-res-a').value = `Error: ${r.error}`;
    $('#compare-res-b').value = `Error: ${r.error}`;
    return;
  }
  $('#compare-res-a').value = r.response_a || '(no response)';
  $('#compare-res-b').value = r.response_b || '(no response)';
});

async function refreshPlugins() {
  const plugins = await api.list_plugins();
  const list = $('#plugins-list');
  list.innerHTML = '';
  if (!plugins || plugins.length === 0) {
    list.innerHTML = '<div class="muted small">No plugins found in plugins/ folder.</div>';
    return;
  }
  for (const p of plugins) {
    const row = document.createElement('div');
    row.className = 'hf-card';
    row.innerHTML = `
      <div class="hf-info">
        <div class="hf-name">${p.loaded ? '✅' : '⚠'} ${escapeHtml(p.name)}</div>
        <div class="hf-file">${escapeHtml(String(p.info))}</div>
      </div>
    `;

    const actions = document.createElement('div');
    actions.style.display = 'flex';
    actions.style.gap = '6px';

    const editBtn = document.createElement('button');
    editBtn.className = 'btn ghost btn-sm';
    editBtn.textContent = 'Edit';
    editBtn.addEventListener('click', async () => {
      await openPluginEditModal(p.name);
    });

    const delBtn = document.createElement('button');
    delBtn.className = 'btn danger btn-sm';
    delBtn.textContent = 'Delete';
    delBtn.addEventListener('click', async () => {
      const ok = confirm(`Delete plugin "${p.name}"? This cannot be undone.`);
      if (!ok) return;
      const r = await api.delete_plugin(p.name);
      if (r?.error) {
        showToast(`Delete failed: ${r.error}`, 'error');
        return;
      }
      showToast(`Deleted plugin "${p.name}"`, 'success');
      await refreshPlugins();
    });

    actions.appendChild(editBtn);
    actions.appendChild(delBtn);
    row.appendChild(actions);
    list.appendChild(row);
  }
}

async function openPluginsModal() {
  await refreshPlugins();
  closeModal('modal-settings');
  openModal('modal-plugins');
}

$('#btn-settings-compare').addEventListener('click', openCompareModal);
$('#btn-settings-plugins').addEventListener('click', openPluginsModal);

$('#btn-reload-plugins').addEventListener('click', async () => {
  const r = await api.reload_plugins();
  if (r?.error) {
    showToast(r.error, 'error');
    return;
  }
  await refreshPlugins();
  showToast('Plugins reloaded', 'info');
});

// ═══ CREATE PLUGIN WITH AI ══════════════════════════════════

let _pluginGenActive = false;
let _pluginEditName = null;

function _setPluginAiCreateMode() {
  _pluginEditName = null;
  const title = document.getElementById('plugin-ai-title');
  const saveBtn = document.getElementById('btn-plugin-ai-save');
  const nameEl = document.getElementById('plugin-ai-name');
  if (title) title.textContent = '✨ Create Plugin with AI';
  if (saveBtn) saveBtn.textContent = 'Save & Load';
  if (nameEl) nameEl.disabled = false;
}

function _setPluginAiEditMode(name) {
  _pluginEditName = String(name || '').trim();
  const title = document.getElementById('plugin-ai-title');
  const saveBtn = document.getElementById('btn-plugin-ai-save');
  const nameEl = document.getElementById('plugin-ai-name');
  if (title) title.textContent = `✏ Edit Plugin: ${_pluginEditName}`;
  if (saveBtn) saveBtn.textContent = 'Save Changes';
  if (nameEl) nameEl.disabled = true;
}

function openPluginAiModal() {
  try {
    _setPluginAiCreateMode();
    // Reset form state
    const nameEl = document.getElementById('plugin-ai-name');
    const descEl = document.getElementById('plugin-ai-desc');
    const codeEl = document.getElementById('plugin-ai-code');
    const genStatus = document.getElementById('plugin-ai-gen-status');
    const testStatus = document.getElementById('plugin-ai-test-status');
    const testOut = document.getElementById('plugin-ai-test-output');
    if (nameEl) nameEl.value = '';
    if (descEl) descEl.value = '';
    if (codeEl) codeEl.value = '';
    if (genStatus) genStatus.textContent = '';
    if (testStatus) testStatus.textContent = '';
    if (testOut) { testOut.classList.add('hidden'); testOut.textContent = ''; }
    _pluginGenActive = false;
    closeModal('modal-plugins');
    openModal('modal-plugin-ai');
  } catch(err) {
    showToast('Could not open plugin editor: ' + err.message, 'error');
  }
}
window.openPluginAiModal = openPluginAiModal;

async function openPluginEditModal(pluginName) {
  const r = await api.get_plugin_code(pluginName);
  if (r?.error) {
    showToast(`Could not load plugin: ${r.error}`, 'error');
    return;
  }
  _setPluginAiEditMode(r.name || pluginName);

  const nameEl = document.getElementById('plugin-ai-name');
  const descEl = document.getElementById('plugin-ai-desc');
  const codeEl = document.getElementById('plugin-ai-code');
  const genStatus = document.getElementById('plugin-ai-gen-status');
  const testStatus = document.getElementById('plugin-ai-test-status');
  const testOut = document.getElementById('plugin-ai-test-output');
  if (nameEl) nameEl.value = r.name || pluginName;
  if (descEl) descEl.value = '';
  if (codeEl) codeEl.value = r.code || '';
  if (genStatus) genStatus.textContent = 'Editing existing plugin source';
  if (testStatus) testStatus.textContent = '';
  if (testOut) { testOut.classList.add('hidden'); testOut.textContent = ''; }
  _pluginGenActive = false;

  closeModal('modal-plugins');
  openModal('modal-plugin-ai');
}
window.openPluginEditModal = openPluginEditModal;

$('#btn-create-plugin-ai').addEventListener('click', openPluginAiModal);

// Stream plugin generation
window.addEventListener('plugin_gen_token', (e) => {
  if (!_pluginGenActive) return;
  $('#plugin-ai-code').value = e.detail.text || '';
});

window.addEventListener('plugin_gen_done', (e) => {
  _pluginGenActive = false;
  $('#plugin-ai-code').value = e.detail.code || '';
  $('#plugin-ai-gen-status').textContent = '✅ Generation complete. Review, test, then save.';
  $('#btn-plugin-ai-generate').disabled = false;
  $('#btn-plugin-ai-stop').disabled = true;
});

window.addEventListener('plugin_gen_error', (e) => {
  _pluginGenActive = false;
  $('#plugin-ai-gen-status').textContent = `❌ ${e.detail.error}`;
  $('#btn-plugin-ai-generate').disabled = false;
  $('#btn-plugin-ai-stop').disabled = true;
  showToast(e.detail.error, 'error');
});

$('#btn-plugin-ai-generate').addEventListener('click', async () => {
  const name = $('#plugin-ai-name').value.trim();
  const desc = $('#plugin-ai-desc').value.trim();
  if (!name) { showToast('Enter a plugin name first', 'warn'); return; }
  if (!desc) { showToast('Describe what the plugin should do', 'warn'); return; }

  _pluginGenActive = true;
  $('#plugin-ai-code').value = '';
  $('#plugin-ai-gen-status').textContent = '⏳ Generating…';
  $('#plugin-ai-test-output').classList.add('hidden');
  $('#btn-plugin-ai-generate').disabled = true;
  $('#btn-plugin-ai-stop').disabled = false;

  const r = await api.generate_plugin_with_ai(name, desc);
  if (r?.error) {
    _pluginGenActive = false;
    $('#plugin-ai-gen-status').textContent = `❌ ${r.error}`;
    $('#btn-plugin-ai-generate').disabled = false;
    $('#btn-plugin-ai-stop').disabled = true;
  }
});

$('#btn-plugin-ai-stop').addEventListener('click', () => {
  if (!_pluginGenActive) return;
  api.stop_generation && api.stop_generation();
  _pluginGenActive = false;
  $('#plugin-ai-gen-status').textContent = 'Stopped.';
  $('#btn-plugin-ai-generate').disabled = false;
  $('#btn-plugin-ai-stop').disabled = true;
});

$('#btn-plugin-ai-test').addEventListener('click', async () => {
  const name = $('#plugin-ai-name').value.trim() || 'test_plugin';
  const code = $('#plugin-ai-code').value.trim();
  if (!code) { showToast('No code to test', 'warn'); return; }

  $('#plugin-ai-test-status').textContent = '⏳ Testing…';
  $('#plugin-ai-test-output').classList.add('hidden');

  const r = await api.test_plugin_code(name, code);
  const out = $('#plugin-ai-test-output');
  out.classList.remove('hidden');
  if (r?.error) {
    out.textContent = '❌ ' + r.error;
    out.classList.add('test-fail');
    out.classList.remove('test-pass');
    $('#plugin-ai-test-status').textContent = '❌ Test failed';
  } else {
    out.textContent = r.output || '(no output)';
    out.classList.add('test-pass');
    out.classList.remove('test-fail');
    $('#plugin-ai-test-status').textContent = '✅ Test passed';
  }
});

$('#btn-plugin-ai-save').addEventListener('click', async () => {
  const name = $('#plugin-ai-name').value.trim();
  const code = $('#plugin-ai-code').value.trim();
  if (!name) { showToast('Enter a plugin name first', 'warn'); return; }
  if (!code) { showToast('No code to save', 'warn'); return; }

  let r;
  if (_pluginEditName) {
    r = await api.update_plugin_code(_pluginEditName, code);
  } else {
    r = await api.save_generated_plugin(name, code);
  }
  if (r?.error) {
    showToast('Save failed: ' + r.error, 'error');
    return;
  }
  if (_pluginEditName) {
    showToast(`Plugin "${r.name}" updated!`, 'success');
  } else {
    showToast(`Plugin "${r.name}" saved and loaded!`, 'success');
  }
  closeModal('modal-plugin-ai');
  _setPluginAiCreateMode();
  await refreshPlugins();
  openModal('modal-plugins');
});

$('#btn-hf-search').addEventListener('click', hfSearch);
$('#hf-query').addEventListener('keydown', (e) => {
  if (e.key === 'Enter') hfSearch();
});

async function hfSearch() {
  const q = $('#hf-query').value.trim();
  if (!q) return;
  $('#hf-status').textContent = 'Searching HuggingFace…';
  $('#hf-results').innerHTML = '';

  const r = await api.search_hf_models(q);
  if (r.error) {
    $('#hf-status').textContent = `Error: ${r.error}`;
    return;
  }
  if (!r.results || r.results.length === 0) {
    $('#hf-status').textContent = 'No GGUF files found.';
    return;
  }
  $('#hf-status').textContent = `Found ${r.results.length} files (recommended: ≤${r.max_gb}GB for ${r.system_ram}GB RAM)`;

  const list = $('#hf-results');
  for (const m of r.results) {
    const badge = m.unknown ? '❓' : (m.compatible ? '✅' : '⚠');
    const sizeStr = m.size_gb != null ? `${m.size_gb} GB` : 'size unknown';
    const fitLabel = m.compatible ? 'fits' : (m.unknown ? '' : 'may be slow');
    const card = document.createElement('div');
    card.className = 'hf-card';
    card.innerHTML = `
      <div class="hf-info">
        <div class="hf-name">${badge} ${escapeHtml(m.model_id)}</div>
        <div class="hf-file">${escapeHtml(m.filename)} — <strong>${sizeStr}</strong>${fitLabel ? ' · ' + fitLabel : ''}</div>
      </div>
    `;
    const btn = document.createElement('button');
    btn.className = 'btn accent';
    btn.textContent = '⬇ Download';
    btn.addEventListener('click', async () => {
      btn.disabled = true;
      btn.textContent = 'Downloading…';
      await api.download_hf_model(m.model_id, m.filename);
    });
    card.appendChild(btn);
    list.appendChild(card);
  }
}

// ═══ FILE UPLOAD ══════════════════════════════════════════════

function syncUploadButton(fileName) {
  attachedDocName = fileName;
  if (fileName) {
    $('#btn-upload').classList.add('active');
    $('#btn-upload').title = `File loaded: ${fileName} (click to clear)`;
  } else {
    $('#btn-upload').classList.remove('active');
    $('#btn-upload').title = 'Upload file';
  }
}

$('#btn-upload').addEventListener('click', () => {
  if (attachedDocName) {
    clearUploadedDocument();
    return;
  }
  uploadDocument();
});

async function clearUploadedDocument() {
  await api.clear_uploaded_document();
  attachedDocName = null;
  $('#btn-upload').classList.remove('active');
  $('#btn-upload').title = 'Upload file';
  showToast('Uploaded file context cleared', 'info');
  
  // Display file cleared message in chat
  appendMessage('assistant', '📄 **File context cleared.** Upload a new file to continue analysis.');
}

async function uploadDocument() {
  const r = await api.open_document_dialog();
  if (r?.error) {
    showToast(r.error, 'error');
    return;
  }
  if (!r?.selected) {
    return;
  }
  attachedDocName = r.name;
  $('#btn-upload').classList.add('active');
  $('#btn-upload').title = `File loaded: ${r.name} (click to clear)`;

  if (r.processing) {
    // PDF is being processed in background (OCR etc.)
    showToast(`Processing PDF: ${r.name} — please wait...`, 'info', 8000);
    const fileIcon = '📄';
    appendMessage('assistant', `${fileIcon} **Processing PDF:** ${r.name} — extracting text and running OCR on scanned pages. This may take a moment for large files...`);
  } else {
    showToast(`File loaded: ${r.name} (${r.chars.toLocaleString()} chars)`, 'info');
    const fileIcon = r.name.endsWith('.csv') || r.name.endsWith('.xlsx') || r.name.endsWith('.xls') ? '📊' : '📄';
    appendMessage('assistant', `${fileIcon} **File uploaded:** ${r.name} (${r.chars.toLocaleString()} chars). You can now ask questions about this file.`);
  }
}

// Handle async PDF processing completion
window.addEventListener('file_upload_done', (e) => {
  const d = e.detail;
  if (d?.error) {
    showToast(`PDF error: ${d.error}`, 'error');
    appendMessage('assistant', `❌ **PDF processing failed:** ${d.error}`);
    return;
  }
  const pages = d.pages || 0;
  const chars = d.chars || 0;
  showToast(`PDF ready: ${pages} pages, ${chars.toLocaleString()} chars`, 'info');
  appendMessage('assistant', `📄 **PDF ready:** ${d.name} — ${pages} pages, ${chars.toLocaleString()} chars extracted. You can now ask questions about this file.`);
});

$('#btn-image').addEventListener('click', async () => {
  if (attachedImageName) {
    await api.clear_attached_image();
    attachedImageName = null;
    $('#btn-image').classList.remove('active');
    $('#btn-image').textContent = '▦';
    showToast('Image cleared', 'info');
    return;
  }

  const r = await api.open_image_dialog();
  if (r?.error) {
    showToast(r.error, 'error');
    return;
  }
  if (!r?.selected) {
    return;
  }

  attachedImageName = r.name;
  $('#btn-image').classList.add('active');
  $('#btn-image').textContent = '✓';

  const vision = await api.get_vision_status();
  if (vision?.ready) {
    showToast(`Image attached: ${r.name}`, 'info');
  } else {
    showToast(
      `Image attached (${r.name}). Vision model assets missing; text fallback will be used.`,
      'warn',
      5500
    );
  }
});

let voiceRecognition = null;
let voiceListening = false;

$('#btn-voice').addEventListener('click', async () => {
  const btn = $('#btn-voice');
  const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
  if (!SpeechRecognition) {
    showToast('Voice input is not supported by this webview runtime.', 'warn', 4500);
    return;
  }

  if (voiceListening && voiceRecognition) {
    try { voiceRecognition.stop(); } catch (_e) {}
    voiceListening = false;
    btn.classList.remove('active');
    btn.textContent = '◉';
    return;
  }

  voiceRecognition = new SpeechRecognition();
  voiceRecognition.lang = 'en-US';
  voiceRecognition.interimResults = false;
  voiceRecognition.maxAlternatives = 1;

  voiceRecognition.onstart = () => {
    voiceListening = true;
    btn.classList.add('active');
    btn.textContent = '●';
    showToast('Listening...', 'info', 1200);
  };

  voiceRecognition.onresult = (event) => {
    const transcript = event?.results?.[0]?.[0]?.transcript?.trim() || '';
    if (transcript) {
      inputEl.value = inputEl.value
        ? `${inputEl.value.trim()} ${transcript}`
        : transcript;
      autoResize(inputEl);
      updateSendButton();
      $('#token-counter').textContent = `~${Math.ceil(inputEl.value.trim().length / 4)} tokens`;
    }
  };

  voiceRecognition.onerror = () => {
    showToast('Voice recognition failed.', 'warn');
  };

  voiceRecognition.onend = () => {
    voiceListening = false;
    btn.classList.remove('active');
    btn.textContent = '◉';
  };

  try {
    voiceRecognition.start();
  } catch (_e) {
    showToast('Could not start voice input.', 'warn');
  }
});

// ═══ SIDEBAR TOGGLE ═══════════════════════════════════════════

$('#sidebar-toggle').addEventListener('click', () => {
  sidebarVisible = !sidebarVisible;
  $('#sidebar').classList.toggle('collapsed', !sidebarVisible);
});

// ═══ KEYBOARD SHORTCUTS ═══════════════════════════════════════

document.addEventListener('keydown', (e) => {
  // Ctrl+Shift+R: reload app (development mode)
  if (e.ctrlKey && e.shiftKey && e.key === 'R') {
    e.preventDefault();
    location.reload();
  }
  // Ctrl+B: toggle sidebar
  if (e.ctrlKey && e.key === 'b') {
    e.preventDefault();
    $('#sidebar-toggle').click();
  }
  // Ctrl+N: new chat
  if (e.ctrlKey && e.key === 'n') {
    e.preventDefault();
    $('#btn-new-chat').click();
  }
  // Escape: stop generation or close modals
  if (e.key === 'Escape') {
    const openModal = document.querySelector('.modal:not(.hidden)');
    if (openModal) {
      openModal.classList.add('hidden');
    } else if (stopBtn.style.display !== 'none') {
      api.stop_generation();
    } else {
      const agentStopBtn = $('#btn-agent-stop');
      if (agentStopBtn && agentStopBtn.style.display !== 'none') {
        api.stop_generation();
      }
    }
  }
  // Ctrl+L: load model
  if (e.ctrlKey && e.key === 'l') {
    e.preventDefault();
    $('#btn-load-model').click();
  }
});

// ═══ AGENT PANEL ═════════════════════════════════════════════

let agentFiles = [];
let agentMessages = [];
let selectedInstructionName = null;
let activeInstructionFormat = 'excel';
let agentGenerationInProgress = false;
let agentStopRequested = false;

function setAgentActionButtons(processing) {
  const send = $('#btn-agent-send');
  const stop = $('#btn-agent-stop');
  if (!send || !stop) return;
  if (processing) {
    send.disabled = true;
    stop.disabled = false;
  } else {
    send.disabled = !fullAccessGranted;
    stop.disabled = true;
  }
}

// Ensure Agent Send/Stop state is initialized on load.
setAgentActionButtons(false);

// ── Instruction Templates ─────────────────────────────────────
const TEMPLATES_KEY = 'simple_ai_instruction_templates';

function normalizeTemplateEntry(entry) {
  let value = entry;
  if (typeof value === 'string') {
    try {
      value = JSON.parse(value);
    } catch (_) {
      return { role: '', task: value, steps: '', format: 'excel' };
    }
  }

  if (!value || typeof value !== 'object') {
    return { role: '', task: '', steps: '', format: 'excel' };
  }

  return {
    role: String(value.role || '').trim(),
    task: String(value.task || value.text || '').trim(),
    steps: String(value.steps || '').trim(),
    format: String(value.format || 'excel').trim() || 'excel',
  };
}

function getTemplates() {
  try {
    const raw = JSON.parse(localStorage.getItem(TEMPLATES_KEY) || '{}');
    const normalized = {};
    for (const [name, entry] of Object.entries(raw || {})) {
      normalized[name] = normalizeTemplateEntry(entry);
    }
    return normalized;
  } catch { return {}; }
}

function saveTemplatesLocal(templates) {
  localStorage.setItem(TEMPLATES_KEY, JSON.stringify(templates));
}

async function syncTemplatesFromBackend() {
  if (!api || typeof api.get_instruction_templates !== 'function') return;
  try {
    const remote = await api.get_instruction_templates();
    if (!remote || typeof remote !== 'object') return;

    const merged = getTemplates();
    for (const [name, entry] of Object.entries(remote)) {
      merged[name] = normalizeTemplateEntry(entry);
    }
    saveTemplatesLocal(merged);
  } catch (_) {
    // Keep local templates if backend templates are unavailable.
  }
}

async function getPredefinedTemplates() {
  if (!api || typeof api.get_instruction_templates !== 'function') {
    return {};
  }
  try {
    const remote = await api.get_instruction_templates();
    if (!remote || typeof remote !== 'object') return {};
    const out = {};
    for (const [name, entry] of Object.entries(remote)) {
      out[name] = normalizeTemplateEntry(entry);
    }
    return out;
  } catch (_) {
    return {};
  }
}

function fillInstructionFormFromTemplate(name, tpl) {
  const t = normalizeTemplateEntry(tpl);
  const nameInput = $('#agent-instr-name');
  const roleInput = $('#agent-instr-role');
  const taskInput = $('#agent-instr-task');
  const stepsInput = $('#agent-instr-steps');
  const formatInput = $('#agent-instr-format');

  if (nameInput && !nameInput.disabled) nameInput.value = String(name || '').trim();
  if (roleInput) roleInput.value = t.role || '';
  if (taskInput) taskInput.value = t.task || '';
  if (stepsInput) stepsInput.value = t.steps || '';
  if (formatInput) formatInput.value = t.format || 'excel';
}

async function openTemplateImportPicker() {
  const templates = await getPredefinedTemplates();
  const names = Object.keys(templates).sort();
  if (!names.length) {
    showToast('No predefined templates found', 'warn');
    return;
  }

  const list = $('#template-import-list');
  if (!list) return;
  list.innerHTML = '';

  for (const name of names) {
    const tpl = templates[name];
    const role = String(tpl.role || '').trim();
    const task = String(tpl.task || '').trim();
    const stepsCount = String(tpl.steps || '').split('\n').filter(s => s.trim()).length;
    const item = document.createElement('div');
    item.className = 'template-pick-item';
    item.innerHTML = `
      <div class="template-pick-name">${escapeHtml(name)}</div>
      <div class="template-pick-meta">${escapeHtml(role || 'No role')} • ${escapeHtml(tpl.format || 'excel')} • ${stepsCount} step(s)</div>
      <div class="template-pick-meta">${escapeHtml(task.substring(0, 120))}${task.length > 120 ? '…' : ''}</div>
    `;
    item.addEventListener('click', () => {
      fillInstructionFormFromTemplate(name, tpl);
      closeModal('modal-template-import');
      showToast(`Loaded template: ${name}`, 'success');
    });
    list.appendChild(item);
  }

  openModal('modal-template-import');
}

// ── Processed Output Files List ───────────────────────────────

async function refreshProcessedFiles() {
  const list = $('#processed-files-list');
  if (!list) return;
  try {
    let result;
    if (api?.list_processed_files) {
      result = await api.list_processed_files();
    } else {
      // Fallback: method may not be available yet
      list.innerHTML = '<p class="muted small" style="padding:12px;text-align:center;">Restart app to enable file listing.</p>';
      return;
    }
    if (!result || result.error || !result.files) {
      list.innerHTML = `<p class="muted small" style="padding:12px;text-align:center;">Could not load files.</p>`;
      return;
    }
    if (result.files.length === 0) {
      list.innerHTML = '<p class="muted small" style="padding:12px;text-align:center;">No output files yet.</p>';
      return;
    }
    list.innerHTML = '';
    for (const f of result.files) {
      const ext = f.name.split('.').pop().toUpperCase();
      const sizeStr = f.size < 1024 ? f.size + ' B'
        : f.size < 1048576 ? (f.size / 1024).toFixed(1) + ' KB'
        : (f.size / 1048576).toFixed(1) + ' MB';
      const date = new Date(f.modified * 1000);
      const dateStr = date.toLocaleDateString() + ' ' + date.toLocaleTimeString([], {hour:'2-digit', minute:'2-digit'});

      const card = document.createElement('div');
      card.className = 'output-file-card';
      card.innerHTML = `
        <div class="output-file-info">
          <div class="output-file-name" title="${escapeHtml(f.name)}">${escapeHtml(f.name)}</div>
          <div class="output-file-meta">${ext} · ${sizeStr} · ${dateStr}</div>
        </div>
        <div class="output-file-actions">
          <button class="btn-open-file" title="Open">📂</button>
          <button class="btn-del-file" title="Delete">🗑</button>
        </div>
      `;

      card.querySelector('.btn-open-file').addEventListener('click', async () => {
        const res = await api.open_file_location(f.path);
        if (res?.error) showToast(`Could not open: ${res.error}`, 'error');
      });

      card.querySelector('.btn-del-file').addEventListener('click', async () => {
        if (!await showConfirm(`Delete "${f.name}"?`, 'Delete File')) return;
        const res = await api.delete_processed_file(f.path);
        if (res?.ok) {
          showToast('File deleted', 'info');
          refreshProcessedFiles();
        } else {
          showToast(`Could not delete: ${res?.error || 'Unknown error'}`, 'error');
        }
      });

      list.appendChild(card);
    }
  } catch (err) {
    console.error('refreshProcessedFiles error:', err);
    list.innerHTML = `<p class="muted small" style="padding:12px;text-align:center;">Error loading files: ${escapeHtml(err.message || String(err))}</p>`;
  }
}

$('#btn-refresh-files')?.addEventListener('click', refreshProcessedFiles);

// ── Right Panel View Switching ────────────────────────────────

function showAgentView(viewName) {
  // viewName: 'empty', 'create', 'chat'
  ['empty', 'create', 'chat'].forEach(v => {
    const el = $(`#agent-view-${v}`);
    if (el) el.classList.toggle('hidden', v !== viewName);
  });
}

// ── Instruction List (Left Panel) ─────────────────────────────

function refreshInstructionList() {
  const list = $('#instruction-list');
  if (!list) return;
  const templates = getTemplates();
  const names = Object.keys(templates).sort();

  if (names.length === 0) {
    list.innerHTML = '<p class="muted small" style="padding:12px;text-align:center;">No saved instructions yet.<br>Click "+ New" to create one.</p>';
    return;
  }

  list.innerHTML = '';
  for (const name of names) {
    const tpl = templates[name];
    const previewText = typeof tpl === 'string' ? tpl : (tpl.task || tpl.text || '');
    const roleBadge = (typeof tpl === 'object' && tpl.role) ? `<span class="role-badge">${escapeHtml(tpl.role.substring(0, 30))}</span> ` : '';
    const card = document.createElement('div');
    card.className = 'instruction-card' + (name === selectedInstructionName ? ' selected' : '');

    const preview = previewText.length > 80
      ? previewText.substring(0, 80) + '…'
      : previewText;

    card.innerHTML = `
      <div class="instruction-card-title">
        <span>${escapeHtml(name)}</span>
        <div class="instruction-card-actions">
          <button class="btn-edit-instr" title="Edit">✏</button>
          <button class="btn-del-instr" title="Delete">🗑</button>
        </div>
      </div>
      <div class="instruction-card-preview">${roleBadge}${escapeHtml(preview)}</div>
    `;

    // Click to select → open agent chat
    card.addEventListener('click', (e) => {
      if (e.target.closest('.instruction-card-actions')) return;
      openAgentChat(name);
    });

    // Edit → open form pre-filled
    card.querySelector('.btn-edit-instr').addEventListener('click', (e) => {
      e.stopPropagation();
      openInstructionForm(name);
    });

    // Delete
    card.querySelector('.btn-del-instr').addEventListener('click', async (e) => {
      e.stopPropagation();
      if (!await showConfirm(`Delete "${name}"?`, 'Delete Instruction')) return;
      const t = getTemplates();
      delete t[name];
      saveTemplatesLocal(t);
      api.delete_instruction_template(name);
      if (selectedInstructionName === name) {
        selectedInstructionName = null;
        showAgentView('empty');
      }
      refreshInstructionList();
      showToast(`Deleted: ${name}`, 'info');
    });

    list.appendChild(card);
  }
}

// ── Open Agent Chat for a selected instruction ────────────────

function openAgentChat(name) {
  const templates = getTemplates();
  const tpl = templates[name];
  if (!tpl) return;

  selectedInstructionName = name;
  // Support both old format (text) and new format (role/task/steps)
  const instrRole = (typeof tpl === 'object' && tpl.role) ? tpl.role : '';
  const instrTask = (typeof tpl === 'object' && tpl.task) ? tpl.task : (typeof tpl === 'string' ? tpl : (tpl.text || ''));
  const instrSteps = (typeof tpl === 'object' && tpl.steps) ? tpl.steps : '';
  activeInstructionFormat = (typeof tpl === 'object' && tpl.format) ? tpl.format : 'excel';

  // Update header
  const nameEl = $('#agent-active-name');
  if (nameEl) nameEl.textContent = `📋 ${name}`;

  // Clear old messages and show chat view
  const msgs = $('#agent-messages');
  if (msgs) msgs.innerHTML = '';
  agentMessages = [];
  agentFiles = [];
  renderAgentFileChips();

  // Show welcome system message with config summary
  let welcomeHtml = `Instruction loaded: <strong>${escapeHtml(name)}</strong>.`;
  if (instrRole) welcomeHtml += `<br><span class="muted small">Role: ${escapeHtml(instrRole)}</span>`;
  if (instrTask) welcomeHtml += `<br><span class="muted small">Task: ${escapeHtml(instrTask.substring(0, 100))}${instrTask.length > 100 ? '…' : ''}</span>`;
  if (instrSteps) {
    const stepCount = instrSteps.split('\n').filter(s => s.trim()).length;
    welcomeHtml += `<br><span class="muted small">Steps: ${stepCount} defined</span>`;
  }
  welcomeHtml += `<br>Upload files and send to process, or ask a question.`;
  addAgentMessage('system', welcomeHtml);

  showAgentView('chat');
  refreshInstructionList();
}

// ── Instruction Form (Create / Edit) ──────────────────────────

let editingInstructionName = null;

function openInstructionForm(existingName) {
  editingInstructionName = existingName || null;
  const title = $('#agent-form-title');
  const nameInput = $('#agent-instr-name');
  const formatSelect = $('#agent-instr-format');

  const roleInput = $('#agent-instr-role');
  const taskInput = $('#agent-instr-task');
  const stepsInput = $('#agent-instr-steps');

  if (existingName) {
    const templates = getTemplates();
    const tpl = templates[existingName];
    // Support both old (text) and new (role/task/steps) formats
    const instrRole = (typeof tpl === 'object' && tpl.role) ? tpl.role : '';
    const instrTask = (typeof tpl === 'object' && tpl.task) ? tpl.task : (typeof tpl === 'string' ? tpl : (tpl.text || ''));
    const instrSteps = (typeof tpl === 'object' && tpl.steps) ? tpl.steps : '';
    const instrFormat = (typeof tpl === 'object' && tpl.format) ? tpl.format : 'excel';
    title.textContent = 'Edit Instruction';
    nameInput.value = existingName;
    roleInput.value = instrRole;
    taskInput.value = instrTask;
    stepsInput.value = instrSteps;
    formatSelect.value = instrFormat;
    nameInput.disabled = true;
  } else {
    title.textContent = 'New Instruction';
    nameInput.value = '';
    roleInput.value = '';
    taskInput.value = '';
    stepsInput.value = '';
    formatSelect.value = 'excel';
    nameInput.disabled = false;
  }

  showAgentView('create');
}

// New instruction button
$('#btn-new-instruction')?.addEventListener('click', () => {
  openInstructionForm(null);
});

// Save instruction
$('#btn-save-instruction')?.addEventListener('click', () => {
  const name = $('#agent-instr-name').value.trim();
  const role = $('#agent-instr-role').value.trim();
  const task = $('#agent-instr-task').value.trim();
  const steps = $('#agent-instr-steps').value.trim();
  const format = $('#agent-instr-format').value;

  if (!name) { showToast('Enter a name', 'warning'); return; }
  if (!task) { showToast('Enter a task description', 'warning'); return; }

  const templates = getTemplates();
  templates[name] = { role, task, steps, format };
  saveTemplatesLocal(templates);
  api.save_instruction_template(name, JSON.stringify({ role, task, steps, format }));

  refreshInstructionList();
  showToast(`Saved: ${name}`, 'success');

  // After saving, open the chat for this instruction
  openAgentChat(name);
});

// Cancel instruction form
$('#btn-cancel-instruction')?.addEventListener('click', () => {
  editingInstructionName = null;
  if (selectedInstructionName) {
    showAgentView('chat');
  } else {
    showAgentView('empty');
  }
});

// Import predefined templates on demand (no auto-import on startup).
$('#btn-import-predefined')?.addEventListener('click', async () => {
  const btn = $('#btn-import-predefined');
  const oldText = btn.textContent;
  btn.disabled = true;
  btn.textContent = 'Loading...';
  try {
    await openTemplateImportPicker();
  } catch (e) {
    showToast(`Import failed: ${e?.message || e}`, 'error');
  } finally {
    btn.disabled = false;
    btn.textContent = oldText;
  }
});

$('#template-import-cancel')?.addEventListener('click', () => closeModal('modal-template-import'));
$('#modal-template-import')?.addEventListener('click', (e) => {
  if (e.target?.id === 'modal-template-import') closeModal('modal-template-import');
});

// Rewrite with AI
$('#btn-rewrite-ai')?.addEventListener('click', async () => {
  const role = $('#agent-instr-role').value.trim();
  const task = $('#agent-instr-task').value.trim();
  const steps = $('#agent-instr-steps').value.trim();

  if (!role && !task && !steps) {
    showToast('Write something first — role, task, or steps', 'warn');
    return;
  }

  const btn = $('#btn-rewrite-ai');
  const oldText = btn.textContent;
  btn.disabled = true;
  btn.textContent = '⏳ Rewriting...';

  try {
    const r = await api.rewrite_instruction({ role, task, steps });
    if (r?.error) {
      showToast(r.error, 'error');
      return;
    }

    // Normalize payload variants without changing backend rewrite logic.
    let payload = r;
    if (typeof payload === 'string') {
      try { payload = JSON.parse(payload); } catch (_) { payload = { task: payload }; }
    }

    const normRole = String(payload?.role ?? payload?.Role ?? '').trim();
    let normTask = String(payload?.task ?? payload?.Task ?? '').trim();
    let normSteps = String(payload?.steps ?? payload?.Steps ?? '').trim();

    const extractLabeled = (text, label) => {
      const src = String(text || '');
      const re = new RegExp(`(?:^|\\n)\\s*${label}\\s*:\\s*([\\s\\S]*?)(?=\\n\\s*(?:Role|Task|Steps)\\s*:|$)`, 'i');
      const m = src.match(re);
      return m ? String(m[1]).trim() : '';
    };

    const stripTaskNoise = (text) => {
      let t = String(text || '').trim();
      if (!t) return '';
      // If task accidentally contains steps-style bullets/sections, keep first sentence/line only.
      if (/\n\s*(?:\d+\.|\*|[-•])\s+/.test(t) || /^\s*\d+\./m.test(t)) {
        t = t.split(/\n/)[0].trim();
      }
      // Remove leading "Task:" label if present.
      t = t.replace(/^\s*Task\s*:\s*/i, '').trim();
      return t;
    };

    // Some models return a JSON object string in task; extract and map to fields.
    if (normTask.startsWith('{') && normTask.endsWith('}')) {
      try {
        const nested = JSON.parse(normTask);
        if (!normRole && nested?.role != null) {
          $('#agent-instr-role').value = String(nested.role).trim();
        }
        if (nested?.task != null) {
          normTask = String(nested.task).trim();
        }
        if (!normSteps && nested?.steps != null) {
          normSteps = String(nested.steps).trim();
        }
      } catch (_) {
        // keep original task text
      }
    }

    // Handle plaintext/markdown responses with explicit labels.
    if (!normTask && typeof payload === 'string') {
      normTask = extractLabeled(payload, 'Task');
    }
    if (!normSteps && typeof payload === 'string') {
      normSteps = extractLabeled(payload, 'Steps');
    }

    normTask = stripTaskNoise(normTask);
    if (!normTask) {
      // Keep user's current task rather than writing unrelated content.
      normTask = task;
    }

    if (normRole) $('#agent-instr-role').value = normRole;
    if (normTask) $('#agent-instr-task').value = normTask;
    if (normSteps) $('#agent-instr-steps').value = normSteps;
    showToast('Instruction polished by AI', 'success');
  } catch (e) {
    showToast('Rewrite failed: ' + e.message, 'error');
  } finally {
    btn.disabled = false;
    btn.textContent = oldText;
  }
});

// Back button from chat to empty
$('#btn-back-to-instructions')?.addEventListener('click', () => {
  selectedInstructionName = null;
  showAgentView('empty');
  refreshInstructionList();
});

// ── Expand / Collapse Chat ────────────────────────────────────

$('#btn-expand-chat')?.addEventListener('click', () => {
  const area = $('#agent-area');
  const btn = $('#btn-expand-chat');
  area.classList.toggle('chat-expanded');
  btn.textContent = area.classList.contains('chat-expanded') ? '⤡' : '⤢';
});

// ── Resize Handle ─────────────────────────────────────────────

(function setupResize() {
  const handle = $('#agent-resize-handle');
  const leftPanel = $('#agent-instructions-panel');
  if (!handle || !leftPanel) return;

  let dragging = false;
  let startX = 0;
  let startW = 0;

  handle.addEventListener('mousedown', (e) => {
    e.preventDefault();
    dragging = true;
    startX = e.clientX;
    startW = leftPanel.getBoundingClientRect().width;
    handle.classList.add('dragging');
    document.body.style.cursor = 'col-resize';
    document.body.style.userSelect = 'none';
  });

  document.addEventListener('mousemove', (e) => {
    if (!dragging) return;
    const delta = e.clientX - startX;
    const newW = Math.max(200, Math.min(600, startW + delta));
    leftPanel.style.width = newW + 'px';
  });

  document.addEventListener('mouseup', () => {
    if (!dragging) return;
    dragging = false;
    handle.classList.remove('dragging');
    document.body.style.cursor = '';
    document.body.style.userSelect = '';
  });
})();

// ── Agent Status ──────────────────────────────────────────────

function setAgentStatus(text, active = false) {
  const bar = $('#agent-status-bar');
  const txt = $('#agent-status-text');
  if (!bar || !txt) return;
  txt.textContent = text;
  bar.classList.toggle('active', active);
}

// ── Agent File Upload ─────────────────────────────────────────

$('#btn-agent-upload')?.addEventListener('click', () => {
  $('#agent-file-input')?.click();
});

$('#agent-file-input')?.addEventListener('change', (e) => {
  const newFiles = [];
  for (const file of e.target.files) {
    if (!agentFiles.find(f => f.name === file.name)) {
      agentFiles.push(file);
      newFiles.push(file);
    }
  }
  renderAgentFileChips();
});

// Drag-drop on the entire agent chat panel
const agentChatPanel = $('#agent-chat-panel');
if (agentChatPanel) {
  agentChatPanel.addEventListener('dragover', (e) => {
    e.preventDefault();
    agentChatPanel.style.outline = '2px dashed var(--accent)';
  });
  agentChatPanel.addEventListener('dragleave', () => {
    agentChatPanel.style.outline = '';
  });
  agentChatPanel.addEventListener('drop', (e) => {
    e.preventDefault();
    agentChatPanel.style.outline = '';
    const newDropFiles = [];
    for (const file of e.dataTransfer.files) {
      if (!agentFiles.find(f => f.name === file.name)) {
        agentFiles.push(file);
        newDropFiles.push(file);
      }
    }
    renderAgentFileChips();
  });
}

function renderAgentFileChips() {
  const container = $('#agent-file-chips');
  if (!container) return;
  container.innerHTML = '';
  for (const file of agentFiles) {
    const chip = document.createElement('span');
    chip.className = 'file-chip';
    const fileIcon = file.name.endsWith('.csv') || file.name.endsWith('.xlsx') || file.name.endsWith('.xls') ? '📊' : '📄';
    chip.innerHTML = `${fileIcon} ${escapeHtml(file.name)} <button class="chip-remove" title="Remove">✕</button>`;
    chip.querySelector('.chip-remove').addEventListener('click', () => {
      agentFiles = agentFiles.filter(f => f.name !== file.name);
      renderAgentFileChips();
    });
    container.appendChild(chip);
  }
}

// ── Agent Chat Messages ───────────────────────────────────────

function addAgentMessage(role, html) {
  const container = $('#agent-messages');
  if (!container) return;
  const msg = document.createElement('div');
  msg.className = `agent-msg ${role}`;
  msg.innerHTML = html;
  container.appendChild(msg);
  container.scrollTop = container.scrollHeight;
  agentMessages.push({ role, html });
}

// ── Read File Content (for passing to backend) ────────────────

function readFileAsText(file) {
  return new Promise((resolve) => {
    const reader = new FileReader();
    reader.onload = () => resolve(reader.result);
    reader.onerror = () => resolve('');
    reader.readAsText(file);
  });
}

function readFileAsDataUrl(file) {
  return new Promise((resolve) => {
    const reader = new FileReader();
    reader.onload = () => resolve(reader.result || '');
    reader.onerror = () => resolve('');
    reader.readAsDataURL(file);
  });
}

function isBinaryAgentFile(fileName) {
  const ext = (fileName.split('.').pop() || '').toLowerCase();
  return ['pdf', 'docx', 'xlsx', 'xls', 'png', 'jpg', 'jpeg', 'webp', 'bmp', 'tiff'].includes(ext);
}

async function readFileForAgent(file) {
  if (isBinaryAgentFile(file.name)) {
    const dataUrl = await readFileAsDataUrl(file);
    const base64 = String(dataUrl).includes(',') ? String(dataUrl).split(',')[1] : '';
    return {
      name: file.name,
      size: file.size,
      mime_type: file.type || '',
      content_base64: base64,
    };
  }

  const content = await readFileAsText(file);
  return {
    name: file.name,
    size: file.size,
    mime_type: file.type || '',
    content: content,
  };
}

// ── Agent Send / Process ──────────────────────────────────────

$('#btn-agent-send')?.addEventListener('click', agentSend);
$('#btn-agent-stop')?.addEventListener('click', async () => {
  if (!agentGenerationInProgress) return;
  const stop = $('#btn-agent-stop');
  if (stop) stop.disabled = true;
  agentStopRequested = true;
  setAgentStatus('Stopping AI generation...', true);
  try {
    await api.stop_generation();
  } catch (_) {
    // Ignore stop RPC failure; backend may already be finishing.
  }
});

$('#agent-input')?.addEventListener('keydown', (e) => {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault();
    if (agentGenerationInProgress) return;
    agentSend();
  }
});

// Auto-resize agent input
$('#agent-input')?.addEventListener('input', function() {
  this.style.height = 'auto';
  this.style.height = Math.min(this.scrollHeight, 120) + 'px';
});

async function agentSend() {
  if (!fullAccessGranted) {
    const unlocked = await ensureActivationGate(true);
    if (!unlocked) {
      showToast('Activation required to continue.', 'warn');
      return;
    }
  }

  if (agentGenerationInProgress) {
    showToast('AI generation is already in progress', 'warning');
    return;
  }

  const input = $('#agent-input');
  const text = input?.value?.trim();
  if (!text && agentFiles.length === 0) {
    showToast('Type a message or upload files', 'warning');
    return;
  }

  // Get saved instruction config from the selected instruction
  const templates = getTemplates();
  const tpl = templates[selectedInstructionName];
  const agentRole = tpl ? (typeof tpl === 'object' && tpl.role ? tpl.role : '') : '';
  const agentTask = tpl ? (typeof tpl === 'object' && tpl.task ? tpl.task : (typeof tpl === 'string' ? tpl : (tpl.text || ''))) : '';
  const agentSteps = tpl ? (typeof tpl === 'object' && tpl.steps ? tpl.steps : '') : '';
  const format = tpl ? (typeof tpl === 'object' && tpl.format ? tpl.format : 'excel') : 'excel';

  // Show user message
  let userHtml = '';
  if (text) {
    userHtml += escapeHtml(text);
  }
  addAgentMessage('user', userHtml || '📎 Processing attached files...');

  // Clear input
  input.value = '';
  input.style.height = 'auto';

  // Show processing status
  agentGenerationInProgress = true;
  agentStopRequested = false;
  setAgentActionButtons(true);
  setAgentStatus('Processing...', true);
  addAgentMessage('system', '⏳ Processing your request...');

  try {
    // Read file contents
    const fileData = [];
    for (const file of agentFiles) {
      setAgentStatus(`Reading ${file.name}...`, true);
      const prepared = await readFileForAgent(file);
      fileData.push(prepared);
    }

    if (fileData.length > 0) {
      // Build structured instructions from Role-Task-Steps
      let fullInstructions = '';
      if (agentRole) fullInstructions += `Role: ${agentRole}\n`;
      if (agentTask) fullInstructions += `Task: ${agentTask}\n`;
      if (agentSteps) fullInstructions += `Steps:\n${agentSteps}\n`;
      if (text) fullInstructions += `\nAdditional notes: ${text}`;
      if (!fullInstructions.trim()) fullInstructions = text || '';
      setAgentStatus('AI processing...', true);
      
      const result = await api.process_files_with_ai(fileData, fullInstructions, format);

      // Remove the "Processing..." system message
      removeLastSystemMsg();

      if (result?.stopped || (result?.error && /stopp?ed by user|generation stopped/i.test(String(result.error)))) {
        addAgentMessage('assistant', '⏹ Generation stopped by user.');
        setAgentStatus('Stopped', false);
        return;
      }

      if (result?.error) {
        addAgentMessage('assistant', `❌ Error: ${escapeHtml(result.error)}`);
        setAgentStatus('Error', false);
        return;
      }

      if (result?.file_path) {
        const fileName = result.file_path.split(/[\\\/]/).pop();
        const fileExt = fileName.split('.').pop().toUpperCase();
        let responseHtml = '';
        if (result.response_text) {
          responseHtml = `<div style="margin-bottom:10px;">${renderMarkdown(result.response_text)}</div>`;
        }
        let warningHtml = '';
        if (result.warning) {
          warningHtml = `<div class="muted small" style="margin:8px 0;color:#f39c12;">⚠ ${escapeHtml(result.warning)}</div>`;
        }
        addAgentMessage('assistant', `
          ${responseHtml}
          ${warningHtml}
          <div>✅ Processing complete!</div>
          <div class="muted small" style="margin:4px 0;">Output: ${escapeHtml(fileName)}</div>
          <div style="display:flex;gap:8px;margin-top:8px;">
            <button class="download-btn" onclick="openAgentFile('${escapeJs(result.file_path)}')">
              ⬇ Open ${fileExt} File
            </button>
            <button class="download-btn" style="background:var(--error,#e74c3c);" onclick="deleteAgentFile('${escapeJs(result.file_path)}', this)">
              🗑 Delete
            </button>
          </div>
        `);
        setAgentStatus('Complete', false);
        refreshProcessedFiles();
      } else {
        // No file_path and no error: UI-only mode or partial result.
        const msg = result?.response_text || 'Analysis completed but no output file was generated.';
        if (result?.success) {
          addAgentMessage('assistant', renderMarkdown(msg));
        } else {
          addAgentMessage('assistant', `⚠ ${renderMarkdown(msg)}`);
        }
        setAgentStatus(result?.success ? 'Complete' : 'Failed', false);
      }
    } else if (text) {
      // Pure chat / question (no files) — pass role/task/steps for context
      setAgentStatus('Generating response...', true);

      const result = await api.agent_chat(text, agentRole, agentTask, agentSteps);
      removeLastSystemMsg();

      if (result?.stopped || (result?.error && /stopp?ed by user|generation stopped/i.test(String(result.error)))) {
        addAgentMessage('assistant', '⏹ Generation stopped by user.');
        setAgentStatus('Stopped', false);
        return;
      }

      if (result?.error) {
        addAgentMessage('assistant', `❌ ${escapeHtml(result.error)}`);
      } else {
        const response = result?.text || '';
        addAgentMessage('assistant', renderMarkdown(response));
      }
      setAgentStatus('Ready', false);
    }
  } catch (err) {
    removeLastSystemMsg();
    addAgentMessage('assistant', `❌ Error: ${escapeHtml(err.message || String(err))}`);
    setAgentStatus('Error', false);
  } finally {
    if (!agentStopRequested) {
      // Clear files after successful/failed processing; keep files on user stop so they can retry.
      agentFiles = [];
      renderAgentFileChips();
    }
    agentGenerationInProgress = false;
    setAgentActionButtons(false);
  }
}

function removeLastSystemMsg() {
  const msgs = $$('#agent-messages .agent-msg.system');
  if (msgs.length > 0) msgs[msgs.length - 1].remove();
}

function escapeJs(str) {
  return str.replace(/\\/g, '\\\\').replace(/'/g, "\\'");
}

// Open file via backend (uses os.startfile on Windows)
window.openAgentFile = async function(filePath) {
  try {
    const result = await api.open_file_location(filePath);
    if (result?.error) {
      showToast(`Could not open file: ${result.error}`, 'error');
    }
  } catch (err) {
    showToast(`Error: ${err.message}`, 'error');
  }
};

// Delete a processed output file
window.deleteAgentFile = async function(filePath, btnEl) {
  if (!await showConfirm('Delete this output file?', 'Delete File')) return;
  try {
    const result = await api.delete_processed_file(filePath);
    if (result?.ok) {
      // Remove the whole assistant message bubble containing this button
      const msgBubble = btnEl.closest('.agent-msg');
      if (msgBubble) {
        msgBubble.innerHTML = '<div class="muted small">🗑 File deleted.</div>';
      }
      showToast('File deleted', 'info');
    } else {
      showToast(`Could not delete: ${result?.error || 'Unknown error'}`, 'error');
    }
  } catch (err) {
    showToast(`Error: ${err.message}`, 'error');
  }
};

// Web search toggle for agent
let agentWebSearch = false;
$('#btn-agent-web')?.addEventListener('click', async () => {
  try {
    const r = await api.toggle_web_search();
    agentWebSearch = !!r.enabled;
    const btn = $('#btn-agent-web');
    btn.classList.toggle('active', agentWebSearch);
    btn.style.background = agentWebSearch ? 'var(--accent)' : '';
    btn.style.color = agentWebSearch ? 'white' : '';
    // Keep chat button in sync
    const chatBtn = $('#btn-web-search');
    if (chatBtn) {
      chatBtn.classList.toggle('active', agentWebSearch);
      chatBtn.title = agentWebSearch ? 'Web search ON' : 'Web search OFF';
    }
    showToast(agentWebSearch ? 'Web search ON' : 'Web search OFF', 'info', 2000);
    setAgentStatus(agentWebSearch ? 'Web search ON' : 'Ready', agentWebSearch);
  } catch (err) {
    showToast(`Web search toggle failed: ${err.message}`, 'error');
  }
});

// Load instruction list on startup
refreshInstructionList();

// ═══ START ════════════════════════════════════════════════════
init().catch(e => {
  console.error('Init error:', e);
  setStatus(`Initialization error: ${e?.message || e}`);
  showToast(`Initialization error: ${e?.message || e}`, 'error', 7000);
});
