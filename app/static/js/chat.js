/**
 * Chat page JavaScript — handles sessions, messages, streaming, and UI.
 */

const API = '';
const TOKEN = localStorage.getItem('token');
if (!TOKEN) window.location.href = '/';

const headers = () => ({
  'Content-Type': 'application/json',
  'Authorization': `Bearer ${TOKEN}`,
});

// ── State ──────────────────────────────────────────────────────────
let currentSessionId = null;
let selectedKBs = [];
let sessions = [];
let kbs = [];
let currentUser = null;
let isStreaming = false;
let pendingImages = []; // {name, size, base64}
let streamCounter = 0;  // unique ID counter for streaming messages
let currentAbortController = null;

function abortCurrentStream() {
  if (currentAbortController) {
    currentAbortController.abort();
    currentAbortController = null;
  }
}

window.addEventListener('beforeunload', () => {
  abortCurrentStream();
});

// ── Init ───────────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', async () => {
  await loadUser();
  await Promise.all([loadSessions(), loadKBs(), loadLLMStatus()]);
  setupInput();
  setupImageHandlers();
});

// ── API helpers ────────────────────────────────────────────────────
async function api(path, opts = {}) {
  const res = await fetch(`${API}${path}`, { headers: headers(), ...opts });
  if (res.status === 401) { logout(); return null; }
  return res;
}
async function apiJSON(path, opts = {}) {
  const res = await api(path, opts);
  if (!res || !res.ok) return null;
  return res.json();
}

// ── User ───────────────────────────────────────────────────────────
async function loadUser() {
  currentUser = await apiJSON('/api/auth/me');
  if (!currentUser) return;
  document.getElementById('userName').textContent = currentUser.display_name || currentUser.username;
  const roleLabel = currentUser.custom_role_name
    ? currentUser.custom_role_name
    : currentUser.role.charAt(0).toUpperCase() + currentUser.role.slice(1);
  document.getElementById('userRole').textContent = roleLabel;
  document.getElementById('userAvatar').textContent = (currentUser.username || '?')[0].toUpperCase();
  if (['admin', 'superadmin'].includes(currentUser.role)) {
    document.getElementById('adminLink').classList.remove('hidden');
  }
}

function toggleUserMenu() {
  document.getElementById('userMenu').classList.toggle('hidden');
}

function logout() {
  localStorage.removeItem('token');
  window.location.href = '/';
}

function toggleSidebar() {
  const sidebar = document.getElementById('sidebar');
  const overlay = document.getElementById('sidebarOverlay');
  
  sidebar.classList.toggle('-translate-x-full');
  
  if (sidebar.classList.contains('-translate-x-full')) {
    overlay.classList.add('hidden');
  } else {
    overlay.classList.remove('hidden');
  }
}

// ── Sessions ───────────────────────────────────────────────────────
async function loadSessions() {
  sessions = await apiJSON('/api/chat/sessions') || [];
  renderSessionList();
}

function renderSessionList() {
  const el = document.getElementById('chatList');
  if (!sessions.length) {
    el.innerHTML = '<p class="text-xs text-gray-500 px-2 py-4 text-center">No chats yet</p>';
    return;
  }
  el.innerHTML = sessions.map(s => {
    const isActive = s.id === currentSessionId;
    const itemClass = isActive 
      ? 'bg-blue-600/20 border-blue-500/50 text-blue-400 shadow-sm' 
      : 'border-transparent text-gray-400 hover:bg-[#2a2a2a] hover:text-gray-200';
    const iconClass = isActive ? 'text-blue-400' : 'text-gray-500 group-hover:text-gray-400';
    
    return `
    <div class="group flex items-center gap-2 px-3 py-2 cursor-pointer rounded-xl text-sm transition-all duration-200 border ${itemClass}" onclick="loadSession('${s.id}')">
      <svg class="w-4 h-4 flex-shrink-0 ${iconClass}" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M8 10h.01M12 10h.01M16 10h.01M9 16H5a2 2 0 01-2-2V6a2 2 0 012-2h14a2 2 0 012 2v8a2 2 0 01-2 2h-5l-5 5v-5z"/></svg>
      <span class="truncate flex-1">${escapeHtml(s.title)}</span>
      <button aria-label="Delete session" onclick="event.stopPropagation(); deleteSession('${s.id}')" class="opacity-0 group-hover:opacity-100 p-1 hover:bg-red-900/30 text-gray-500 hover:text-red-400 rounded transition-colors">
        <svg class="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12"/></svg>
      </button>
    </div>
    `;
  }).join('');
}

async function newChat() {
  const res = await apiJSON('/api/chat/sessions', {
    method: 'POST',
    body: JSON.stringify({ title: 'New Chat', kb_ids: selectedKBs }),
  });
  if (res) {
    currentSessionId = res.id;
    await loadSessions();
    clearMessages();
    
    // Close sidebar on mobile
    const sidebar = document.getElementById('sidebar');
    if (sidebar && !sidebar.classList.contains('-translate-x-full') && window.innerWidth < 768) {
      toggleSidebar();
    }
  }
}

async function loadSession(sessionId) {
  abortCurrentStream();
  currentSessionId = sessionId;
  renderSessionList();
  
  try {
    const [session, messages] = await Promise.all([
      apiJSON(`/api/chat/sessions/${sessionId}`),
      apiJSON(`/api/chat/sessions/${sessionId}/messages`)
    ]);
    
    if (session) {
      selectedKBs = session.kb_ids || [];
      renderKBCheckboxes();
    }
    if (messages) {
      renderMessages(messages);
    }
  } catch (err) {
    console.error('Failed to load session:', err);
  }
  
  // Close sidebar on mobile
  const sidebar = document.getElementById('sidebar');
  if (sidebar && !sidebar.classList.contains('-translate-x-full') && window.innerWidth < 768) {
    toggleSidebar();
  }
}

async function deleteSession(sessionId) {
  if (!confirm('Delete this chat?')) return;
  if (currentSessionId === sessionId) {
    abortCurrentStream();
  }
  await api(`/api/chat/sessions/${sessionId}`, { method: 'DELETE' });
  if (currentSessionId === sessionId) {
    currentSessionId = null;
    clearMessages();
  }
  await loadSessions();
}

// ── Knowledge Bases ────────────────────────────────────────────────
async function loadKBs() {
  kbs = await apiJSON('/api/kb/') || [];
  renderKBCheckboxes();
  // Update the welcome screen if it's visible
  const welcome = document.getElementById('welcomeScreen');
  if (welcome) renderWelcomeKBs();
}

function renderKBCheckboxes() {
  const el = document.getElementById('kbCheckboxes');
  if (!kbs.length) {
    el.innerHTML = '<p class="text-xs text-gray-400 p-2">No knowledge bases</p>';
    return;
  }
  el.innerHTML = kbs.map(kb => `
    <label class="flex items-center gap-2 px-2 py-1.5 rounded hover:bg-[#2a2a2a] cursor-pointer text-sm">
      <input type="checkbox" class="kb-checkbox" value="${kb.id}" ${selectedKBs.includes(kb.id) ? 'checked' : ''} onchange="toggleKB('${kb.id}')">
      ${escapeHtml(kb.name)}
    </label>
  `).join('');
  updateKBLabel();
}

function toggleKBDropdown() {
  document.getElementById('kbDropdown').classList.toggle('hidden');
}

function toggleKB(kbId) {
  const idx = selectedKBs.indexOf(kbId);
  if (idx >= 0) selectedKBs.splice(idx, 1);
  else selectedKBs.push(kbId);
  updateKBLabel();
  // Update current session KB selection
  if (currentSessionId) {
    api(`/api/chat/sessions/${currentSessionId}`, {
      method: 'PUT',
      body: JSON.stringify({ kb_ids: selectedKBs }),
    });
  }
}

function selectKBFromWelcome(kbId) {
  const idx = selectedKBs.indexOf(kbId);
  if (idx >= 0) {
    selectedKBs.splice(idx, 1);
  } else {
    selectedKBs.push(kbId);
  }
  updateKBLabel();
  renderKBCheckboxes();
  renderWelcomeKBs();
  // Focus on input
  document.getElementById('chatInput').focus();
}

function updateKBLabel() {
  const label = document.getElementById('kbSelectedLabel');
  if (!selectedKBs.length) { label.textContent = 'Select KB'; return; }
  const names = selectedKBs.map(id => kbs.find(k => k.id === id)?.name || id);
  label.textContent = names.length <= 2 ? names.join(', ') : `${names.length} selected`;
}

// Close dropdowns when clicking outside
document.addEventListener('click', (e) => {
  const kbDd = document.getElementById('kbDropdown');
  if (!e.target.closest('#kbDropdown') && !e.target.closest('[onclick*="toggleKBDropdown"]')) {
    kbDd?.classList.add('hidden');
  }
  const sourceDd = document.getElementById('sourceDropdown');
  if (!e.target.closest('#sourceDropdown') && !e.target.closest('[onclick*="toggleSourceDropdown"]')) {
    sourceDd?.classList.add('hidden');
  }
});

function toggleSourceDropdown() {
  document.getElementById('sourceDropdown').classList.toggle('hidden');
}

function updateSourceLabel() {
  const checkboxes = document.querySelectorAll('.source-filter-checkbox:checked');
  const label = document.getElementById('sourceSelectedLabel');
  if (checkboxes.length === 0) {
    label.textContent = 'All';
  } else if (checkboxes.length === 1) {
    label.textContent = checkboxes[0].parentNode.textContent.trim();
  } else {
    label.textContent = `${checkboxes.length} selected`;
  }
}



// ── LLM Status Badge ──────────────────────────────────────────────
let _llmPollTimer = null;

async function loadLLMStatus() {
  const data = await apiJSON('/api/chat/llm-status');
  if (!data) return;

  const dot = document.getElementById('modelStatusDot');
  const label = document.getElementById('modelNameLabel');
  if (!dot || !label) return;

  const status = data.status;           // "ready" | "switching" | "error"
  const modelName = data.current_model || 'Unknown Model';

  if (status === 'switching') {
    dot.className = 'w-2 h-2 rounded-full bg-yellow-400 animate-pulse flex-shrink-0';
    label.textContent = 'Switching model…';
    // Poll more frequently while switching
    if (!_llmPollTimer) {
      _llmPollTimer = setInterval(loadLLMStatus, 10_000);
    }
  } else {
    // Stop rapid polling once settled
    if (_llmPollTimer) { clearInterval(_llmPollTimer); _llmPollTimer = null; }

    if (status === 'error') {
      dot.className = 'w-2 h-2 rounded-full bg-red-500 flex-shrink-0';
    } else {
      dot.className = 'w-2 h-2 rounded-full bg-green-500 flex-shrink-0';
    }
    label.textContent = modelName;
  }
}

// ── Messages ───────────────────────────────────────────────────────
function clearMessages() {
  const el = document.getElementById('messages');
  el.innerHTML = `<div id="welcomeScreen" class="flex flex-col items-center justify-center h-full text-center py-8 sm:py-12 px-2"></div>`;
  renderWelcomeKBs();
}

function renderWelcomeKBs() {
  const el = document.getElementById('welcomeScreen');
  if (!el) return;

  const searchHtml = kbs.length > 4 ? `
    <div class="w-full max-w-lg mb-4">
      <div class="relative">
        <svg class="absolute left-3 top-2.5 w-4 h-4 text-gray-500" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z"/></svg>
        <input type="text" id="welcomeKBSearch" placeholder="Search knowledge bases..."
               oninput="filterWelcomeKBs()"
               class="w-full pl-9 pr-4 py-2 bg-[#212121] border border-gray-700 rounded-xl text-sm text-white placeholder-gray-400 focus:outline-none focus:border-blue-500 transition">
      </div>
    </div>
  ` : '';

  let kbGridHtml = '';
  if (kbs.length === 0) {
    kbGridHtml = '<p class="text-gray-400 text-sm">No knowledge bases available. Contact your admin.</p>';
  } else {
    kbGridHtml = `<div id="welcomeKBGrid" class="grid grid-cols-1 sm:grid-cols-2 gap-3 w-full max-w-lg">${kbs.map(kb => {
      const isSelected = selectedKBs.includes(kb.id);
      return `
        <button onclick="selectKBFromWelcome('${kb.id}')"
                class="kb-welcome-card text-left px-4 py-3 rounded-xl border transition-all ${isSelected
                  ? 'bg-blue-900/30 border-blue-600 ring-1 ring-blue-500/50'
                  : 'bg-[#171717] border-gray-700 hover:border-gray-500 hover:bg-[#1c1c1c]'}"
                data-kb-name="${escapeHtml(kb.name).toLowerCase()}">
          <div class="flex items-center gap-2">
            <span class="flex-shrink-0">${isSelected ? '<svg class="w-5 h-5 text-green-400" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M5 13l4 4L19 7"/></svg>' : '<svg class="w-5 h-5 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 6.253v13m0-13C10.832 5.477 9.246 5 7.5 5S4.168 5.477 3 6.253v13C4.168 18.477 5.754 18 7.5 18s3.332.477 4.5 1.253m0-13C13.168 5.477 14.754 5 16.5 5c1.747 0 3.332.477 4.5 1.253v13C19.832 18.477 18.247 18 16.5 18c-1.746 0-3.332.477-4.5 1.253"/></svg>'}</span>
            <div class="min-w-0 flex-1">
              <p class="font-medium text-sm text-white truncate">${escapeHtml(kb.name)}</p>
              <p class="text-xs text-gray-500 truncate">${escapeHtml(kb.description || 'Knowledge base')}</p>
            </div>
          </div>
        </button>`;
    }).join('')}</div>`;
  }

  el.innerHTML = `
    <div class="mb-3"><svg class="w-12 h-12 mx-auto text-blue-400" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="1.5" d="M12 6.253v13m0-13C10.832 5.477 9.246 5 7.5 5S4.168 5.477 3 6.253v13C4.168 18.477 5.754 18 7.5 18s3.332.477 4.5 1.253m0-13C13.168 5.477 14.754 5 16.5 5c1.747 0 3.332.477 4.5 1.253v13C19.832 18.477 18.247 18 16.5 18c-1.746 0-3.332.477-4.5 1.253"/></svg></div>
    <h2 class="text-xl font-semibold text-white mb-1">Select a Knowledge Base</h2>
    <p class="text-gray-400 text-sm mb-5 max-w-md">Choose one or more knowledge bases to chat with, then type your question below.</p>
    ${searchHtml}
    ${kbGridHtml}
    ${selectedKBs.length > 0 ? `<p class="mt-4 text-xs text-blue-400">${selectedKBs.length} selected — type a question to start</p>` : ''}
  `;
}

function filterWelcomeKBs() {
  const q = (document.getElementById('welcomeKBSearch')?.value || '').toLowerCase();
  document.querySelectorAll('.kb-welcome-card').forEach(card => {
    const name = card.dataset.kbName || '';
    card.style.display = name.includes(q) ? '' : 'none';
  });
}

function renderMessages(messages) {
  const el = document.getElementById('messages');
  const welcome = document.getElementById('welcomeScreen');
  if (welcome) welcome.classList.add('hidden');

  el.innerHTML = messages.map(m => createMessageHTML(m.role, m.content, m.sources, m.images, m.id, m.version, m.parent_id, m.total_versions)).join('');
  forceScrollToBottom();
}

function buildInlineRefsHTML(sources, sid) {
  if (!sources || !sources.length) return '';
  window.responseSources = window.responseSources || {};
  window.responseSources[sid] = sources;
  const count = sources.length;
  const cards = sources.map((s, i) => {
    const kbName = s.kb_name || 'Unknown KB';
    const title = s.document_title || 'Document';
    const linkHtml = s.source_url ? `<a href="${s.source_url}" target="_blank" class="inline-flex items-center gap-1 text-blue-400 hover:underline text-xs"><svg class="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M10 6H6a2 2 0 00-2 2v10a2 2 0 002 2h10a2 2 0 002-2v-4M14 4h6m0 0v6m0-6L10 14"/></svg>Open</a>` : '';
    let tsHtml = '';
    if (s.source_type === 'youtube' && s.start_time !== undefined && s.source_url) {
      const timeUrl = s.source_url.includes('?') ? `${s.source_url}&t=${Math.floor(s.start_time)}s` : `${s.source_url}?t=${Math.floor(s.start_time)}s`;
      tsHtml = `<a href="${timeUrl}" target="_blank" class="inline-flex items-center gap-1 text-blue-400 hover:underline text-xs ml-2"><svg class="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z"/></svg>${formatTime(s.start_time)}</a>`;
    }
    return `
      <div class="bg-[#1a1a1a] rounded-lg p-3 border border-gray-700/50">
        <div class="flex items-start gap-2 mb-1">
          <span class="text-sm flex-shrink-0">${getSourceIcon(s.source_type)}</span>
          <div class="min-w-0 flex-1">
            <span class="text-xs text-gray-500 uppercase tracking-wider font-semibold">${escapeHtml(kbName)}</span>
            <p class="text-xs font-medium text-gray-300 truncate mt-0.5">${escapeHtml(title)}</p>
          </div>
        </div>
        <div class="text-xs text-gray-400 mt-2 max-h-32 overflow-y-auto pr-1 leading-relaxed" style="scrollbar-width:thin;scrollbar-color:#333 transparent;">${escapeHtml(s.content || '')}</div>
        <div class="flex items-center gap-1 mt-1.5">${linkHtml}${tsHtml}</div>
      </div>`;
  }).join('');

  // Store the formatted cards HTML globally to be used by the sidebar
  window.formattedCards = window.formattedCards || {};
  window.formattedCards[sid] = cards;

  return `
    <div class="mt-2">
      <button type="button" onclick="openRefSidebar('${sid}', ${count})" class="inline-flex items-center gap-1.5 px-3 py-1.5 text-xs text-gray-400 hover:text-gray-200 transition-colors rounded-lg hover:bg-[#2a2a2a]">
        <svg class="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 6.253v13m0-13C10.832 5.477 9.246 5 7.5 5S4.168 5.477 3 6.253v13C4.168 18.477 5.754 18 7.5 18s3.332.477 4.5 1.253m0-13C13.168 5.477 14.754 5 16.5 5c1.747 0 3.332.477 4.5 1.253v13C19.832 18.477 18.247 18 16.5 18c-1.746 0-3.332.477-4.5 1.253"/></svg>
        <span>${count} Reference Source${count > 1 ? 's' : ''}</span>
      </button>
    </div>`;
}

window.openRefSidebar = function(sid, count) {
  const sidebar = document.getElementById('refSidebar');
  const title = document.getElementById('refSidebarTitle');
  const content = document.getElementById('refSidebarContent');
  
  if (!sidebar || !content) return;
  
  title.textContent = `Reference Sources (${count})`;
  content.innerHTML = window.formattedCards[sid] || '<p class="text-gray-500 text-sm">No references found.</p>';
  
  sidebar.classList.remove('hidden');
  sidebar.classList.add('flex');
};

window.closeRefSidebar = function() {
  const sidebar = document.getElementById('refSidebar');
  if (sidebar) {
    sidebar.classList.remove('flex');
    sidebar.classList.add('hidden');
  }
};

function buildActionBar(role, rawContent, sid) {
  const isUser = role === 'user';
  const escapedContent = rawContent.replace(/\\/g, '\\\\').replace(/'/g, "\\'").replace(/"/g, '&quot;').replace(/\n/g, '\\n');
  const copyBtn = `<button onclick="copyMessage(this)" data-raw="${rawContent.replace(/"/g, '&quot;').replace(/</g, '&lt;')}" title="Copy" class="msg-action-btn p-1 rounded-md text-gray-500 hover:text-gray-200 hover:bg-[#333] transition-colors"><svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M8 16H6a2 2 0 01-2-2V6a2 2 0 012-2h8a2 2 0 012 2v2m-6 12h8a2 2 0 002-2v-8a2 2 0 00-2-2h-8a2 2 0 00-2 2v8a2 2 0 002 2z"/></svg></button>`;

  if (isUser) {
    const editBtn = `<button onclick="editMessage(this)" title="Edit" class="msg-action-btn p-1 rounded-md text-gray-500 hover:text-gray-200 hover:bg-[#333] transition-colors"><svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M11 5H6a2 2 0 00-2 2v11a2 2 0 002 2h11a2 2 0 002-2v-5m-1.414-9.414a2 2 0 112.828 2.828L11.828 15H9v-2.828l8.586-8.586z"/></svg></button>`;
    return `<div class="msg-actions flex items-center gap-0.5 mt-1 opacity-0 transition-opacity">${editBtn}${copyBtn}</div>`;
  } else {
    const regenBtn = `<button onclick="regenerateMessage(this)" title="Regenerate" class="msg-action-btn p-1 rounded-md text-gray-500 hover:text-gray-200 hover:bg-[#333] transition-colors"><svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15"/></svg></button>`;
    return `<div class="msg-actions flex items-center gap-0.5 mt-1 opacity-0 transition-opacity">${copyBtn}${regenBtn}</div>`;
  }
}

function createMessageHTML(role, content, sources, images, msgId, version, parentId, totalVersions) {
  const isUser = role === 'user';
  const sid = msgId || 'msg_' + Math.random().toString(36).substr(2, 9);
  const total = totalVersions || 1;
  const ver = version || 1;
  const hasEdits = total > 1;
  const rootId = parentId || sid;

  // Show attached images or names for user messages
  let imageHTML = '';
  if (images && images.length && isUser) {
    imageHTML = `<div class="mb-2 flex flex-wrap gap-2 justify-end">${
      images.map(img => {
        if (img && img.startsWith('data:image/')) {
          return `<div class="relative max-w-[30px] max-h-[30px] rounded-xl overflow-hidden border border-gray-700 shadow-md">
            <img src="${img}" class="object-cover w-full h-full max-h-[110px] cursor-pointer hover:opacity-90 transition" alt="Attached image" onclick="window.open('${img}', '_blank')" />
          </div>`;
        } else {
          return `<span class="inline-flex items-center gap-1 px-2 py-1 bg-[#2a2a2a] border border-gray-600 rounded-lg text-xs text-gray-300">
            <svg class="w-3 h-3 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z"/></svg>
            ${escapeHtml(img || 'image')}</span>`;
        }
      }).join('')
    }</div>`;
  }

  const refsHTML = (!isUser && sources && sources.length) ? buildInlineRefsHTML(sources, sid) : '';
  const actionBar = buildActionBar(role, content || '', sid);

  // Version navigation for edited messages
  let versionNavHTML = '';
  if (isUser && hasEdits) {
    versionNavHTML = `
      <div class="flex items-center gap-1.5 mt-1 text-xs text-gray-500" id="version-nav-${sid}">
        <button onclick="navigateVersion('${sid}', '${rootId}', -1)" class="p-0.5 rounded hover:bg-[#333] hover:text-gray-300 transition-colors">
          <svg class="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15 19l-7-7 7-7"/></svg>
        </button>
        <span class="version-label">${ver} / ${total}</span>
        <button onclick="navigateVersion('${sid}', '${rootId}', 1)" class="p-0.5 rounded hover:bg-[#333] hover:text-gray-300 transition-colors">
          <svg class="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 5l7 7-7 7"/></svg>
        </button>
      </div>`;
    // Fetch full version data for navigation
    setTimeout(() => loadVersionLabel(sid, rootId), 50);
  }

  if (isUser) {
    return `
    <div class="chat-msg-row group flex w-full justify-end" data-role="user" data-msg-id="${sid}" data-root-id="${rootId}" data-raw-content="${escapeHtml(content || '')}">
      <div class="max-w-[95%] sm:max-w-[85%] flex flex-col items-end">
        ${imageHTML}
        <div class="msg-bubble bg-[#2a2a2a] rounded-2xl rounded-tr-sm text-gray-100 shadow-sm px-5 py-3">
          <div class="msg-content text-base leading-relaxed">${formatContent(content)}</div>
        </div>
        <div class="flex items-center gap-2">
          ${actionBar}
          ${versionNavHTML}
        </div>
      </div>
    </div>`;
  } else {
    return `
    <div class="chat-msg-row group flex w-full justify-start" data-role="assistant" data-msg-id="${sid}" data-raw-content="${escapeHtml(content || '')}">
      <div class="max-w-[95%] sm:max-w-[85%]">
        <div class="text-gray-200 px-1 py-3">
          <div class="msg-content text-base leading-relaxed">${formatContent(content)}</div>
        </div>
        ${refsHTML}
        ${actionBar}
      </div>
    </div>`;
  }
}

function getSourceIcon(sourceType) {
  const cls = 'w-4 h-4 text-gray-400';
  switch (sourceType) {
    case 'youtube': return `<svg class="${cls} text-red-400" fill="currentColor" viewBox="0 0 24 24"><path d="M23.498 6.186a3.016 3.016 0 0 0-2.122-2.136C19.505 3.545 12 3.545 12 3.545s-7.505 0-9.377.505A3.017 3.017 0 0 0 .502 6.186C0 8.07 0 12 0 12s0 3.93.502 5.814a3.016 3.016 0 0 0 2.122 2.136c1.871.505 9.376.505 9.376.505s7.505 0 9.377-.505a3.015 3.015 0 0 0 2.122-2.136C24 15.93 24 12 24 12s0-3.93-.502-5.814zM9.545 15.568V8.432L15.818 12l-6.273 3.568z"/></svg>`;
    case 'pdf': return `<svg class="${cls} text-red-300" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M7 21h10a2 2 0 002-2V9.414a1 1 0 00-.293-.707l-5.414-5.414A1 1 0 0012.586 3H7a2 2 0 00-2 2v14a2 2 0 002 2z"/></svg>`;
    case 'markdown': case 'md': return `<svg class="${cls} text-blue-300" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M11 5H6a2 2 0 00-2 2v11a2 2 0 002 2h11a2 2 0 002-2v-5m-1.414-9.414a2 2 0 112.828 2.828L11.828 15H9v-2.828l8.586-8.586z"/></svg>`;
    case 'docx': case 'pptx': case 'xlsx': return `<svg class="${cls} text-blue-400" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 17v-2m3 2v-4m3 4v-6m2 10H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"/></svg>`;
    case 'html': case 'htm': return `<svg class="${cls} text-green-400" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M21 12a9 9 0 01-9 9m9-9a9 9 0 00-9-9m9 9H3m9 9a9 9 0 01-9-9m9 9c1.657 0 3-4.03 3-9s-1.343-9-3-9m0 18c-1.657 0-3-4.03-3-9s1.343-9 3-9m-9 9a9 9 0 019-9"/></svg>`;
    case 'txt': return `<svg class="${cls}" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"/></svg>`;
    case 'csv': return `<svg class="${cls} text-green-300" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M3 10h18M3 14h18m-9-4v8m-7 0h14a2 2 0 002-2V8a2 2 0 00-2-2H5a2 2 0 00-2 2v8a2 2 0 002 2z"/></svg>`;
    default: return `<svg class="${cls}" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15.172 7l-6.586 6.586a2 2 0 102.828 2.828l6.414-6.586a4 4 0 00-5.656-5.656l-6.415 6.585a6 6 0 108.486 8.486L20.5 13"/></svg>`;
  }
}

function appendAssistantMessage() {
  const el = document.getElementById('messages');
  const welcome = document.getElementById('welcomeScreen');
  if (welcome) welcome.remove();

  streamCounter++;
  const sid = streamCounter;

  const div = document.createElement('div');
  div.id = `streamingMessage-${sid}`;
  div.dataset.streamId = sid;
  div.innerHTML = `
    <div class="chat-msg-row group flex w-full justify-start" data-role="assistant">
      <div class="max-w-[95%] sm:max-w-[85%]">
        <div class="text-gray-200 px-1 py-3">
          <div class="msg-content text-base leading-relaxed" id="streamingContent-${sid}">
            <span class="typing-dot"></span>
            <span class="typing-dot"></span>
            <span class="typing-dot"></span>
          </div>
        </div>
        <div id="streamingSources-${sid}"></div>
        <div id="streamingActions-${sid}"></div>
      </div>
    </div>`;
  el.appendChild(div);
  forceScrollToBottom();
  return sid;
}

// ── Message Actions ────────────────────────────────────────────────
function copyMessage(btn) {
  const row = btn.closest('.chat-msg-row');
  const raw = row?.dataset.rawContent || row?.querySelector('.msg-content')?.innerText || '';
  navigator.clipboard.writeText(raw).then(() => {
    const origHTML = btn.innerHTML;
    btn.innerHTML = '<svg class="w-4 h-4 text-green-400" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M5 13l4 4L19 7"/></svg>';
    setTimeout(() => { btn.innerHTML = origHTML; }, 1500);
  });
}

function editMessage(btn) {
  const row = btn.closest('.chat-msg-row');
  const raw = row?.dataset.rawContent || '';
  const msgId = row?.dataset.msgId;
  const bubble = row?.querySelector('.msg-bubble');
  if (!bubble) return;

  // Replace bubble content with an editable textarea
  const contentDiv = bubble.querySelector('.msg-content');
  if (!contentDiv) return;

  // Store original content for cancel
  bubble.dataset.originalContent = raw;

  bubble.innerHTML = `
    <textarea class="edit-textarea w-full bg-transparent text-gray-100 text-base leading-relaxed resize-none outline-none" rows="3">${escapeHtml(raw)}</textarea>
    <div class="flex items-center justify-end gap-2 mt-2 border-t border-gray-700 pt-2">
      <button onclick="cancelEdit(this)" class="px-3 py-1 text-xs text-gray-400 hover:text-gray-200 rounded-lg hover:bg-[#333] transition">Cancel</button>
      <button onclick="submitEdit(this, '${msgId}')" class="px-3 py-1 text-xs bg-blue-600 hover:bg-blue-700 text-white rounded-lg transition font-medium">Save & Submit</button>
    </div>
  `;

  // Focus and auto-size the textarea
  const textarea = bubble.querySelector('.edit-textarea');
  textarea.focus();
  textarea.style.height = 'auto';
  textarea.style.height = textarea.scrollHeight + 'px';
  textarea.addEventListener('input', () => {
    textarea.style.height = 'auto';
    textarea.style.height = textarea.scrollHeight + 'px';
  });
}

function cancelEdit(btn) {
  const bubble = btn.closest('.msg-bubble');
  const row = bubble?.closest('.chat-msg-row');
  if (!bubble || !row) return;
  const raw = bubble.dataset.originalContent || row.dataset.rawContent || '';
  bubble.innerHTML = `<div class="msg-content text-base leading-relaxed">${formatContent(raw)}</div>`;
}

function getPrecedingUserRow(row) {
  const rows = Array.from(document.querySelectorAll('.chat-msg-row'));
  const index = rows.indexOf(row);
  if (index === -1) return null;
  for (let i = index - 1; i >= 0; i--) {
    if (rows[i].dataset.role === 'user') {
      return rows[i];
    }
  }
  return null;
}

async function streamResponse(message, messageId = null, images = null) {
  const sid = appendAssistantMessage();
  isStreaming = true;
  document.getElementById('sendBtn').disabled = true;

  try {
    const body = { message };
    if (messageId) {
      body.message_id = messageId;
    }
    if (images && images.length > 0) {
      body.images = images.map(img => img.base64);
    }

    const selectedSources = Array.from(document.querySelectorAll('.source-filter-checkbox:checked')).map(cb => cb.value);
    if (selectedSources.length > 0) {
      body.metadata_filters = { source_type: selectedSources };
    }

    abortCurrentStream();
    currentAbortController = new AbortController();

    const res = await fetch(`${API}/api/chat/sessions/${currentSessionId}/chat/stream`, {
      method: 'POST',
      headers: headers(),
      body: JSON.stringify(body),
      signal: currentAbortController.signal,
    });

    if (res.status === 503) {
      const errData = await res.json().catch(() => null);
      const detail = errData?.detail || 'The AI is currently undergoing a model swap. Please wait.';
      const contentEl = document.getElementById(`streamingContent-${sid}`);
      if (contentEl) contentEl.innerHTML = `<span class="inline-flex items-center gap-1.5 text-yellow-400"><svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-2.5L13.732 4.5c-.77-.833-2.694-.833-3.464 0L3.34 16.5c-.77.833.192 2.5 1.732 2.5z"/></svg>${escapeHtml(detail)}</span>`;
      loadLLMStatus();
      return;
    }

    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let fullText = '';
    const contentEl = document.getElementById(`streamingContent-${sid}`);

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      const chunk = decoder.decode(value, { stream: true });
      const lines = chunk.split('\n');

      for (const line of lines) {
        if (!line.startsWith('data: ')) continue;
        try {
          const data = JSON.parse(line.slice(6));
          if (data.token) {
            fullText += data.token;
            if (contentEl) contentEl.innerHTML = formatContent(fullText);
            scrollToBottom();
          }
          if (data.done && data.sources) {
            const srcEl = document.getElementById(`streamingSources-${sid}`);
            if (srcEl) {
              srcEl.innerHTML = buildInlineRefsHTML(data.sources, sid);
            }
            const actEl = document.getElementById(`streamingActions-${sid}`);
            const msgRow = document.getElementById(`streamingMessage-${sid}`)?.querySelector('.chat-msg-row');
            if (actEl && msgRow) {
              msgRow.dataset.rawContent = fullText;
              actEl.innerHTML = buildActionBar('assistant', fullText, sid);
            }
          }
        } catch {}
      }
    }
  } catch (err) {
    if (err.name === 'AbortError') {
      console.log('Stream aborted');
      const contentEl = document.getElementById(`streamingContent-${sid}`);
      if (contentEl) {
        contentEl.innerHTML = '<span class="text-gray-500 italic">Generation stopped.</span>';
      }
      return;
    }
    console.error('Stream error:', err);
    const contentEl = document.getElementById(`streamingContent-${sid}`);
    if (contentEl) {
      const escapedMsg = message.replace(/&/g, '&amp;').replace(/"/g, '&quot;').replace(/'/g, '&#39;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
      contentEl.innerHTML = `
        <span class="text-red-400">Error generating response.</span>
        <button onclick="retryMessage(this)" data-msg="${escapedMsg}" class="ml-2 px-3 py-1 bg-red-500/20 text-red-400 hover:bg-red-500/30 rounded-lg text-xs transition-colors">
          Regenerate
        </button>
      `;
    }
  } finally {
    isStreaming = false;
    document.getElementById('sendBtn').disabled = false;
    currentAbortController = null;
    const streamMsg = document.getElementById(`streamingMessage-${sid}`);
    if (streamMsg) streamMsg.removeAttribute('id');
    if (currentSessionId) {
      const messages = await apiJSON(`/api/chat/sessions/${currentSessionId}/messages`);
      if (messages) renderMessages(messages);
    }
    await loadSessions(); // refresh titles
  }
}

async function submitEdit(btn, msgId) {
  if (isStreaming || !currentSessionId) return;
  const bubble = btn.closest('.msg-bubble');
  const row = bubble?.closest('.chat-msg-row');
  if (!bubble || !row) return;

  const textarea = bubble.querySelector('.edit-textarea');
  const newContent = textarea?.value?.trim();
  if (!newContent) return;

  // Disable the button
  btn.disabled = true;
  btn.textContent = 'Saving...';

  try {
    // Call edit API
    const result = await apiJSON(`/api/chat/sessions/${currentSessionId}/messages/${msgId}/edit`, {
      method: 'PUT',
      body: JSON.stringify({ content: newContent }),
    });

    if (!result) {
      cancelEdit(btn);
      return;
    }

    // Reload all messages to get the correct state
    const messages = await apiJSON(`/api/chat/sessions/${currentSessionId}/messages`);
    if (messages) renderMessages(messages);

    // Stream response for the edited message
    await streamResponse(newContent, result.id);
  } catch (err) {
    console.error('Edit failed:', err);
    cancelEdit(btn);
  }
}

async function loadVersionLabel(sid, rootId) {
  if (!currentSessionId) return;
  try {
    const versions = await apiJSON(`/api/chat/sessions/${currentSessionId}/messages/${rootId}/versions`);
    if (!versions || !versions.length) return;
    const navEl = document.getElementById(`version-nav-${sid}`);
    if (!navEl) return;
    const currentVersion = versions.find(v => v.is_active);
    const label = navEl.querySelector('.version-label');
    if (label && currentVersion) {
      label.textContent = `${currentVersion.version} / ${versions.length}`;
      label.dataset.version = currentVersion.version;
      label.dataset.total = versions.length;
    }
    // Store versions data on the nav element
    navEl.dataset.versions = JSON.stringify(versions);
  } catch {}
}

async function navigateVersion(sid, rootId, direction) {
  if (!currentSessionId) return;
  const navEl = document.getElementById(`version-nav-${sid}`);
  if (!navEl) return;
  
  let versions;
  try {
    versions = JSON.parse(navEl.dataset.versions || '[]');
  } catch { return; }
  if (!versions.length) return;

  const activeIdx = versions.findIndex(v => v.is_active);
  const newIdx = activeIdx + direction;
  if (newIdx < 0 || newIdx >= versions.length) return;

  const targetVersion = versions[newIdx];

  // Switch to this version without creating a new one
  try {
    await apiJSON(`/api/chat/sessions/${currentSessionId}/messages/${targetVersion.id}/switch`, {
      method: 'PUT',
    });
    // Reload the full conversation
    const messages = await apiJSON(`/api/chat/sessions/${currentSessionId}/messages`);
    if (messages) renderMessages(messages);
  } catch (err) {
    console.error('Version navigation failed:', err);
  }
}

async function regenerateMessage(btn) {
  if (isStreaming || !currentSessionId) return;
  const row = btn.closest('.chat-msg-row');
  if (!row) return;

  const userRow = getPrecedingUserRow(row);
  const userContent = userRow?.dataset?.rawContent || '';
  const userMsgId = userRow?.dataset?.msgId;
  if (!userContent || !userMsgId) return;

  btn.disabled = true;

  try {
    // Call the edit API with the same content to spawn a new version
    const result = await apiJSON(`/api/chat/sessions/${currentSessionId}/messages/${userMsgId}/edit`, {
      method: 'PUT',
      body: JSON.stringify({ content: userContent }),
    });

    if (!result) {
      btn.disabled = false;
      return;
    }

    // Reload all messages to show the new user message version
    const messages = await apiJSON(`/api/chat/sessions/${currentSessionId}/messages`);
    if (messages) renderMessages(messages);

    // Stream response for the new user message version
    await streamResponse(userContent, result.id);
  } catch (err) {
    console.error('Regeneration failed:', err);
    btn.disabled = false;
  }
}


// ── Image Upload & Handling ─────────────────────────────────────
function clearImageUpload() {
  pendingImages = [];
  document.getElementById('imagePreview').classList.add('hidden');
  document.getElementById('imageFileList').innerHTML = '';
  document.getElementById('imageInput').value = '';
}

function removeImage(index) {
  pendingImages.splice(index, 1);
  if (pendingImages.length === 0) {
    clearImageUpload();
  } else {
    renderImageFileList();
  }
}

function renderImageFileList() {
  const listEl = document.getElementById('imageFileList');
  const previewEl = document.getElementById('imagePreview');

  if (pendingImages.length === 0) {
    previewEl.classList.add('hidden');
    listEl.innerHTML = '';
    return;
  }

  previewEl.classList.remove('hidden');
  listEl.innerHTML = pendingImages.map((img, i) => `
    <div class="flex items-center gap-2 py-1">
      <svg class="w-4 h-4 text-gray-400 flex-shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z"/>
      </svg>
      <span class="text-sm text-white truncate flex-1">${escapeHtml(img.name)}</span>
      <span class="text-xs text-gray-400 flex-shrink-0">${formatFileSize(img.size)}</span>
      <button type="button" aria-label="Remove image" onclick="removeImage(${i})" class="flex-shrink-0 w-5 h-5 hover:bg-[#3a3a3a] rounded-full flex items-center justify-center">
        <svg class="w-3 h-3 text-gray-400 hover:text-red-400" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12"/></svg>
      </button>
    </div>
  `).join('');
}

function handleImageFile(file) {
  if (!file || !file.type.startsWith('image/')) return;
  const reader = new FileReader();
  reader.onload = (e) => {
    pendingImages.push({
      name: file.name,
      size: file.size,
      base64: e.target.result,
    });
    renderImageFileList();
  };
  reader.readAsDataURL(file);
}

function handleImageFiles(files) {
  for (const file of files) {
    if (file.type.startsWith('image/')) {
      handleImageFile(file);
    }
  }
}

function setupImageHandlers() {
  const imageInput = document.getElementById('imageInput');
  const dropZone = document.getElementById('dropZone');
  const dropOverlay = document.getElementById('dropOverlay');
  const chatInput = document.getElementById('chatInput');

  // File input selection (multiple)
  imageInput.addEventListener('change', () => {
    handleImageFiles(imageInput.files);
    imageInput.value = '';
  });

  // Drag & drop on the input area
  if (dropZone) {
    dropZone.addEventListener('dragover', (e) => {
      e.preventDefault();
      e.stopPropagation();
      if (dropOverlay) dropOverlay.classList.remove('hidden');
    });

    dropZone.addEventListener('dragleave', (e) => {
      e.preventDefault();
      e.stopPropagation();
      if (dropOverlay) dropOverlay.classList.add('hidden');
    });

    dropZone.addEventListener('drop', (e) => {
      e.preventDefault();
      e.stopPropagation();
      if (dropOverlay) dropOverlay.classList.add('hidden');
      handleImageFiles(e.dataTransfer.files);
    });
  }

  // Paste image from clipboard
  chatInput.addEventListener('paste', (e) => {
    const items = e.clipboardData?.items;
    if (!items) return;
    for (const item of items) {
      if (item.type.startsWith('image/')) {
        e.preventDefault();
        const file = item.getAsFile();
        if (file) handleImageFile(file);
        break;
      }
    }
  });
}

// ── Chat submission ────────────────────────────────────────────────
function setupInput() {
  const input = document.getElementById('chatInput');
  const form = document.getElementById('chatForm');

  // Auto-resize textarea
  input.addEventListener('input', () => {
    input.style.height = 'auto';
    input.style.height = Math.min(input.scrollHeight, 200) + 'px';
    scrollToBottom();
  });

  // Submit on Enter (Shift+Enter for newline)
  input.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  });

  // Also handle native form submit (button click)
  form.addEventListener('submit', (e) => {
    e.preventDefault();
    sendMessage();
  });
}

async function sendMessage() {
  if (isStreaming) return;

  const input = document.getElementById('chatInput');
  const message = input.value.trim();
  if (!message) return;

  // Create session if none selected
  if (!currentSessionId) {
    const res = await apiJSON('/api/chat/sessions', {
      method: 'POST',
      body: JSON.stringify({ title: message.slice(0, 100), kb_ids: selectedKBs }),
    });
    if (!res) return;
    currentSessionId = res.id;
    await loadSessions();
  }

  // Show user message
  const msgsEl = document.getElementById('messages');
  const welcome = document.getElementById('welcomeScreen');
  if (welcome) welcome.remove();

  // Capture and clear images
  const imagesCopy = [...pendingImages];
  clearImageUpload();

  msgsEl.innerHTML += createMessageHTML('user', message, null, imagesCopy.map(img => img.base64));
  input.value = '';
  input.style.height = 'auto';

  // Stream the response using unified helper
  await streamResponse(message, null, imagesCopy);
}

// ── Utilities ──────────────────────────────────────────────────────
function scrollToBottom() {
  const container = document.getElementById('messagesContainer');
  const isNearBottom = container.scrollHeight - container.scrollTop - container.clientHeight < 100;
  if (isNearBottom) {
    container.scrollTop = container.scrollHeight;
  }
}

function forceScrollToBottom() {
  const container = document.getElementById('messagesContainer');
  container.scrollTop = container.scrollHeight;
}

function retryMessage(btn) {
  const input = document.getElementById('chatInput');
  input.value = btn.dataset.msg;
  // Remove the failed assistant message and the original user message
  const assistantContainer = btn.closest('.flex.gap-3');
  if (assistantContainer && assistantContainer.previousElementSibling) {
    assistantContainer.previousElementSibling.remove();
  }
  if (assistantContainer) assistantContainer.remove();
  
  sendMessage();
}

function escapeHtml(str) {
  if (!str) return '';
  const div = document.createElement('div');
  div.textContent = str;
  return div.innerHTML;
}

function formatContent(text) {
  if (!text) return '';
  // Simple Markdown-ish rendering
  let html = escapeHtml(text);
  // Code blocks
  html = html.replace(/```(\w*)\n([\s\S]*?)```/g, '<pre><code>$2</code></pre>');
  // Inline code
  html = html.replace(/`([^`]+)`/g, '<code>$1</code>');
  // Bold
  html = html.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');
  // Italic
  html = html.replace(/\*(.+?)\*/g, '<em>$1</em>');
  // Headers
  html = html.replace(/^### (.+)$/gm, '<h3>$1</h3>');
  html = html.replace(/^## (.+)$/gm, '<h2>$1</h2>');
  html = html.replace(/^# (.+)$/gm, '<h1>$1</h1>');
  // Line breaks
  html = html.replace(/\n/g, '<br>');
  return html;
}

function formatFileSize(bytes) {
  if (bytes < 1024) return bytes + ' B';
  if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB';
  return (bytes / (1024 * 1024)).toFixed(1) + ' MB';
}

function formatTime(seconds) {
  const m = Math.floor(seconds / 60);
  const s = Math.floor(seconds % 60);
  return `${m}:${s.toString().padStart(2, '0')}`;
}
