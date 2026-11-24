/**
 * Admin panel JavaScript — KB management, model registry, roles, config editor,
 * document management with pagination/search/multi-upload/index/delete.
 */

const API = '';
const TOKEN = localStorage.getItem('token');
if (!TOKEN) window.location.href = '/';

const headers = () => ({
  'Content-Type': 'application/json',
  'Authorization': `Bearer ${TOKEN}`,
});

async function api(path, opts = {}) {
  const res = await fetch(`${API}${path}`, { headers: headers(), ...opts });
  if (res.status === 401) { logout(); return null; }
  if (res.status === 403) { showAlert('Insufficient permissions'); return null; }
  return res;
}
async function apiJSON(path, opts = {}) {
  const res = await api(path, opts);
  if (!res || !res.ok) return null;
  return res.json();
}

function logout() {
  localStorage.removeItem('token');
  window.location.href = '/';
}

// ── Init ───────────────────────────────────────────────────────────
let currentUser = null;

document.addEventListener('DOMContentLoaded', async () => {
  const user = await apiJSON('/api/auth/me');
  if (!user || !['admin', 'superadmin'].includes(user.role)) {
    showAlert('Admin access required');
    window.location.href = '/chat';
    return;
  }
  currentUser = user;
  document.getElementById('adminUser').textContent = user.username;
  document.getElementById('adminAvatar').textContent = (user.username || '?')[0].toUpperCase();
  await Promise.all([loadKBs(), loadModels(), loadLLMSwap(), loadRoles(), loadConfig(), loadUsers()]);
  if (user.role === 'superadmin') {
    loadSystemStats();
  } else {
    const utilsTab = document.querySelector('button[data-tab="utilities"]');
    if (utilsTab) utilsTab.style.display = 'none';
  }
  setupUploadDragDrop();
});

// ── Tabs ───────────────────────────────────────────────────────────
function showTab(tab) {
  document.querySelectorAll('.tab-content').forEach(el => el.classList.add('hidden'));
  
  document.querySelectorAll('.tab-btn').forEach(el => {
    el.setAttribute('aria-selected', 'false');
    el.setAttribute('tabindex', '-1');
    el.classList.remove('border-blue-500', 'text-blue-400');
    el.classList.add('border-transparent', 'text-gray-500');
  });
  
  const activePanel = document.getElementById(`tab-${tab}`);
  if (activePanel) activePanel.classList.remove('hidden');
  
  const activeBtn = document.querySelector(`[data-tab="${tab}"]`);
  if (activeBtn) {
    activeBtn.setAttribute('aria-selected', 'true');
    activeBtn.setAttribute('tabindex', '0');
    activeBtn.classList.remove('border-transparent', 'text-gray-500');
    activeBtn.classList.add('border-blue-500', 'text-blue-400');
  }
}

function showModal(id) { document.getElementById(id).classList.remove('hidden'); }
function hideModal(id) { document.getElementById(id).classList.add('hidden'); }

let focusBeforeAlert = null;

function showAlert(msg) {
  const modal = document.getElementById('alertModal');
  const panel = document.getElementById('alertModalPanel');
  const msgEl = document.getElementById('alertModalMessage');
  const btn = document.getElementById('alertModalBtn');
  
  if (modal && msgEl && panel) {
    focusBeforeAlert = document.activeElement;
    msgEl.textContent = msg;
    
    modal.classList.remove('hidden');
    // Force reflow
    void modal.offsetWidth;
    
    modal.classList.remove('opacity-0');
    panel.classList.remove('scale-95', 'opacity-0');
    panel.classList.add('scale-100', 'opacity-100');
    
    setTimeout(() => { if (btn) btn.focus(); }, 100);
  } else {
    window.alert(msg); // fallback
  }
}

function hideAlertModal() {
  const modal = document.getElementById('alertModal');
  const panel = document.getElementById('alertModalPanel');
  
  if (modal && panel) {
    modal.classList.add('opacity-0');
    panel.classList.remove('scale-100', 'opacity-100');
    panel.classList.add('scale-95', 'opacity-0');
    
    setTimeout(() => {
      modal.classList.add('hidden');
      if (focusBeforeAlert) focusBeforeAlert.focus();
    }, 300);
  } else if (modal) {
    hideModal('alertModal');
  }
}

// ── Knowledge Bases ────────────────────────────────────────────────
let kbs = [];

async function loadKBs() {
  // Show loading state in the Import KB dropdown
  const importSel = document.getElementById('importKBSelect');
  const importBtn = document.getElementById('importBtn');
  if (importSel) {
    importSel.innerHTML = '<option value="" disabled selected>Loading knowledge bases…</option>';
    importSel.disabled = true;
  }
  if (importBtn) importBtn.disabled = true;

  try {
    const data = await apiJSON('/api/kb/');
    if (!data) throw new Error('Failed to fetch knowledge bases');
    kbs = data;
  } catch (err) {
    // Error toast + keep upload disabled
    showToast('Could not load knowledge bases. Please refresh the page.', 'error');
    if (importSel) {
      importSel.innerHTML = '<option value="" disabled selected>Failed to load — refresh</option>';
      importSel.disabled = false;
    }
    return;
  }

  renderKBs();

  if (importSel) {
    importSel.disabled = false;
    if (kbs.length) {
      importSel.innerHTML = kbs.map(kb => `<option value="${kb.id}">${escapeHtml(kb.name)}</option>`).join('');
      if (importBtn) importBtn.disabled = false;
    } else {
      importSel.innerHTML = '<option value="" disabled selected>No knowledge bases found</option>';
    }
  }
}

function renderKBs() {
  const el = document.getElementById('kbList');
  if (!kbs.length) {
    el.innerHTML = '<p class="text-gray-500">No knowledge bases yet.</p>';
    return;
  }
  el.innerHTML = kbs.map(kb => `
    <div class="bg-[#171717] border border-gray-800 rounded-xl p-4 flex flex-col sm:flex-row items-start sm:items-center justify-between gap-3">
      <div>
        <h3 class="font-medium">${escapeHtml(kb.name)}</h3>
        <p class="text-sm text-gray-400">${escapeHtml(kb.description || 'No description')}</p>
        <div class="flex gap-4 mt-2 text-xs text-gray-500">
          <span>Collection: ${kb.collection_name || '—'}</span>
          <span>Dim: ${kb.vector_dimension || '—'}</span>
        </div>
      </div>
      <div class="flex items-center gap-2">
        <button onclick="viewKBDocs('${kb.id}')" class="px-3 py-1 bg-gray-700 hover:bg-gray-600 rounded-lg text-xs transition">Documents</button>
        <button onclick="deleteKB('${kb.id}')" class="px-3 py-1 bg-red-900 hover:bg-red-800 rounded-lg text-xs transition">Delete</button>
      </div>
    </div>
  `).join('');
}

function showCreateKBModal() { showModal('createKBModal'); }

async function createKB() {
  const name = document.getElementById('newKBName').value.trim();
  if (!name) return;
  const desc = document.getElementById('newKBDesc').value.trim();
  await apiJSON('/api/kb/', {
    method: 'POST',
    body: JSON.stringify({ name, description: desc || null }),
  });
  hideModal('createKBModal');
  document.getElementById('newKBName').value = '';
  document.getElementById('newKBDesc').value = '';
  await loadKBs();
}

async function deleteKB(id) {
  if (!confirm('Delete this knowledge base and all its documents? This cannot be undone.')) return;
  await api(`/api/kb/${id}`, { method: 'DELETE' });
  await loadKBs();
}

// ── Documents ──────────────────────────────────────────────────────
let docPage = 1;
const DOC_LIMIT = 100;
let docData = { documents: [], total: 0, page: 1, total_pages: 1 };
let selectedDocs = new Map();
let currentDocKBId = null;

function viewKBDocs(kbId) {
  currentDocKBId = kbId;
  const kb = kbs.find(k => k.id === kbId);
  docPage = 1;
  selectedDocs.clear();

  const el = document.getElementById('kbList');
  el.innerHTML = `
    <div class="flex flex-col sm:flex-row items-start sm:items-center justify-between mb-4 gap-3">
      <div class="flex items-center gap-3">
        <button onclick="renderKBs()" class="px-3 py-1.5 bg-gray-700 hover:bg-gray-600 rounded-lg text-sm transition">← Back</button>
        <h2 class="text-xl font-semibold">${escapeHtml(kb?.name || '')} — Documents</h2>
      </div>
      <div class="flex items-center gap-2 flex-wrap">
        <button onclick="loadDocuments()" class="px-4 py-2 bg-gray-700 hover:bg-gray-600 rounded-xl text-sm font-medium transition">
          Refresh
        </button>
        <button onclick="deleteSelectedDocs()" id="deleteSelectedBtn"
                class="px-4 py-2 bg-red-700 hover:bg-red-800 rounded-xl text-sm font-medium transition disabled:opacity-50 disabled:cursor-not-allowed" disabled>
          Delete Selected
        </button>
        <button onclick="indexSelectedDocs()" id="indexSelectedBtn"
                class="px-4 py-2 bg-yellow-600 hover:bg-yellow-700 rounded-xl text-sm font-medium transition disabled:opacity-50 disabled:cursor-not-allowed" disabled>
          Index Selected Vectors
        </button>
        <button onclick="showMultiUploadModal()" class="px-4 py-2 bg-blue-600 hover:bg-blue-700 rounded-xl text-sm font-medium transition">+ Upload Files</button>
      </div>
    </div>
    <div class="flex flex-col sm:flex-row items-start sm:items-center gap-3 mb-4">
      <div class="flex-1 relative">
        <input type="text" id="docSearch" placeholder="Search by filename..."
               oninput="debouncedSearchDocs()"
               class="w-full px-4 py-2 bg-[#171717] border border-gray-700 rounded-lg text-sm focus:outline-none focus:border-blue-500 pl-9">
        <svg class="absolute left-3 top-2.5 w-4 h-4 text-gray-500" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z"/></svg>
      </div>
      <select id="docStatusFilter" onchange="loadDocuments()"
              class="bg-[#171717] border border-gray-700 rounded-lg px-3 py-2 text-sm focus:outline-none focus:border-blue-500">
        <option value="">All Statuses</option>
        <option value="pending">Pending</option>
        <option value="chunked">Chunked</option>
        <option value="indexing">Indexing</option>
        <option value="processed">Processed</option>
        <option value="error">Error</option>
      </select>
      <select id="docSourceFilter" onchange="loadDocuments()"
              class="bg-[#171717] border border-gray-700 rounded-lg px-3 py-2 text-sm focus:outline-none focus:border-blue-500">
        <option value="">All Sources</option>
        <option value="youtube">YouTube</option>
        <option value="upload">Upload</option>
      </select>
    </div>
    <div class="flex items-center justify-between mb-3 text-sm text-gray-400">
      <div class="flex items-center gap-4">
        <label class="flex items-center gap-2 cursor-pointer">
          <input type="checkbox" id="selectAllDocs" onchange="toggleSelectAll()" class="rounded">
          <span>Select all on page</span>
        </label>
        <div class="h-4 w-px bg-gray-700"></div>
        <button onclick="deleteAllErrorDocs()" id="deleteAllErrorBtn" class="text-red-400 hover:text-red-300 transition text-xs font-medium">Delete All Errors</button>
        <button onclick="indexAllChunkedDocs()" id="indexAllChunkedBtn" class="text-yellow-400 hover:text-yellow-300 transition text-xs font-medium">Index All Chunked</button>
      </div>
      <div id="docPageInfo">Loading...</div>
    </div>
    <div id="documentList" class="space-y-2 mb-4"><p class="text-gray-500">Loading...</p></div>
    <div id="docPagination" class="flex items-center justify-center gap-2"></div>
  `;

  loadDocuments();
}

async function deleteAllErrorDocs() {
  if (!confirm('Are you sure you want to delete ALL documents with "Error" status in this KB?')) return;
  const btn = document.getElementById('deleteAllErrorBtn');
  btn.textContent = 'Deleting...';
  btn.disabled = true;

  try {
    // Paginate through all error docs (API max is 1000 per page)
    let allIds = [];
    let page = 1;
    while (true) {
      const data = await apiJSON(`/api/documents/kb/${currentDocKBId}?page=${page}&limit=1000&status=error`);
      if (!data || !data.documents || data.documents.length === 0) break;
      allIds.push(...data.documents.map(d => d.id));
      if (page >= data.total_pages) break;
      page++;
    }

    if (allIds.length > 0) {
      await apiJSON('/api/documents/delete-batch', {
        method: 'POST',
        body: JSON.stringify({ doc_ids: allIds }),
      });
      showAlert(`Deleted ${allIds.length} error documents.`, 'success');
    } else {
      showAlert('No error documents found to delete.', 'info');
    }
  } catch (e) {
    showAlert('Failed to delete error documents.', 'error');
  }

  btn.textContent = 'Delete All Errors';
  btn.disabled = false;
  selectedDocs.clear();
  loadDocuments();
}

async function indexAllChunkedDocs() {
  if (!confirm('Are you sure you want to index ALL "Chunked" documents in this KB? This may take some time.')) return;
  const btn = document.getElementById('indexAllChunkedBtn');
  btn.textContent = 'Indexing...';
  btn.disabled = true;

  try {
    // Paginate through all chunked docs (API max is 1000 per page)
    let allDocs = [];
    let page = 1;
    while (true) {
      const data = await apiJSON(`/api/documents/kb/${currentDocKBId}?page=${page}&limit=1000&status=chunked`);
      if (!data || !data.documents || data.documents.length === 0) break;
      allDocs.push(...data.documents);
      if (page >= data.total_pages) break;
      page++;
    }

    if (allDocs.length > 0) {
      const byFolder = {};
      allDocs.forEach(d => {
        const fid = d.folder_id;
        if (!byFolder[fid]) byFolder[fid] = [];
        byFolder[fid].push(d.id);
      });

      let totalIndexed = 0;
      let totalErrors = 0;

      for (const [folderId, docIds] of Object.entries(byFolder)) {
        const result = await apiJSON(`/api/documents/index/${folderId}`, {
          method: 'POST',
          body: JSON.stringify({ doc_ids: docIds }),
        });
        if (result && result.results) {
          result.results.forEach(r => {
            if (r.status === 'success') totalIndexed++;
            else totalErrors++;
          });
        }
      }
      showAlert(`Indexed ${totalIndexed} documents successfully. Errors: ${totalErrors}`, 'success');
    } else {
      showAlert('No chunked documents found to index.', 'info');
    }
  } catch (e) {
    showAlert('Failed to index chunked documents.', 'error');
  }

  btn.textContent = 'Index All Chunked';
  btn.disabled = false;
  selectedDocs.clear();
  loadDocuments();
}

function clearDocSelection() {
  selectedDocs.clear();
  renderDocuments();
  updateIndexButton();
}

let _searchTimeout;
function debouncedSearchDocs() {
  clearTimeout(_searchTimeout);
  _searchTimeout = setTimeout(() => { docPage = 1; loadDocuments(); }, 300);
}

async function loadDocuments() {
  if (!currentDocKBId) return;
  const search = document.getElementById('docSearch')?.value || '';
  const status = document.getElementById('docStatusFilter')?.value || '';
  const source = document.getElementById('docSourceFilter')?.value || '';

  const data = await apiJSON(`/api/documents/kb/${currentDocKBId}?page=${docPage}&limit=${DOC_LIMIT}&search=${encodeURIComponent(search)}&status=${encodeURIComponent(status)}&source_type=${encodeURIComponent(source)}`);
  if (data) {
    docData = data;
  }
  renderDocuments();
  updateIndexButton();
}

function renderDocuments() {
  const el = document.getElementById('documentList');
  const info = document.getElementById('docPageInfo');
  const selAll = document.getElementById('selectAllDocs');
  if (!el || !info) return;
  // Sync selectAll checkbox state with current page docs
  if (selAll) {
    const allPageSelected = docData.documents.length > 0 && docData.documents.every(d => selectedDocs.has(d.id));
    selAll.checked = allPageSelected;
  }

  if (selectedDocs.size > 0) {
    info.innerHTML = `Showing ${docData.documents.length} of ${docData.total} documents — Page ${docData.page}/${docData.total_pages}
      <div class="mt-1 text-blue-400 font-medium">
        ${selectedDocs.size} selected on this page
        <button onclick="clearDocSelection()" class="underline hover:text-blue-300 ml-2 text-xs">Clear</button>
      </div>`;
  } else {
    info.textContent = `Showing ${docData.documents.length} of ${docData.total} documents — Page ${docData.page}/${docData.total_pages}`;
  }

  if (!docData.documents.length) {
    el.innerHTML = '<div class="text-center text-gray-500 py-8">No documents found.</div>';
    renderPagination();
    return;
  }

  el.innerHTML = docData.documents.map(d => {
    const statusColor = {
      processed: 'bg-green-900/40 text-green-400 border-green-800',
      chunked: 'bg-blue-900/40 text-blue-400 border-blue-800',
      indexing: 'bg-yellow-900/40 text-yellow-400 border-yellow-800',
      pending: 'bg-gray-800/40 text-gray-400 border-gray-700',
      processing: 'bg-yellow-900/40 text-yellow-400 border-yellow-800',
      error: 'bg-red-900/40 text-red-400 border-red-800',
    }[d.status] || 'bg-gray-800 text-gray-400 border-gray-700';

    const canIndex = d.status === 'chunked' || d.status === 'error' || d.status === 'pending';
    const fileSize = d.file_size ? formatBytes(d.file_size) : '';

    return `
    <div class="bg-[#171717] border border-gray-800 rounded-lg px-3 sm:px-4 py-3 flex flex-col sm:flex-row sm:items-center gap-2 sm:gap-3 hover:border-gray-700 transition">
      <input type="checkbox" class="doc-checkbox rounded flex-shrink-0 self-start sm:self-center" value="${d.id}" data-folder="${d.folder_id}"
             data-can-index="${canIndex}" ${selectedDocs.has(d.id) ? 'checked' : ''}
             onchange="toggleDocSelect('${d.id}', '${d.folder_id}')">
      <div class="flex-1 min-w-0">
        <div class="flex items-center gap-2">
          <span class="text-sm font-medium truncate">${escapeHtml(d.title)}</span>
          <span class="text-xs text-gray-600 uppercase flex-shrink-0">${d.file_type}</span>
          ${d.source_type && d.source_type !== 'upload' ? `<span class="text-xs px-1.5 py-0.5 bg-purple-900/30 border border-purple-800 rounded text-purple-300 flex-shrink-0">${d.source_type}</span>` : ''}
        </div>
        <div class="flex items-center gap-3 mt-1 text-xs text-gray-500">
          <span>${fileSize}</span>
          <span>${d.chunk_count} chunks</span>
          ${d.error_message ? `<span class="text-red-500 truncate max-w-[300px]" title="${escapeHtml(d.error_message)}">${escapeHtml(d.error_message)}</span>` : ''}
        </div>
      </div>
      <div class="flex items-center gap-2 flex-shrink-0 self-end sm:self-center">
        <span class="px-2 py-0.5 rounded border text-xs ${statusColor}">${d.status}</span>
        <button onclick="deleteDoc('${d.id}')" class="p-1.5 text-gray-500 hover:text-red-400 hover:bg-red-900/30 rounded-lg transition" title="Delete document">
          <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16"/></svg>
        </button>
      </div>
    </div>`;
  }).join('');

  renderPagination();
}

function renderPagination() {
  const el = document.getElementById('docPagination');
  if (docData.total_pages <= 1) { el.innerHTML = ''; return; }

  let html = '';
  const p = docData.page;
  const tp = docData.total_pages;

  html += `<button onclick="goDocPage(1)" class="px-3 py-1 rounded-lg text-xs ${p === 1 ? 'bg-gray-800 text-gray-600' : 'bg-gray-700 hover:bg-gray-600'}" ${p === 1 ? 'disabled' : ''}>«</button>`;
  html += `<button onclick="goDocPage(${p - 1})" class="px-3 py-1 rounded-lg text-xs ${p === 1 ? 'bg-gray-800 text-gray-600' : 'bg-gray-700 hover:bg-gray-600'}" ${p === 1 ? 'disabled' : ''}>‹</button>`;

  // Page numbers (show max 7 pages around current)
  let start = Math.max(1, p - 3);
  let end = Math.min(tp, p + 3);
  if (start > 1) html += `<span class="text-gray-600 text-xs px-1">...</span>`;
  for (let i = start; i <= end; i++) {
    html += `<button onclick="goDocPage(${i})" class="px-3 py-1 rounded-lg text-xs ${i === p ? 'bg-blue-600 text-white' : 'bg-gray-700 hover:bg-gray-600'}">${i}</button>`;
  }
  if (end < tp) html += `<span class="text-gray-600 text-xs px-1">...</span>`;

  html += `<button onclick="goDocPage(${p + 1})" class="px-3 py-1 rounded-lg text-xs ${p === tp ? 'bg-gray-800 text-gray-600' : 'bg-gray-700 hover:bg-gray-600'}" ${p === tp ? 'disabled' : ''}>›</button>`;
  html += `<button onclick="goDocPage(${tp})" class="px-3 py-1 rounded-lg text-xs ${p === tp ? 'bg-gray-800 text-gray-600' : 'bg-gray-700 hover:bg-gray-600'}" ${p === tp ? 'disabled' : ''}>»</button>`;

  el.innerHTML = html;
}

function goDocPage(page) {
  docPage = page;
  loadDocuments();
}

function toggleDocSelect(docId, folderId, canIndex) {
  if (selectedDocs.has(docId)) {
    selectedDocs.delete(docId);
  } else {
    selectedDocs.set(docId, { folderId, canIndex });
  }
  updateIndexButton();
}

function toggleSelectAll() {
  const checked = document.getElementById('selectAllDocs').checked;
  document.querySelectorAll('.doc-checkbox').forEach(cb => {
    cb.checked = checked;
    if (checked) selectedDocs.set(cb.value, { folderId: cb.dataset.folder, canIndex: cb.dataset.canIndex === 'true' });
    else selectedDocs.delete(cb.value);
  });
  updateIndexButton();
  renderDocuments(); // to update the banner count
}

function updateIndexButton() {
  const btn = document.getElementById('indexSelectedBtn');
  const delBtn = document.getElementById('deleteSelectedBtn');

  let indexableCount = 0;
  for (const doc of selectedDocs.values()) {
    if (doc.canIndex) indexableCount++;
  }

  const totalSelected = selectedDocs.size;

  if (btn) {
    btn.disabled = indexableCount === 0;
    btn.textContent = indexableCount > 0
      ? `Index ${indexableCount} Selected`
      : 'Index Selected Vectors';
  }
  if (delBtn) {
    delBtn.disabled = totalSelected === 0;
    delBtn.textContent = totalSelected > 0
      ? `Delete ${totalSelected} Selected`
      : 'Delete Selected';
  }
}

async function indexSelectedDocs() {
  if (selectedDocs.size === 0) return;
  if (!confirm(`Index ${selectedDocs.size} document(s) into Milvus? This will embed and store vectors.`)) return;

  const btn = document.getElementById('indexSelectedBtn');
  btn.textContent = 'Indexing...';
  btn.disabled = true;

  const byFolder = {};
  for (const [docId, doc] of selectedDocs.entries()) {
    if (!doc.canIndex) continue;
    const fid = doc.folderId;
    if (!byFolder[fid]) byFolder[fid] = [];
    byFolder[fid].push(docId);
  }

  let totalIndexed = 0;
  let totalErrors = 0;
  for (const [folderId, docIds] of Object.entries(byFolder)) {
    const result = await apiJSON(`/api/documents/index/${folderId}`, {
      method: 'POST',
      body: JSON.stringify({ doc_ids: docIds }),
    });
    if (result) {
      totalIndexed += result.indexed;
      totalErrors += result.errors;
    }
  }

  showAlert(`Indexing complete!\nIndexed: ${totalIndexed}\nErrors: ${totalErrors}`);
  selectedDocs.clear();
  loadDocuments();
}

async function deleteDoc(docId) {
  if (!confirm('Delete this document? This will remove the file, chunks, and vectors.')) return;
  await api(`/api/documents/${docId}`, { method: 'DELETE' });
  loadDocuments();
}

async function deleteSelectedDocs() {
  if (selectedDocs.size === 0) return;
  if (!confirm(`Delete ${selectedDocs.size} document(s)? This will remove files, chunks, and vectors. This cannot be undone.`)) return;

  const btn = document.getElementById('deleteSelectedBtn');
  btn.textContent = 'Deleting...';
  btn.disabled = true;

  const result = await apiJSON('/api/documents/delete-batch', {
    method: 'POST',
    body: JSON.stringify({ doc_ids: Array.from(selectedDocs.keys()) }),
  });

  if (result) {
    showAlert(`Deleted: ${result.deleted}\nErrors: ${result.errors}`);
  }

  selectedDocs.clear();
  loadDocuments();
}

// ── Multi-File Upload ──────────────────────────────────────────────

function showMultiUploadModal() {
  // Populate KB select
  const kbSel = document.getElementById('uploadKBSelect');
  kbSel.innerHTML = kbs.map(kb => `<option value="${kb.id}" ${kb.id === currentDocKBId ? 'selected' : ''}>${escapeHtml(kb.name)}</option>`).join('');
  loadUploadFolders();
  document.getElementById('uploadFileList').innerHTML = '';
  document.getElementById('uploadResult').classList.add('hidden');
  document.getElementById('uploadBtn').disabled = true;
  const fileInput = document.getElementById('uploadFiles');
  fileInput.value = '';
  showModal('uploadDocModal');
}

async function loadUploadFolders() {
  const kbId = document.getElementById('uploadKBSelect').value;
  if (!kbId) return;
  const folders = await apiJSON(`/api/kb/${kbId}/folders`);
  const sel = document.getElementById('uploadFolderSelect');
  if (folders && folders.length) {
    sel.innerHTML = folders.map(f => `<option value="${f.id}">${escapeHtml(f.name)}</option>`).join('');
  } else {
    sel.innerHTML = '<option value="">No folders</option>';
  }
}

function setupUploadDragDrop() {
  const dz = document.getElementById('dropZone');
  const fileInput = document.getElementById('uploadFiles');
  if (!dz || !fileInput) return;

  fileInput.addEventListener('change', () => previewUploadFiles(fileInput.files));

  dz.addEventListener('click', (e) => {
    e.stopPropagation();
    fileInput.click();
  });

  dz.addEventListener('dragover', (e) => { e.preventDefault(); dz.classList.add('border-blue-500'); });
  dz.addEventListener('dragleave', () => { dz.classList.remove('border-blue-500'); });
  dz.addEventListener('drop', (e) => {
    e.preventDefault();
    dz.classList.remove('border-blue-500');
    if (e.dataTransfer.files.length) {
      fileInput.files = e.dataTransfer.files;
      previewUploadFiles(e.dataTransfer.files);
    }
  });
}

const SUPPORTED_EXTS = new Set(['pdf','docx','pptx','xlsx','html','htm','txt','md','csv','json','xml','rtf']);

function previewUploadFiles(files) {
  const el = document.getElementById('uploadFileList');
  const btn = document.getElementById('uploadBtn');

  if (!files || !files.length) {
    el.innerHTML = '';
    btn.disabled = true;
    return;
  }

  let html = '';
  let hasValid = false;
  for (const f of files) {
    const ext = f.name.split('.').pop()?.toLowerCase() || '';
    const ok = SUPPORTED_EXTS.has(ext);
    if (ok) hasValid = true;
    html += `
      <div class="flex items-center justify-between px-3 py-1.5 rounded-lg ${ok ? 'bg-[#0f0f0f]' : 'bg-red-900/20 border border-red-900/40'}">
        <div class="flex items-center gap-2 text-sm min-w-0">
          <span class="${ok ? 'text-green-400' : 'text-red-400'}">${ok ? '<svg class=\"w-3 h-3\" fill=\"none\" stroke=\"currentColor\" viewBox=\"0 0 24 24\"><path stroke-linecap=\"round\" stroke-linejoin=\"round\" stroke-width=\"2\" d=\"M5 13l4 4L19 7\"/></svg>' : '<svg class=\"w-3 h-3\" fill=\"none\" stroke=\"currentColor\" viewBox=\"0 0 24 24\"><path stroke-linecap=\"round\" stroke-linejoin=\"round\" stroke-width=\"2\" d=\"M6 18L18 6M6 6l12 12\"/></svg>'}</span>
          <span class="truncate">${escapeHtml(f.name)}</span>
        </div>
        <div class="text-xs text-gray-500 flex-shrink-0">
          ${ok ? formatBytes(f.size) : '<span class="text-red-400">Unsupported</span>'}
        </div>
      </div>`;
  }
  el.innerHTML = html;
  btn.disabled = !hasValid;
}

async function uploadMultipleDocuments() {
  const folderId = document.getElementById('uploadFolderSelect').value;
  if (!folderId) { showAlert('Select a folder'); return; }

  const fileInput = document.getElementById('uploadFiles');
  if (!fileInput.files.length) return;

  const btn = document.getElementById('uploadBtn');
  btn.textContent = 'Uploading...';
  btn.disabled = true;

  const formData = new FormData();
  for (const f of fileInput.files) {
    formData.append('files', f);
  }

  try {
    const res = await fetch(`${API}/api/documents/upload/${folderId}`, {
      method: 'POST',
      headers: { 'Authorization': `Bearer ${TOKEN}` },
      body: formData,
    });
    const data = await res.json();

    // Show results
    const resEl = document.getElementById('uploadResult');
    resEl.classList.remove('hidden');
    let html = `<div class="space-y-1">`;
    html += `<p class="font-medium ${data.errors > 0 || data.incompatible > 0 ? 'text-yellow-400' : 'text-green-400'}">Upload Complete — ${data.uploaded} chunked, ${data.incompatible} incompatible, ${data.errors} errors</p>`;
    for (const r of data.results) {
      const icon = r.status === 'chunked' ? '<svg class=\"w-3 h-3\" fill=\"none\" stroke=\"currentColor\" viewBox=\"0 0 24 24\"><path stroke-linecap=\"round\" stroke-linejoin=\"round\" stroke-width=\"2\" d=\"M5 13l4 4L19 7\"/></svg>' : r.status === 'incompatible' ? '<svg class=\"w-3.5 h-3.5\" fill=\"none\" stroke=\"currentColor\" viewBox=\"0 0 24 24\"><path stroke-linecap=\"round\" stroke-linejoin=\"round\" stroke-width=\"2\" d=\"M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-2.5L13.732 4.5c-.77-.833-2.694-.833-3.464 0L3.34 16.5c-.77.833.192 2.5 1.732 2.5z\"/></svg>' : '<svg class=\"w-3 h-3\" fill=\"none\" stroke=\"currentColor\" viewBox=\"0 0 24 24\"><path stroke-linecap=\"round\" stroke-linejoin=\"round\" stroke-width=\"2\" d=\"M6 18L18 6M6 6l12 12\"/></svg>';
      const color = r.status === 'chunked' ? 'text-green-400' : r.status === 'incompatible' ? 'text-yellow-400' : 'text-red-400';
      html += `<div class="flex items-center gap-2 text-xs">
        <span class="${color}">${icon}</span>
        <span class="truncate">${escapeHtml(r.filename)}</span>
        <span class="${color}">${r.status}</span>
        ${r.chunk_count ? `<span class="text-gray-500">${r.chunk_count} chunks</span>` : ''}
        ${r.error ? `<span class="text-red-500 truncate">${escapeHtml(r.error)}</span>` : ''}
      </div>`;
    }
    html += `</div>`;
    resEl.innerHTML = html;

    loadDocuments();
  } catch (err) {
    showAlert('Upload failed: ' + err.message);
  }

  btn.textContent = 'Upload & Chunk';
  btn.disabled = false;
}

function showUploadModal(folderId) {
  showMultiUploadModal();
  // Pre-select the folder if we know the KB
  setTimeout(() => {
    const folderSel = document.getElementById('uploadFolderSelect');
    if (folderSel) {
      for (const opt of folderSel.options) {
        if (opt.value === folderId) { opt.selected = true; break; }
      }
    }
  }, 300);
}

// ── Models ─────────────────────────────────────────────────────────
let models = [];

async function loadModels() {
  models = await apiJSON('/api/admin/models') || [];
  renderModels();
}

function renderModels() {
  const el = document.getElementById('modelList');
  if (!models.length) {
    el.innerHTML = '<p class="text-gray-500">No models registered.</p>';
    return;
  }
  el.innerHTML = models.map(m => `
    <div class="bg-[#171717] border border-gray-800 rounded-xl p-4 flex flex-col sm:flex-row items-start sm:items-center justify-between gap-3">
      <div>
        <h3 class="font-medium">${escapeHtml(m.name)} ${m.is_default ? '<span class="text-yellow-400 text-xs">★ default</span>' : ''}</h3>
        <div class="flex gap-4 mt-1 text-xs text-gray-500">
          <span class="px-2 py-0.5 bg-gray-800 rounded">${m.model_type}</span>
          <span>${m.model_path}</span>
          ${m.vector_dimension ? `<span>dim: ${m.vector_dimension}</span>` : ''}
          <span class="${m.status === 'available' ? 'text-green-400' : 'text-yellow-400'}">${m.status}</span>
        </div>
      </div>
      <div class="flex items-center gap-2">
        ${!m.is_default ? `<button onclick="setDefaultModel('${m.id}')" class="px-3 py-1 bg-gray-700 hover:bg-gray-600 rounded-lg text-xs transition">Set Default</button>` : ''}
        <button onclick="deleteModel('${m.id}')" class="px-3 py-1 bg-red-900 hover:bg-red-800 rounded-lg text-xs transition">Delete</button>
      </div>
    </div>
  `).join('');
}

function showRegisterModelModal() { showModal('registerModelModal'); }

async function registerModel() {
  const name = document.getElementById('newModelName').value.trim();
  const model_type = document.getElementById('newModelType').value;
  const model_path = document.getElementById('newModelPath').value.trim();
  const dim = document.getElementById('newModelDim').value;
  const is_default = document.getElementById('newModelDefault').checked;

  if (!name || !model_path) return;

  await apiJSON('/api/admin/models', {
    method: 'POST',
    body: JSON.stringify({
      name, model_type, model_path, is_default,
      vector_dimension: dim ? parseInt(dim) : null,
    }),
  });
  hideModal('registerModelModal');
  await loadModels();
}

async function setDefaultModel(id) {
  await api(`/api/admin/models/${id}/default`, { method: 'PUT' });
  await loadModels();
}

async function deleteModel(id) {
  if (!confirm('Delete this model?')) return;
  await api(`/api/admin/models/${id}`, { method: 'DELETE' });
  await loadModels();
}

// ── LLM Hot-Swap ──────────────────────────────────────────────────
let _adminLLMPollTimer = null;

async function loadLLMSwap() {
  const data = await apiJSON('/api/admin/models/llm/available');
  if (!data) return;

  // Populate current model
  const curEl = document.getElementById('llmCurrentModel');
  if (curEl) curEl.textContent = data.current_model || '—';

  // Populate target dropdown (exclude current model)
  const sel = document.getElementById('llmTargetSelect');
  if (sel) {
    sel.innerHTML = '<option value="">Select a model…</option>' +
      (data.available_models || [])
        .filter(m => m !== data.current_model)
        .map(m => `<option value="${m}">${escapeHtml(m)}</option>`)
        .join('');
  }

  // Update status badge
  updateLLMStatusUI(data.status, data.error);
}

function updateLLMStatusUI(status, error) {
  const badge = document.getElementById('llmStatusBadge');
  const btn = document.getElementById('llmSwitchBtn');
  const errBox = document.getElementById('llmErrorBox');

  if (badge) {
    if (status === 'switching') {
      badge.className = 'px-3 py-1.5 rounded-lg text-xs font-medium bg-yellow-900/40 text-yellow-400 border border-yellow-800 animate-pulse';
      badge.textContent = 'Switching…';
    } else if (status === 'error') {
      badge.className = 'px-3 py-1.5 rounded-lg text-xs font-medium bg-red-900/40 text-red-400 border border-red-800';
      badge.textContent = 'Error';
    } else {
      badge.className = 'px-3 py-1.5 rounded-lg text-xs font-medium bg-green-900/40 text-green-400 border border-green-800';
      badge.textContent = 'Ready';
    }
  }

  if (btn) {
    btn.disabled = status === 'switching';
    btn.textContent = status === 'switching' ? 'Switching…' : 'Switch Model';
  }

  if (errBox) {
    if (error) {
      errBox.classList.remove('hidden');
      errBox.textContent = error;
    } else {
      errBox.classList.add('hidden');
      errBox.textContent = '';
    }
  }

  // Auto-poll while switching
  if (status === 'switching') {
    if (!_adminLLMPollTimer) {
      _adminLLMPollTimer = setInterval(loadLLMSwap, 8_000);
    }
  } else {
    if (_adminLLMPollTimer) {
      clearInterval(_adminLLMPollTimer);
      _adminLLMPollTimer = null;
    }
  }
}

async function switchLLM() {
  const sel = document.getElementById('llmTargetSelect');
  const target = sel?.value;
  if (!target) { showAlert('Select a target model'); return; }
  if (!confirm(`Switch vLLM to "${target}"?\n\nThis will stop the current model, free VRAM, and start the new one. Chat will be unavailable during the switch.`)) return;

  const btn = document.getElementById('llmSwitchBtn');
  if (btn) { btn.disabled = true; btn.textContent = 'Initiating…'; }

  const res = await api('/api/admin/models/llm/switch', {
    method: 'POST',
    body: JSON.stringify({ model_dir_name: target }),
  });

  if (res && res.ok) {
    updateLLMStatusUI('switching', null);
    // Start polling
    if (!_adminLLMPollTimer) {
      _adminLLMPollTimer = setInterval(loadLLMSwap, 8_000);
    }
  } else {
    const err = await res?.json().catch(() => null);
    showAlert(err?.detail || 'Failed to initiate model switch');
    if (btn) { btn.disabled = false; btn.textContent = 'Switch Model'; }
  }
}

// ── Roles ──────────────────────────────────────────────────────────
let roles = [];

async function loadRoles() {
  roles = await apiJSON('/api/admin/roles') || [];
  renderRoles();
}

function renderRoles() {
  const el = document.getElementById('roleList');
  if (!roles.length) {
    el.innerHTML = `
      <div class="bg-[#171717] border border-gray-800 rounded-xl p-6 text-center">
        <p class="text-gray-400">No custom roles yet.</p>
        <p class="text-gray-500 text-sm mt-1">Create a role to restrict which knowledge bases users can access.</p>
      </div>`;
    return;
  }
  el.innerHTML = roles.map(r => {
    const kbNames = r.kb_ids.map(id => {
      const kb = kbs.find(k => k.id === id);
      return kb ? escapeHtml(kb.name) : 'Unknown KB';
    });
    return `
    <div class="bg-[#171717] border border-gray-800 rounded-xl p-4">
      <div class="flex flex-col sm:flex-row items-start sm:items-center justify-between mb-2 gap-2">
        <div>
          <h3 class="font-medium">${escapeHtml(r.name)}</h3>
          <p class="text-sm text-gray-400">${escapeHtml(r.description || 'No description')}</p>
        </div>
        <div class="flex items-center gap-2">
          <span class="text-xs text-gray-500">${r.user_count} user${r.user_count !== 1 ? 's' : ''}</span>
          <button onclick="editRole('${r.id}')" class="px-3 py-1 bg-gray-700 hover:bg-gray-600 rounded-lg text-xs transition">Edit</button>
          <button onclick="deleteRole('${r.id}')" class="px-3 py-1 bg-red-900 hover:bg-red-800 rounded-lg text-xs transition">Delete</button>
        </div>
      </div>
      <div class="flex flex-wrap gap-1 mt-2">
        ${kbNames.length
          ? kbNames.map(n => `<span class="px-2 py-0.5 bg-blue-900/40 border border-blue-800 rounded-md text-xs text-blue-300">${n}</span>`).join('')
          : '<span class="text-xs text-gray-500">No KB access assigned</span>'
        }
      </div>
    </div>`;
  }).join('');
}

function showCreateRoleModal() {
  document.getElementById('newRoleName').value = '';
  document.getElementById('newRoleDesc').value = '';
  renderKBCheckboxesForRole('roleKBCheckboxes', []);
  showModal('createRoleModal');
}

function renderKBCheckboxesForRole(containerId, selectedKBIds) {
  const el = document.getElementById(containerId);
  if (!kbs.length) {
    el.innerHTML = '<p class="text-gray-500 text-xs p-1">No knowledge bases available</p>';
    return;
  }
  el.innerHTML = kbs.map(kb => `
    <label class="flex items-center gap-2 px-2 py-1.5 rounded hover:bg-[#1a1a1a] cursor-pointer text-sm">
      <input type="checkbox" value="${kb.id}" ${selectedKBIds.includes(kb.id) ? 'checked' : ''} class="rounded">
      <span>${escapeHtml(kb.name)}</span>
    </label>
  `).join('');
}

function getCheckedKBIds(containerId) {
  const el = document.getElementById(containerId);
  return Array.from(el.querySelectorAll('input[type="checkbox"]:checked')).map(cb => cb.value);
}

async function createRole() {
  const name = document.getElementById('newRoleName').value.trim();
  if (!name) { showAlert('Role name is required'); return; }
  const description = document.getElementById('newRoleDesc').value.trim();
  const kbIds = getCheckedKBIds('roleKBCheckboxes');

  const result = await apiJSON('/api/admin/roles', {
    method: 'POST',
    body: JSON.stringify({ name, description: description || null }),
  });
  if (!result) return;

  if (kbIds.length) {
    await apiJSON(`/api/admin/roles/${result.id}/kb-access`, {
      method: 'PUT',
      body: JSON.stringify({ kb_ids: kbIds }),
    });
  }

  hideModal('createRoleModal');
  await loadRoles();
}

async function editRole(roleId) {
  const role = roles.find(r => r.id === roleId);
  if (!role) return;
  document.getElementById('editRoleId').value = roleId;
  document.getElementById('editRoleName').value = role.name;
  document.getElementById('editRoleDesc').value = role.description || '';
  document.getElementById('editRoleTitle').textContent = role.name;
  renderKBCheckboxesForRole('editRoleKBCheckboxes', role.kb_ids || []);
  showModal('editRoleModal');
}

async function saveRole() {
  const roleId = document.getElementById('editRoleId').value;
  const name = document.getElementById('editRoleName').value.trim();
  if (!name) { showAlert('Role name is required'); return; }
  const description = document.getElementById('editRoleDesc').value.trim();
  const kbIds = getCheckedKBIds('editRoleKBCheckboxes');

  await apiJSON(`/api/admin/roles/${roleId}`, {
    method: 'PUT',
    body: JSON.stringify({ name, description: description || null }),
  });

  await apiJSON(`/api/admin/roles/${roleId}/kb-access`, {
    method: 'PUT',
    body: JSON.stringify({ kb_ids: kbIds }),
  });

  hideModal('editRoleModal');
  await loadRoles();
}

async function deleteRole(roleId) {
  const role = roles.find(r => r.id === roleId);
  if (!confirm(`Delete role "${role?.name || roleId}"? Users with this role will lose their KB restrictions.`)) return;
  await api(`/api/admin/roles/${roleId}`, { method: 'DELETE' });
  await loadRoles();
  await loadUsers();
}

// ── Config ─────────────────────────────────────────────────────────
let currentConfig = {};

async function loadConfig() {
  const config = await apiJSON('/api/admin/config');
  if (config) {
    currentConfig = config;
    renderConfigForm(config);
  }
}

function renderConfigForm(config) {
  const el = document.getElementById('configForm');
  if (!el) return;

  let html = '';
  for (const [section, values] of Object.entries(config)) {
    html += `
      <div class="bg-[#171717] rounded-xl border border-gray-800 p-6">
        <h3 class="text-lg font-semibold text-white mb-4 capitalize">${escapeHtml(section)}</h3>
        <div class="grid grid-cols-1 md:grid-cols-2 gap-4">
    `;
    
    for (const [key, val] of Object.entries(values)) {
      const type = typeof val;
      const inputId = `cfg_${section}_${key}`;
      const escapedKey = escapeHtml(key);
      const safeSection = section.replace(/'/g, "\\'");
      const safeKey = key.replace(/'/g, "\\'");

      if (type === 'boolean') {
        html += `
          <div class="flex items-center gap-2 mt-7">
            <input type="checkbox" id="${inputId}" class="rounded w-4 h-4" ${val ? 'checked' : ''} 
                   onchange="updateConfigValue('${safeSection}', '${safeKey}', this.checked)">
            <label for="${inputId}" class="text-sm font-medium text-gray-300 cursor-pointer">${escapedKey}</label>
          </div>
        `;
      } else if (type === 'number') {
        html += `
          <div>
            <label for="${inputId}" class="block text-sm font-medium text-gray-300 mb-1">${escapedKey}</label>
            <input type="number" id="${inputId}" value="${val}" 
                   onchange="updateConfigValue('${safeSection}', '${safeKey}', parseFloat(this.value))"
                   class="w-full px-4 py-2 bg-[#0f0f0f] border border-gray-700 rounded-lg text-sm focus:outline-none focus:border-blue-500">
          </div>
        `;
      } else {
        if (key.toLowerCase().includes('prompt')) {
          html += `
            <div class="md:col-span-2">
              <label for="${inputId}" class="block text-sm font-medium text-gray-300 mb-1">${escapedKey}</label>
              <textarea id="${inputId}" rows="6"
                     onchange="updateConfigValue('${safeSection}', '${safeKey}', this.value)"
                     class="w-full px-4 py-2 bg-[#0f0f0f] border border-gray-700 rounded-lg text-sm focus:outline-none focus:border-blue-500">${escapeHtml(String(val))}</textarea>
            </div>
          `;
        } else {
          html += `
            <div>
              <label for="${inputId}" class="block text-sm font-medium text-gray-300 mb-1">${escapedKey}</label>
              <input type="text" id="${inputId}" value="${escapeHtml(String(val))}" 
                     onchange="updateConfigValue('${safeSection}', '${safeKey}', this.value)"
                     class="w-full px-4 py-2 bg-[#0f0f0f] border border-gray-700 rounded-lg text-sm focus:outline-none focus:border-blue-500">
            </div>
          `;
        }
      }
    }
    html += `</div></div>`;
  }
  
  el.innerHTML = html;
}

window.updateConfigValue = function(section, key, value) {
  if (currentConfig[section]) {
    currentConfig[section][key] = value;
  }
};

async function saveConfig() {
  try {
    await apiJSON('/api/admin/config', {
      method: 'PUT',
      body: JSON.stringify({ updates: currentConfig }),
    });
    showAlert('Configuration saved!');
  } catch (err) {
    showAlert('Error saving configuration: ' + err.message);
  }
}

function exportConfig() {
  if (!currentConfig || Object.keys(currentConfig).length === 0) return;
  const text = JSON.stringify(currentConfig, null, 2);
  const blob = new Blob([text], { type: 'application/json' });
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = 'config.json';
  a.click();
}

// ── Users ──────────────────────────────────────────────────────────
let userPage = 1;
const USER_LIMIT = 50;

async function loadUsers() {
  const data = await apiJSON(`/api/admin/users?skip=${(userPage - 1) * USER_LIMIT}&limit=${USER_LIMIT}`);
  if (!data) return;
  const el = document.getElementById('userList');
  if (!data.items || !data.items.length) {
    el.innerHTML = '<p class="text-gray-500">No users found.</p>';
    renderUserPagination(0, USER_LIMIT);
    return;
  }
  el.innerHTML = data.items.map(u => {
    const roleOptions = roles.map(r =>
      `<option value="${r.id}" ${u.custom_role_id === r.id ? 'selected' : ''}>${escapeHtml(r.name)}</option>`
    ).join('');
    const avatar = (u.username || '?')[0].toUpperCase();
    const isSelf = currentUser && u.id === currentUser.id;
    const isProtectedSuperAdmin = u.role === 'superadmin' && currentUser && currentUser.role !== 'superadmin';
    const cannotEditRole = isSelf || isProtectedSuperAdmin;
    return `
    <div class="bg-[#171717] border border-gray-800 rounded-xl p-4 flex flex-col sm:flex-row items-start sm:items-center justify-between gap-3">
      <div class="flex items-center gap-3">
        <div class="w-8 h-8 bg-blue-600 rounded-full flex items-center justify-center text-sm font-bold flex-shrink-0">${avatar}</div>
        <div>
          <h3 class="font-medium">${escapeHtml(u.username)} <span class="text-gray-500 text-sm">(${u.email})</span>${isSelf ? ' <span class="text-xs text-blue-400">(you)</span>' : ''}</h3>
          <div class="flex gap-3 mt-1 text-xs">
            <span class="px-2 py-0.5 bg-gray-800 rounded">${u.role}</span>
            ${u.custom_role_name ? `<span class="px-2 py-0.5 bg-blue-900/40 border border-blue-800 rounded text-blue-300">${escapeHtml(u.custom_role_name)}</span>` : ''}
            <span class="${u.is_active ? 'text-green-400' : 'text-red-400'}">${u.is_active ? 'Active' : 'Inactive'}</span>
          </div>
        </div>
      </div>
      <div class="flex items-center gap-3">
        <div class="flex flex-col gap-1">
          ${cannotEditRole 
            ? `<div class="w-44 bg-[#0f0f0f] border border-gray-700 rounded-lg px-2 py-1 text-xs text-gray-500 capitalize cursor-default text-center" title="${isSelf ? 'You cannot change your own role' : 'You cannot change a superadmin role'}">${u.role}</div>`
            : `<select onchange="changeRole('${u.id}', this.value)" class="w-44 bg-[#0f0f0f] border border-gray-700 rounded-lg px-2 py-1 text-xs" title="Admin level">
                 <option value="user" ${u.role === 'user' ? 'selected' : ''}>User</option>
                 <option value="admin" ${u.role === 'admin' ? 'selected' : ''}>Admin</option>
                 <option value="superadmin" ${u.role === 'superadmin' ? 'selected' : ''}>Superadmin</option>
               </select>`
          }
          <select onchange="changeCustomRole('${u.id}', '${u.role}', this.value)" class="w-44 bg-[#0f0f0f] border border-gray-700 rounded-lg px-2 py-1 text-xs" title="KB access role">
            <option value="">No KB restriction</option>
            ${roleOptions}
          </select>
        </div>
        ${!isSelf 
          ? `<button onclick="deleteUserAccount('${u.id}', '${escapeHtml(u.username)}')" class="p-2 text-gray-500 hover:text-red-400 hover:bg-red-900/30 rounded-lg transition flex-shrink-0" title="Delete user">
               <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                 <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16"/>
               </svg>
             </button>`
          : `<div class="w-8 h-8 flex-shrink-0"></div>`
        }
      </div>
    </div>`;
  }).join('');
  renderUserPagination(data.total, data.limit);
}

function renderUserPagination(total, limit) {
  const el = document.getElementById('userPagination');
  const totalPages = Math.ceil(total / limit);
  if (totalPages <= 1) { el.innerHTML = ''; return; }
  
  let html = '';
  html += `<button onclick="goUserPage(1)" class="px-3 py-1 rounded-lg text-xs ${userPage === 1 ? 'bg-gray-800 text-gray-600' : 'bg-gray-700 hover:bg-gray-600'}" ${userPage === 1 ? 'disabled' : ''}>«</button>`;
  html += `<button onclick="goUserPage(${userPage - 1})" class="px-3 py-1 rounded-lg text-xs ${userPage === 1 ? 'bg-gray-800 text-gray-600' : 'bg-gray-700 hover:bg-gray-600'}" ${userPage === 1 ? 'disabled' : ''}>‹</button>`;
  for (let i = 1; i <= totalPages; i++) {
    html += `<button onclick="goUserPage(${i})" class="px-3 py-1 rounded-lg text-xs ${i === userPage ? 'bg-blue-600 text-white' : 'bg-gray-700 hover:bg-gray-600'}">${i}</button>`;
  }
  html += `<button onclick="goUserPage(${userPage + 1})" class="px-3 py-1 rounded-lg text-xs ${userPage === totalPages ? 'bg-gray-800 text-gray-600' : 'bg-gray-700 hover:bg-gray-600'}" ${userPage === totalPages ? 'disabled' : ''}>›</button>`;
  html += `<button onclick="goUserPage(${totalPages})" class="px-3 py-1 rounded-lg text-xs ${userPage === totalPages ? 'bg-gray-800 text-gray-600' : 'bg-gray-700 hover:bg-gray-600'}" ${userPage === totalPages ? 'disabled' : ''}>»</button>`;
  el.innerHTML = html;
}

function goUserPage(page) {
  userPage = page;
  loadUsers();
}

async function changeRole(userId, role) {
  // Prevent self-demotion
  if (currentUser && userId === currentUser.id) {
    showAlert('You cannot change your own role.');
    await loadUsers(); // Reset the dropdown
    return;
  }
  const data = await apiJSON(`/api/admin/users?skip=${(userPage - 1) * USER_LIMIT}&limit=${USER_LIMIT}`);
  const user = data?.items?.find(u => u.id === userId);
  await api(`/api/admin/users/${userId}/role`, {
    method: 'PUT',
    body: JSON.stringify({ role, custom_role_id: user?.custom_role_id || null }),
  });
  await loadUsers();
}

async function changeCustomRole(userId, adminLevel, customRoleId) {
  await api(`/api/admin/users/${userId}/role`, {
    method: 'PUT',
    body: JSON.stringify({ role: adminLevel, custom_role_id: customRoleId || null }),
  });
  await loadUsers();
}

async function deleteUserAccount(userId, username) {
  if (!confirm(`Are you sure you want to permanently delete user "${username}" and all their data? This cannot be undone.`)) return;
  const res = await api(`/api/admin/users/${userId}/delete`, { method: 'DELETE' });
  if (res && res.ok) {
    showToast('User deleted successfully', 'success');
    await loadUsers();
  } else {
    showToast('Failed to delete user', 'error');
  }
}

// ── Import Transcripts ─────────────────────────────────────────────
async function importTranscripts() {
  const kbId = document.getElementById('importKBSelect').value;
  const batchSize = parseInt(document.getElementById('importBatchSize').value) || 16;
  if (!kbId) { showToast('Please select a knowledge base first', 'error'); return; }

  const btn = document.getElementById('importBtn');
  btn.textContent = 'Importing…';
  btn.disabled = true;

  // Show inline spinner
  const resEl = document.getElementById('importResult');
  resEl.classList.remove('hidden');
  resEl.innerHTML = `
    <div class="flex items-center gap-3 text-gray-400">
      <svg class="animate-spin w-5 h-5 text-blue-400 flex-shrink-0" fill="none" viewBox="0 0 24 24">
        <circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"></circle>
        <path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v8z"></path>
      </svg>
      <span>Importing transcripts — this may take several minutes…</span>
    </div>`;

  const result = await apiJSON('/api/admin/import-transcripts', {
    method: 'POST',
    body: JSON.stringify({ kb_id: kbId, batch_size: batchSize }),
  });

  btn.textContent = 'Start Import';
  btn.disabled = false;

  if (result) {
    const hasErrors = result.errors > 0;
    resEl.innerHTML = `
      <p class="font-medium ${hasErrors ? 'text-yellow-400' : 'text-green-400'} mb-2">Import Complete</p>
      <div class="grid grid-cols-2 gap-2 text-sm text-gray-300">
        <span class="text-gray-500">Total files:</span><span>${result.total_files}</span>
        <span class="text-gray-500">Imported:</span><span class="text-green-400">${result.imported}</span>
        <span class="text-gray-500">Skipped:</span><span class="text-gray-400">${result.skipped}</span>
        <span class="text-gray-500">Errors:</span><span class="${hasErrors ? 'text-red-400' : 'text-gray-400'}">${result.errors}</span>
      </div>`;
  } else {
    resEl.innerHTML = `
      <div class="flex items-center gap-2 text-red-400">
        <svg class="w-4 h-4 flex-shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"/></svg>
        <span>Import failed. Check server logs for details.</span>
      </div>`;
    showToast('Transcript import failed — see server logs', 'error');
  }
}

// ── System Utilities ───────────────────────────────────────────────

async function loadSystemStats() {
  const stats = await apiJSON('/api/admin/utilities/stats');
  if (!stats) return;

  // Render summary cards
  const el = document.getElementById('systemStats');
  el.innerHTML = `
    <div class="bg-[#171717] border border-gray-800 rounded-xl p-4 text-center">
      <p class="text-2xl font-bold text-white">${stats.users}</p>
      <p class="text-xs text-gray-400 mt-1">Users</p>
    </div>
    <div class="bg-[#171717] border border-gray-800 rounded-xl p-4 text-center">
      <p class="text-2xl font-bold text-white">${stats.knowledge_bases}</p>
      <p class="text-xs text-gray-400 mt-1">Knowledge Bases</p>
    </div>
    <div class="bg-[#171717] border border-gray-800 rounded-xl p-4 text-center">
      <p class="text-2xl font-bold text-white">${stats.documents}</p>
      <p class="text-xs text-gray-400 mt-1">Documents</p>
    </div>
    <div class="bg-[#171717] border border-gray-800 rounded-xl p-4 text-center">
      <p class="text-2xl font-bold text-white">${stats.chunks}</p>
      <p class="text-xs text-gray-400 mt-1">Chunks</p>
    </div>
    <div class="bg-[#171717] border border-gray-800 rounded-xl p-4 text-center">
      <p class="text-2xl font-bold text-white">${stats.chat_sessions}</p>
      <p class="text-xs text-gray-400 mt-1">Chat Sessions</p>
    </div>
    <div class="bg-[#171717] border border-gray-800 rounded-xl p-4 text-center">
      <p class="text-2xl font-bold text-white">${stats.chat_messages}</p>
      <p class="text-xs text-gray-400 mt-1">Messages</p>
    </div>
    <div class="bg-[#171717] border border-gray-800 rounded-xl p-4 text-center">
      <p class="text-2xl font-bold ${stats.milvus_connected ? 'text-green-400' : 'text-red-400'}">${stats.milvus_connected ? 'Connected' : 'Offline'}</p>
      <p class="text-xs text-gray-400 mt-1">Milvus Status</p>
    </div>
    <div class="bg-[#171717] border border-gray-800 rounded-xl p-4 text-center">
      <p class="text-2xl font-bold text-white">${stats.milvus_collections}</p>
      <p class="text-xs text-gray-400 mt-1">Milvus Collections</p>
    </div>
  `;

  // Render per-KB stats with reindex buttons
  const kbEl = document.getElementById('kbStats');
  if (!stats.kb_details.length) {
    kbEl.innerHTML = '<p class="text-gray-500 text-sm">No knowledge bases.</p>';
    return;
  }
  kbEl.innerHTML = stats.kb_details.map(kb => `
    <div class="bg-[#171717] border border-gray-800 rounded-xl p-4">
      <div class="flex items-center justify-between mb-3">
        <div>
          <h4 class="font-medium">${escapeHtml(kb.name)}</h4>
          <p class="text-xs text-gray-500 mt-0.5">Collection: ${kb.collection_name || '—'}</p>
        </div>
        <div class="flex items-center gap-2">
          <button onclick="reindexKB('${kb.id}', '${escapeHtml(kb.name)}')"
                  class="px-3 py-1.5 bg-yellow-700 hover:bg-yellow-600 rounded-lg text-xs transition"
                  id="reindexBtn-${kb.id}">
            Reindex Vectors
          </button>
          <button onclick="rebuildCollection('${kb.id}', '${escapeHtml(kb.name)}')"
                  class="px-3 py-1.5 bg-orange-800 hover:bg-orange-700 rounded-lg text-xs transition">
            Rebuild Collection
          </button>
        </div>
      </div>
      <div class="grid grid-cols-1 sm:grid-cols-3 gap-3 text-center">
        <div class="bg-[#0f0f0f] rounded-lg p-2">
          <p class="text-lg font-semibold text-white">${kb.documents}</p>
          <p class="text-xs text-gray-500">Documents</p>
        </div>
        <div class="bg-[#0f0f0f] rounded-lg p-2">
          <p class="text-lg font-semibold text-white">${kb.chunks}</p>
          <p class="text-xs text-gray-500">DB Chunks</p>
        </div>
        <div class="bg-[#0f0f0f] rounded-lg p-2">
          <p class="text-lg font-semibold text-white">${kb.milvus_entities}</p>
          <p class="text-xs text-gray-500">Milvus Vectors</p>
        </div>
      </div>
    </div>
  `).join('');
}

async function reindexKB(kbId, kbName) {
  if (!confirm(`Reindex "${kbName}"? This will re-embed and re-insert ALL chunks into Milvus. This can take a long time for large KBs.`)) return;

  const btn = document.getElementById(`reindexBtn-${kbId}`);
  if (btn) { btn.textContent = 'Reindexing...'; btn.disabled = true; }

  const result = await apiJSON(`/api/admin/utilities/reindex/${kbId}`, {
    method: 'POST',
    body: JSON.stringify({ batch_size: 16 }),
  });

  if (btn) { btn.textContent = 'Reindex Vectors'; btn.disabled = false; }

  if (result) {
    showAlert(`Reindex complete!\n\nDocuments processed: ${result.documents_processed}\nChunks indexed: ${result.chunks_indexed}\nErrors: ${result.errors}\nMilvus entities: ${result.milvus_entities}`);
    loadSystemStats();
  } else {
    showAlert('Reindex failed. Check server logs.');
  }
}

async function rebuildCollection(kbId, kbName) {
  if (!confirm(`Rebuild collection for "${kbName}"? This will DROP the Milvus collection and recreate it empty. All documents will be marked as pending and need to be reindexed.`)) return;

  const result = await apiJSON(`/api/admin/utilities/rebuild-collection/${kbId}`, {
    method: 'POST',
  });

  if (result) {
    showAlert(`Collection rebuilt. ${result.documents_reset} documents marked as pending.\nUse "Reindex Vectors" to re-embed them.`);
    loadSystemStats();
  } else {
    showAlert('Rebuild failed. Check server logs.');
  }
}

async function clearAllChats() {
  if (!confirm('Delete ALL chat sessions and messages for ALL users? This cannot be undone.')) return;
  if (!confirm('Are you SURE? This will permanently delete all conversations.')) return;

  const result = await apiJSON('/api/admin/utilities/clear-chats', { method: 'DELETE' });
  if (result) {
    showAlert(`Cleared ${result.sessions_deleted} sessions and ${result.messages_deleted} messages.`);
    loadSystemStats();
  }
}

async function purgeOrphanChunks() {
  if (!confirm('Remove orphan document chunks from the database?')) return;

  const result = await apiJSON('/api/admin/utilities/orphan-chunks', { method: 'DELETE' });
  if (result) {
    showAlert(`Purged ${result.orphan_documents_removed} orphan documents.`);
    loadSystemStats();
  }
}

// ── Utils ──────────────────────────────────────────────────────────

/**
 * Show a transient toast notification.
 * @param {string} message  Text to display.
 * @param {'success'|'error'|'info'} type  Visual variant. Defaults to 'info'.
 * @param {number} duration  Auto-dismiss delay in ms. Default 4000.
 */
function showToast(message, type = 'info', duration = 4000) {
  const container = document.getElementById('toastContainer');
  if (!container) return;

  const colorMap = {
    success: 'bg-green-900/90 border-green-700 text-green-200',
    error:   'bg-red-900/90   border-red-700   text-red-200',
    info:    'bg-gray-800/90  border-gray-600  text-gray-200',
  };
  const iconMap = {
    success: '<path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M5 13l4 4L19 7"/>',
    error:   '<path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"/>',
    info:    '<path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"/>',
  };

  const toast = document.createElement('div');
  toast.className = [
    'flex items-center gap-3 px-4 py-3 rounded-xl border shadow-lg backdrop-blur-sm',
    'text-sm max-w-sm w-full transition-all duration-300 opacity-0 translate-y-2',
    colorMap[type] || colorMap.info,
  ].join(' ');

  toast.innerHTML = `
    <svg class="w-4 h-4 flex-shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24">
      ${iconMap[type] || iconMap.info}
    </svg>
    <span class="flex-1">${escapeHtml(message)}</span>
    <button onclick="this.parentElement.remove()" class="text-current opacity-60 hover:opacity-100 flex-shrink-0">
      <svg class="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12"/>
      </svg>
    </button>`;

  container.appendChild(toast);

  // Animate in
  requestAnimationFrame(() => {
    toast.classList.remove('opacity-0', 'translate-y-2');
    toast.classList.add('opacity-100', 'translate-y-0');
  });

  // Auto-dismiss
  setTimeout(() => {
    toast.classList.add('opacity-0', 'translate-y-2');
    setTimeout(() => toast.remove(), 300);
  }, duration);
}

function escapeHtml(str) {
  if (!str) return '';
  const div = document.createElement('div');
  div.textContent = str;
  return div.innerHTML;
}

function formatBytes(bytes) {
  if (!bytes || bytes === 0) return '0 B';
  const k = 1024;
  const sizes = ['B', 'KB', 'MB', 'GB'];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  return parseFloat((bytes / Math.pow(k, i)).toFixed(1)) + ' ' + sizes[i];
}
