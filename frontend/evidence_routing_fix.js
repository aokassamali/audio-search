(() => {
  const $ = (id) => document.getElementById(id);
  const esc = (value = '') => String(value).replace(/[&<>'"]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;',"'":'&#39;','"':'&quot;'}[c]));

  window.__AUDIO_SEARCH_FRONTEND_BUILD__ = '20260822-evidence-routing-fix';

  async function api(path) {
    const response = await fetch(path);
    if (!response.ok) {
      let message = `${response.status} ${response.statusText}`;
      try { message = (await response.json()).detail || message; } catch (_) {}
      throw new Error(message);
    }
    return response.json();
  }

  function formatTime(value) {
    if (value == null || Number(value) < 0) return 'Untimed';
    const total = Math.floor(Number(value));
    const h = Math.floor(total / 3600);
    const m = Math.floor((total % 3600) / 60);
    const s = total % 60;
    return h
      ? `${h}:${String(m).padStart(2, '0')}:${String(s).padStart(2, '0')}`
      : `${m}:${String(s).padStart(2, '0')}`;
  }

  function activateTranscriptView() {
    document.querySelectorAll('.nav-item[data-view]').forEach(item => {
      item.classList.toggle('active', item.dataset.view === 'transcript');
    });
    $('askView')?.classList.remove('view-active');
    $('transcriptView')?.classList.add('view-active');
    if ($('viewEyebrow')) $('viewEyebrow').textContent = 'Human-verifiable source of truth';
    if ($('viewTitle')) $('viewTitle').textContent = 'Audit the transcript';
  }

  function syncSourcePicker(source) {
    const pickerValue = $('transcriptSourceButton')?.querySelector('.picker-value');
    if (pickerValue) pickerValue.textContent = source.display_name;

    const legacy = $('transcriptSource');
    if (legacy) legacy.value = source.source_key;

    document.querySelectorAll('[data-transcript-key]').forEach(input => {
      const shouldCheck = input.dataset.transcriptKey === source.source_key;
      if (input.checked !== shouldCheck) {
        input.checked = shouldCheck;
        input.dispatchEvent(new Event('change', {bubbles:true}));
      }
    });
  }

  function renderContext(chunks, source, chunkId) {
    const list = $('transcriptList');
    if (!list) return;
    list.innerHTML = chunks.map(chunk => {
      const focused = Number(chunk.chunk_id) === Number(chunkId);
      return `<article class="chunk-row ${focused ? 'context-focus' : ''}">
        <div class="chunk-time">${formatTime(chunk.start)}</div>
        <div class="chunk-text">${esc(chunk.speaker_text || chunk.text || '')}</div>
        <div class="chunk-id"><div>chunk ${esc(chunk.chunk_id)}</div><div class="chunk-source">${esc(source.display_name)}</div></div>
      </article>`;
    }).join('');
    setTimeout(() => list.querySelector('.context-focus')?.scrollIntoView({behavior:'smooth', block:'center'}), 0);
  }

  async function showContext(source, chunkId, radius = 3) {
    activateTranscriptView();
    syncSourcePicker(source);
    if ($('transcriptSearch')) $('transcriptSearch').value = '';
    if ($('transcriptList')) {
      $('transcriptList').innerHTML = '<div class="loading-card"><div class="loading-line"></div><div class="loading-line"></div></div>';
    }

    const data = await api(`/sources/${encodeURIComponent(source.source_key)}/chunks?limit=2000`);
    const all = data.chunks || [];
    const index = all.findIndex(chunk => Number(chunk.chunk_id) === Number(chunkId));
    if (index < 0) {
      if ($('transcriptMeta')) $('transcriptMeta').textContent = `Chunk ${chunkId} was not found in ${source.display_name}.`;
      if ($('transcriptList')) $('transcriptList').innerHTML = '';
      return;
    }

    const start = Math.max(0, index - radius);
    const end = Math.min(all.length, index + radius + 1);
    const context = all.slice(start, end);
    if ($('transcriptMeta')) {
      $('transcriptMeta').textContent = `${context.length} chunks · ${source.display_name} · context around chunk ${chunkId}`;
    }
    renderContext(context, source, chunkId);
  }

  async function drawerTarget() {
    const title = $('drawerTitle')?.textContent?.trim();
    const chips = [...document.querySelectorAll('#drawerBody .meta-chip')];
    const match = chips[0]?.textContent?.match(/Chunk\s+(\d+)/i);
    if (!title || !match) return null;
    const data = await api('/sources');
    const source = (data.sources || []).find(item => item.display_name === title);
    if (!source) return null;
    return {source, chunkId:Number(match[1])};
  }

  async function routeDrawerToTranscript() {
    const target = await drawerTarget();
    if (!target) return;
    $('evidenceDrawer')?.classList.remove('open');
    await showContext(target.source, target.chunkId);
  }

  // Route both versions of the evidence drawer through one source-of-truth path.
  // Capture phase prevents the legacy single-source transcript loader from racing
  // the newer audit/context UI and leaving a stale source label behind.
  document.addEventListener('click', event => {
    const button = event.target.closest?.('#viewTranscriptEvidence, #auditLocateEvidence');
    if (!button) return;
    event.preventDefault();
    event.stopPropagation();
    event.stopImmediatePropagation();
    routeDrawerToTranscript().catch(error => {
      if ($('transcriptList')) $('transcriptList').innerHTML = `<div class="answer-card refusal">${esc(error.message)}</div>`;
    });
  }, true);

  // Direct evidence-card action: first open that card's drawer so the exact
  // source/chunk metadata is available, then route through the same context path.
  document.addEventListener('click', event => {
    const action = event.target.closest?.('[data-evidence-action="transcript"]');
    if (!action) return;
    event.preventDefault();
    event.stopPropagation();
    event.stopImmediatePropagation();
    const card = action.closest('.evidence-card');
    if (!card) return;
    card.click();
    setTimeout(() => {
      routeDrawerToTranscript().catch(error => {
        if ($('transcriptList')) $('transcriptList').innerHTML = `<div class="answer-card refusal">${esc(error.message)}</div>`;
      });
    }, 0);
  }, true);
})();
