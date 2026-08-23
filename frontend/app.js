(() => {
  const state = {
    sources: [],
    selected: new Set(),
    transcriptSelected: new Set(),
    transcriptInitialized: false,
    batch: null,
    importBusy: false,
    audioJobs: [],
    audioTimer: null,
    focusedJobId: null,
    ingestWorkers: 1,
    clip: { start: 0, end: 0, sourceKey: null, chunkId: null },
    capturedEtaJobs: new Set(),
  };

  const $ = (id) => document.getElementById(id);
  const ext = (name) => (name.split('.').pop() || '').toLowerCase();
  const stem = (name) => name.replace(/\.[^.]+$/, '');
  const transcriptExts = new Set(['json', 'srt', 'vtt', 'txt']);
  const audioExts = new Set(['mp3', 'wav', 'm4a', 'flac', 'ogg', 'aac']);
  const stageKeys = ['normalize', 'transcribe', 'speakers', 'chunk', 'embed'];
  const ETA_STORAGE_KEY = 'audio-search-local-rtf-v1';
  const HOME_MARKUP = '<div class="empty-orb">⌁</div><h2>Ask across hours of audio in seconds.</h2><p>Answers stay traceable to transcript chunks, speakers, timestamps, and the original recording when audio is available.</p>';

  window.__AUDIO_SEARCH_FRONTEND_BUILD__ = '20260822-duplicate-guard';

  const esc = (value = '') => String(value).replace(/[&<>'"]/g, c => ({
    '&': '&amp;', '<': '&lt;', '>': '&gt;', "'": '&#39;', '"': '&quot;'
  }[c]));

  async function api(path, options = {}) {
    const response = await fetch(path, options);
    if (!response.ok) {
      let message = `${response.status} ${response.statusText}`;
      try { message = (await response.json()).detail || message; } catch (_) {}
      throw new Error(message);
    }
    return response.json();
  }

  const formatTime = (value) => {
    if (value == null || Number(value) < 0) return 'Untimed';
    const total = Math.floor(Number(value));
    const h = Math.floor(total / 3600);
    const m = Math.floor((total % 3600) / 60);
    const s = total % 60;
    return h
      ? `${h}:${String(m).padStart(2, '0')}:${String(s).padStart(2, '0')}`
      : `${m}:${String(s).padStart(2, '0')}`;
  };

  const formatDuration = (value) => {
    if (!Number.isFinite(value) || value <= 0) return null;
    const total = Math.round(value);
    const h = Math.floor(total / 3600);
    const m = Math.floor((total % 3600) / 60);
    if (h) return `${h}h ${m}m`;
    return `${Math.max(1, m)} min`;
  };

  const bytes = (value) => {
    if (!value) return '0 B';
    const units = ['B', 'KB', 'MB', 'GB'];
    const i = Math.min(Math.floor(Math.log(value) / Math.log(1024)), units.length - 1);
    return `${(value / Math.pow(1024, i)).toFixed(i > 1 ? 1 : 0)} ${units[i]}`;
  };

  function showHome() {
    const answer = $('answerState');
    if (!answer) return;
    answer.className = 'answer-state empty-state';
    answer.innerHTML = HOME_MARKUP;
    updateHomeLayout();
  }

  function updateHomeLayout() {
    document.querySelector('.ask-stage')?.classList.toggle(
      'home-layout',
      $('answerState')?.classList.contains('empty-state') || false,
    );
  }

  function setView(view, { loadTranscript = true } = {}) {
    document.querySelectorAll('.nav-item[data-view]').forEach(item => {
      item.classList.toggle('active', item.dataset.view === view);
    });
    $('askView')?.classList.toggle('view-active', view === 'ask');
    $('transcriptView')?.classList.toggle('view-active', view === 'transcript');
    if ($('viewEyebrow')) $('viewEyebrow').textContent = view === 'ask' ? 'Search across your library' : 'Human-verifiable source of truth';
    if ($('viewTitle')) $('viewTitle').textContent = view === 'ask' ? 'Ask your recordings' : 'Audit the transcript';
    if (view === 'transcript' && loadTranscript) loadAuditTranscript();
  }

  function sourceByKey(key) {
    return state.sources.find(source => source.source_key === key || source.source_id === key) || null;
  }

  function sourceForChunk(chunk) {
    return sourceByKey(chunk?.source_key) || sourceByKey(chunk?.source_id);
  }

  function updateSelectedSummary() {
    const count = state.selected.size;
    if (!state.sources.length || count === state.sources.length) $('selectedSourceText').textContent = 'All recordings';
    else if (!count) $('selectedSourceText').textContent = 'No sources selected';
    else if (count === 1) $('selectedSourceText').textContent = state.sources.find(s => state.selected.has(s.source_key))?.display_name || '1 source';
    else $('selectedSourceText').textContent = `${count} of ${state.sources.length} sources`;
  }

  function progressBar(percent, active = false) {
    const pct = Math.max(0, Math.min(100, Number(percent) || 0));
    return `<div class="sidebar-progress ${active ? 'indeterminate' : ''}"><span style="width:${pct}%"></span></div>`;
  }

  function jobStageText(job) {
    if (job.status === 'queued') return 'Queued';
    if (job.status === 'cancelling') return 'Cancelling';
    if (job.status === 'cancelled') return 'Cancelled';
    if (job.status === 'failed') return 'Failed';
    if (job.status === 'complete') return 'Ready';
    const labels = {
      normalize: 'Normalizing audio',
      transcribe: 'Transcribing',
      speakers: 'Identifying speakers',
      chunk: 'Building chunks',
      embed: 'Generating embeddings',
    };
    return labels[job.active_stage] || 'Starting Dagster run';
  }

  function openProcessingJob(jobId) {
    const job = state.audioJobs.find(item => item.job_id === jobId);
    if (!job || ['complete', 'cancelled'].includes(job.status) || !state.batch?.started) return;
    state.focusedJobId = jobId;
    $('importModal').hidden = false;
    setProcessingModal();
  }

  function renderSources() {
    const activeJobs = state.audioJobs.filter(job => ['queued', 'running', 'cancelling'].includes(job.status));
    const failedJobs = state.audioJobs.filter(job => job.status === 'failed');
    const showGroups = activeJobs.length > 0 || failedJobs.length > 0 || state.importBusy;

    const jobHtml = activeJobs.map(job => `
      <div class="source-row processing-source" data-job-id="${esc(job.job_id || '')}" title="Open processing details for ${esc(job.display_name || job.plan?.name || 'Audio source')}">
        <div class="job-dot"></div>
        <div class="source-copy">
          <div class="source-name">${esc(job.display_name || job.plan?.name || 'Audio source')}</div>
          <div class="source-meta">${esc(jobStageText(job))}${job.status === 'running' ? ` · ${Math.round(job.overall_progress || 0)}%` : ''}</div>
          ${progressBar(job.overall_progress || 0, job.status === 'running' || job.status === 'cancelling')}
        </div>
      </div>`).join('');

    const failedHtml = failedJobs.map(job => `
      <div class="source-row processing-source" data-job-id="${esc(job.job_id || '')}" title="Open failure details for ${esc(job.display_name || job.plan?.name || 'Audio source')}">
        <div class="job-dot failed"></div>
        <div class="source-copy">
          <div class="source-name">${esc(job.display_name || job.plan?.name || 'Audio source')}</div>
          <div class="source-meta">Failed</div>
        </div>
      </div>`).join('');

    const sourceHtml = state.sources.map(source => `
      <label class="source-row" title="${esc(source.display_name)}">
        <input class="source-check" type="checkbox" data-key="${esc(source.source_key)}" ${state.selected.has(source.source_key) ? 'checked' : ''}>
        <div class="source-copy">
          <div class="source-name">${esc(source.display_name)}</div>
          <div class="source-meta">${source.chunk_count} chunks · ${source.has_audio ? 'audio + transcript' : 'transcript'}</div>
        </div>
      </label>`).join('');

    $('sourceList').innerHTML = showGroups
      ? `${activeJobs.length || state.importBusy ? `<div class="section-label">Processing</div>${jobHtml}` : ''}${failedJobs.length ? `<div class="section-label">Failed</div>${failedHtml}` : ''}<div class="section-label">Ready</div>${sourceHtml}`
      : sourceHtml;

    document.querySelectorAll('[data-job-id]').forEach(row => row.addEventListener('click', () => openProcessingJob(row.dataset.jobId)));
    document.querySelectorAll('.source-check').forEach(input => input.addEventListener('change', () => {
      input.checked ? state.selected.add(input.dataset.key) : state.selected.delete(input.dataset.key);
      updateSelectedSummary();
    }));
  }

  function normalizeTranscriptSelection() {
    const valid = new Set(state.sources.map(source => source.source_key));
    state.transcriptSelected = new Set([...state.transcriptSelected].filter(key => valid.has(key)));
    if (!state.transcriptInitialized && state.sources.length) {
      state.transcriptSelected = new Set([state.sources[0].source_key]);
      state.transcriptInitialized = true;
    }
  }

  function renderAuditSourcePicker() {
    normalizeTranscriptSelection();
    const button = $('transcriptSourceButton');
    const options = $('transcriptSourceOptions');
    if (!button || !options) return;
    const selected = state.sources.filter(source => state.transcriptSelected.has(source.source_key));
    const value = button.querySelector('.picker-value');
    if (!selected.length) value.textContent = 'Select sources';
    else if (selected.length === state.sources.length && state.sources.length > 1) value.textContent = 'All sources';
    else if (selected.length === 1) value.textContent = selected[0].display_name;
    else value.textContent = `${selected.length} sources`;

    options.innerHTML = state.sources.map(source => `
      <label class="multi-select-option">
        <input type="checkbox" data-transcript-key="${esc(source.source_key)}" ${state.transcriptSelected.has(source.source_key) ? 'checked' : ''}>
        <span>${esc(source.display_name)}</span>
      </label>`).join('');

    options.querySelectorAll('[data-transcript-key]').forEach(input => input.addEventListener('change', () => {
      input.checked ? state.transcriptSelected.add(input.dataset.transcriptKey) : state.transcriptSelected.delete(input.dataset.transcriptKey);
      renderAuditSourcePicker();
    }));
  }

  async function loadSources() {
    const previousKeys = new Set(state.sources.map(source => source.source_key));
    const oldSelected = new Set(state.selected);
    const initialLoad = state.sources.length === 0;
    state.sources = (await api('/sources')).sources || [];
    state.selected = new Set();
    state.sources.forEach(source => {
      const isNew = !previousKeys.has(source.source_key);
      if (initialLoad || isNew || oldSelected.has(source.source_key)) state.selected.add(source.source_key);
    });
    normalizeTranscriptSelection();
    renderSources();
    renderAuditSourcePicker();
    updateSelectedSummary();
    $('connectionBadge')?.classList.add('online');
    if ($('connectionBadge')) $('connectionBadge').innerHTML = '<span></span>API ready';
  }

  function selectAllLibrarySources() {
    state.selected = new Set(state.sources.map(source => source.source_key));
    renderSources();
    updateSelectedSummary();
  }

  function clearLibrarySources() {
    state.selected.clear();
    renderSources();
    updateSelectedSummary();
  }

  function renderAuditChunks(chunks, focusKey = null) {
    const list = $('transcriptList');
    if (!list) return;
    list.innerHTML = chunks.map((chunk, index) => {
      const source = sourceForChunk(chunk);
      const key = `${chunk.source_key || source?.source_key || chunk.source_id}:${chunk.chunk_id}`;
      return `<article class="chunk-row ${focusKey === key ? 'context-focus' : ''}" data-audit-chunk="${index}">
        <div class="chunk-time">${formatTime(chunk.start)}</div>
        <div class="chunk-text">${esc(chunk.speaker_text || chunk.text || '')}</div>
        <div class="chunk-id"><div>chunk ${esc(chunk.chunk_id)}</div><div class="chunk-source">${esc(source?.display_name || chunk.source_id || '')}</div></div>
      </article>`;
    }).join('') || '<div class="empty-state"><h2>No matching chunks</h2><p>Try a different query or source selection.</p></div>';

    list.querySelectorAll('[data-audit-chunk]').forEach(row => {
      row.addEventListener('click', () => openEvidence(chunks[Number(row.dataset.auditChunk)]));
    });
    if (focusKey) setTimeout(() => list.querySelector('.context-focus')?.scrollIntoView({ behavior: 'smooth', block: 'center' }), 0);
  }

  async function showAuditContext(sourceKey, chunkId, radius = 3) {
    const source = sourceByKey(sourceKey);
    if (!source) throw new Error('Evidence source is no longer available.');
    state.transcriptSelected = new Set([source.source_key]);
    renderAuditSourcePicker();
    $('transcriptSearch').value = '';
    setView('transcript', { loadTranscript: false });
    $('transcriptList').innerHTML = '<div class="loading-card"><div class="loading-line"></div><div class="loading-line"></div></div>';

    const data = await api(`/sources/${encodeURIComponent(source.source_key)}/chunks?limit=2000`);
    const all = data.chunks || [];
    const index = all.findIndex(chunk => Number(chunk.chunk_id) === Number(chunkId));
    if (index < 0) {
      $('transcriptMeta').textContent = `Chunk ${chunkId} was not found in ${source.display_name}.`;
      $('transcriptList').innerHTML = '';
      return;
    }
    const start = Math.max(0, index - radius);
    const end = Math.min(all.length, index + radius + 1);
    const context = all.slice(start, end);
    $('transcriptMeta').textContent = `${context.length} chunks · ${source.display_name} · context around chunk ${chunkId}`;
    renderAuditChunks(context, `${source.source_key}:${chunkId}`);
  }

  async function loadAuditTranscript() {
    try {
      const keys = [...state.transcriptSelected];
      if (!keys.length) {
        $('transcriptMeta').textContent = 'Select at least one source.';
        $('transcriptList').innerHTML = '';
        return;
      }
      const raw = $('transcriptSearch').value.trim();
      const numeric = raw.match(/^(?:chunk\s*)?#?(\d+)$/i);
      const chunkId = numeric ? Number(numeric[1]) : null;
      $('transcriptList').innerHTML = '<div class="loading-card"><div class="loading-line"></div><div class="loading-line"></div></div>';

      let chunks = [];
      let mode = 'full transcript';
      if (chunkId != null) {
        const results = await Promise.all(keys.map(async key => {
          const params = new URLSearchParams({ limit: '100', chunk_id: String(chunkId) });
          const data = await api(`/sources/${encodeURIComponent(key)}/chunks?${params}`);
          return data.chunks || [];
        }));
        chunks = results.flat();
        mode = 'chunk lookup';
      } else if (raw) {
        const [hybridResult, exactResults] = await Promise.all([
          api('/search', {
            method: 'POST', headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ query: raw, top_k: 20, source_keys: keys, top_k_per_source: 5 }),
          }),
          Promise.all(keys.map(async key => {
            const params = new URLSearchParams({ limit: '50', query: raw });
            const data = await api(`/sources/${encodeURIComponent(key)}/chunks?${params}`);
            return data.chunks || [];
          })),
        ]);
        const merged = [...exactResults.flat(), ...(hybridResult.results || [])];
        const seen = new Set();
        chunks = merged.filter(chunk => {
          const id = `${chunk.source_key || chunk.source_id}:${chunk.chunk_id}`;
          if (seen.has(id)) return false;
          seen.add(id);
          return true;
        }).slice(0, 50);
        mode = 'hybrid BM25 + dense retrieval + exact transcript matches';
      } else {
        const results = await Promise.all(keys.map(async key => {
          const data = await api(`/sources/${encodeURIComponent(key)}/chunks?limit=800`);
          return data.chunks || [];
        }));
        chunks = results.flat();
      }
      $('transcriptMeta').textContent = `${chunks.length} chunk${chunks.length === 1 ? '' : 's'} · ${keys.length} source${keys.length === 1 ? '' : 's'} · ${mode}`;
      renderAuditChunks(chunks);
    } catch (error) {
      $('transcriptList').innerHTML = `<div class="answer-card refusal">${esc(error.message)}</div>`;
    }
  }

  function configureClip(source, chunk) {
    state.clip.start = Math.max(0, Number(chunk.start) || 0);
    state.clip.end = Math.max(state.clip.start, Number(chunk.end) || state.clip.start);
    state.clip.sourceKey = source.source_key;
    state.clip.chunkId = chunk.chunk_id;
    const duration = Math.max(0, state.clip.end - state.clip.start);
    $('clipSeek').min = '0';
    $('clipSeek').max = String(Math.max(duration, 0.05));
    $('clipSeek').value = '0';
    $('clipElapsed').textContent = '0:00';
    $('clipDuration').textContent = formatTime(duration);
    $('audioDockTitle').textContent = source.display_name;
    $('audioDockSubtitle').textContent = `Chunk ${chunk.chunk_id} · ${formatTime(state.clip.start)}–${formatTime(state.clip.end)}`;
  }

  function playSourceAudio(source, chunk) {
    if (!source?.has_audio) return;
    const player = $('audioPlayer');
    configureClip(source, chunk);
    $('audioDock').hidden = false;
    const expectedSrc = `/sources/${encodeURIComponent(source.source_key)}/audio`;
    const startPlayback = () => {
      player.currentTime = state.clip.start;
      player.play().catch(() => {});
    };
    if (player.dataset.sourceKey !== source.source_key) {
      player.dataset.sourceKey = source.source_key;
      player.src = expectedSrc;
      player.addEventListener('loadedmetadata', startPlayback, { once: true });
      player.load();
    } else {
      startPlayback();
    }
  }

  function syncClipPlayer() {
    const player = $('audioPlayer');
    if (!state.clip.sourceKey) return;
    const duration = Math.max(0, state.clip.end - state.clip.start);
    let relative = Math.max(0, player.currentTime - state.clip.start);
    if (duration && player.currentTime >= state.clip.end) {
      player.pause();
      player.currentTime = state.clip.end;
      relative = duration;
    }
    $('clipSeek').value = String(Math.min(relative, duration || relative));
    $('clipElapsed').textContent = formatTime(Math.min(relative, duration || relative));
    $('clipToggle').textContent = player.paused ? '▶' : '❚❚';
  }

  function jumpClip(delta) {
    const player = $('audioPlayer');
    if (!state.clip.sourceKey) return;
    player.currentTime = Math.max(state.clip.start, Math.min(state.clip.end, player.currentTime + delta));
    syncClipPlayer();
  }

  function openEvidence(chunk) {
    if (!chunk) return;
    const source = sourceForChunk(chunk);
    $('drawerTitle').textContent = source?.display_name || chunk.source_id || 'Source';
    $('drawerBody').innerHTML = `
      <div class="drawer-meta">
        <span class="meta-chip">Chunk ${esc(chunk.chunk_id)}</span>
        <span class="meta-chip">${formatTime(chunk.start)}${Number(chunk.end) >= 0 ? `–${formatTime(chunk.end)}` : ''}</span>
        <span class="meta-chip">${source?.has_audio ? 'Audio attached' : 'Transcript only'}</span>
      </div>
      <div class="drawer-transcript">${esc(chunk.speaker_text || chunk.text || 'Transcript excerpt unavailable.')}</div>
      <div class="drawer-actions">
        ${source?.has_audio ? '<button id="drawerPlayEvidence" class="primary-button" type="button">▶ Play audio</button>' : ''}
        <button id="drawerViewTranscript" class="secondary-button" type="button">View in transcript</button>
      </div>`;
    $('evidenceDrawer').classList.add('open');
    $('drawerPlayEvidence')?.addEventListener('click', () => playSourceAudio(source, chunk));
    $('drawerViewTranscript')?.addEventListener('click', () => {
      $('evidenceDrawer').classList.remove('open');
      if (!source) return;
      showAuditContext(source.source_key, chunk.chunk_id).catch(error => {
        $('transcriptList').innerHTML = `<div class="answer-card refusal">${esc(error.message)}</div>`;
      });
    });
  }

  async function hydrateCitations(answer, searchResults) {
    const lookup = new Map((searchResults || []).map(chunk => [`${chunk.source_id}:${chunk.chunk_id}`, chunk]));
    return Promise.all((answer.citations || []).map(async citation => {
      const existing = lookup.get(citation.citation_id);
      if (existing) return existing;
      const source = state.sources.find(item => item.source_id === citation.source_id || item.source_key === citation.source_id);
      if (!source) return citation;
      try {
        const params = new URLSearchParams({ limit: '10', chunk_id: String(citation.chunk_id) });
        const data = await api(`/sources/${encodeURIComponent(source.source_key)}/chunks?${params}`);
        return data.chunks?.[0] || citation;
      } catch (_) {
        return citation;
      }
    }));
  }

  async function askQuestion(question) {
    const query = question.trim();
    if (!query) {
      showHome();
      return;
    }
    if (!state.selected.size) {
      $('answerState').className = 'answer-state';
      $('answerState').innerHTML = '<div class="answer-card refusal"><div class="answer-kicker">No sources selected</div><div class="answer-text">Select at least one recording before asking a question.</div></div>';
      updateHomeLayout();
      return;
    }
    const sourceKeys = state.selected.size === state.sources.length ? null : [...state.selected];
    const payload = { query, top_k: 6, source_keys: sourceKeys, top_k_per_source: 3 };
    $('askButton').disabled = true;
    $('answerState').className = 'answer-state';
    $('answerState').innerHTML = '<div class="loading-card"><div class="loading-line"></div><div class="loading-line"></div><div class="loading-line"></div></div>';
    updateHomeLayout();

    const [searchResult, answerResult] = await Promise.allSettled([
      api('/search', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload) }),
      api('/answer', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload) }),
    ]);
    $('askButton').disabled = false;

    if (answerResult.status === 'rejected') {
      $('answerState').innerHTML = `<div class="answer-card refusal"><div class="answer-kicker">LLM unavailable</div><div class="answer-text">${esc(answerResult.reason.message)}. Retrieval remains available through the API.</div></div>`;
      return;
    }

    const answer = answerResult.value;
    const search = searchResult.status === 'fulfilled' ? searchResult.value : { results: [] };
    const cited = await hydrateCitations(answer, search.results || []);
    const clarify = !answer.answerable && String(answer.answer || '').startsWith("I don't understand the question well enough to answer it reliably.");

    let html = `<div class="answer-card ${answer.answerable ? '' : 'refusal'}"><div class="answer-kicker">${answer.answerable ? 'Grounded answer' : clarify ? 'Clarify question' : 'Not enough evidence'}</div><div class="answer-text">${esc(answer.answer)}</div></div>`;
    if (cited.length) {
      html += `<div class="evidence-section"><div class="evidence-title">Evidence · ${cited.length}</div><div class="evidence-grid">${cited.map((chunk, i) => {
        const source = sourceForChunk(chunk);
        return `<article class="evidence-card" data-evidence-index="${i}">
          <div class="evidence-card-head"><div class="evidence-source">${esc(source?.display_name || chunk.source_id || 'Source')}</div><div class="evidence-time">${formatTime(chunk.start)}${Number(chunk.end) >= 0 ? `–${formatTime(chunk.end)}` : ''}</div></div>
          <div class="evidence-excerpt">${esc(chunk.speaker_text || chunk.text || 'Open to inspect source evidence.')}</div>
          <div class="evidence-direct-actions">
            <button type="button" data-evidence-action="transcript">View transcript</button>
            ${source?.has_audio ? '<button type="button" data-evidence-action="play">▶ Play</button>' : ''}
          </div>
        </article>`;
      }).join('')}</div></div>`;
    }
    $('answerState').innerHTML = html;
    updateHomeLayout();

    document.querySelectorAll('[data-evidence-index]').forEach(card => {
      const chunk = cited[Number(card.dataset.evidenceIndex)];
      card.addEventListener('click', event => {
        if (event.target.closest('[data-evidence-action]')) return;
        openEvidence(chunk);
      });
      card.querySelector('[data-evidence-action="transcript"]')?.addEventListener('click', () => {
        const source = sourceForChunk(chunk);
        if (!source) return;
        showAuditContext(source.source_key, chunk.chunk_id).catch(error => {
          $('transcriptList').innerHTML = `<div class="answer-card refusal">${esc(error.message)}</div>`;
        });
      });
      card.querySelector('[data-evidence-action="play"]')?.addEventListener('click', () => {
        const source = sourceForChunk(chunk);
        if (source) playSourceAudio(source, chunk);
      });
    });
  }

  function makePlan(audio, transcript, suffix = '') {
    const baseName = stem((transcript || audio).name) + suffix;
    if (audio && transcript) {
      return { id: `${baseName}-${Math.random().toString(36).slice(2, 8)}`, name: baseName, mode: 'both', label: 'Audio + supplied transcript', audio, transcript, nodes: [['Validate', 'audio + transcript'], ['Parse', 'timestamps / speakers'], ['Attach audio', 'retain source audio'], ['Chunk', 'retrieval units'], ['Embed', 'searchable corpus']], duration: null };
    }
    if (transcript) {
      return { id: `${baseName}-${Math.random().toString(36).slice(2, 8)}`, name: baseName, mode: 'transcript', label: 'Supplied transcript', audio: null, transcript, nodes: [['Validate', 'transcript supplied'], ['Parse', 'timestamps / speakers'], ['Skip ASR', 'Whisper not needed', 'skip'], ['Chunk', 'retrieval units'], ['Embed', 'searchable corpus']], duration: null };
    }
    return { id: `${baseName}-${Math.random().toString(36).slice(2, 8)}`, name: baseName, mode: 'audio', label: 'Raw audio pipeline', audio, transcript: null, nodes: [['Normalize', '16 kHz mono'], ['Transcribe', 'faster-whisper'], ['Speakers', 'pyannote + roles'], ['Chunk', 'retrieval units'], ['Embed', 'searchable corpus']], duration: null };
  }

  function plansForFiles(files) {
    const supported = files.filter(file => transcriptExts.has(ext(file.name)) || audioExts.has(ext(file.name)));
    const groups = new Map();
    for (const file of supported) {
      const key = stem(file.name).toLowerCase();
      if (!groups.has(key)) groups.set(key, { audios: [], transcripts: [] });
      const group = groups.get(key);
      if (audioExts.has(ext(file.name))) group.audios.push(file);
      else group.transcripts.push(file);
    }
    const plans = [];
    for (const group of groups.values()) {
      const pairs = Math.min(group.audios.length, group.transcripts.length);
      for (let i = 0; i < pairs; i += 1) plans.push(makePlan(group.audios[i], group.transcripts[i]));
      for (let i = pairs; i < group.audios.length; i += 1) plans.push(makePlan(group.audios[i], null, group.audios.length > 1 ? ` ${i + 1}` : ''));
      for (let i = pairs; i < group.transcripts.length; i += 1) plans.push(makePlan(null, group.transcripts[i], group.transcripts.length > 1 ? ` ${i + 1}` : ''));
    }
    return { supported, plans };
  }

  function normalizeSourceName(value) {
    return String(value || '')
      .trim()
      .toLowerCase()
      .replace(/_/g, ' ')
      .replace(/\s+/g, ' ');
  }

  function findDuplicateSources(plans) {
    const readyAudioSources = state.sources.filter(source => source.has_audio);
    const matches = [];
    for (const plan of plans) {
      if (!plan.audio) continue;
      const incoming = normalizeSourceName(stem(plan.audio.name));
      const source = readyAudioSources.find(item => normalizeSourceName(item.display_name) === incoming);
      if (!source) continue;
      matches.push({
        planId: plan.id,
        planName: plan.name,
        sourceKey: source.source_key,
        sourceName: source.display_name,
      });
    }
    return matches;
  }

  function renderDag(plan, stages = null) {
    $('dagGraph').innerHTML = plan.nodes.map((node, index) => {
      const skipped = node[2] === 'skip';
      const stage = stages && plan.mode === 'audio' ? stages[stageKeys[index]] : null;
      const done = !skipped && stage?.status === 'complete';
      const running = !skipped && stage?.status === 'running';
      const hasNumericProgress = Number.isFinite(stage?.progress);
      const pct = done ? 100 : hasNumericProgress ? Math.round(stage.progress) : 0;
      const statusText = running ? `${hasNumericProgress ? `${pct}% · ` : 'Running · '}${esc(node[1])}` : esc(node[1]);
      return `<div class="dag-node ${done ? 'complete' : ''} ${running ? 'active' : ''} ${skipped ? 'skipped' : ''}"><div class="node-dot">${done ? '✓' : skipped ? '↷' : index + 1}</div><div class="node-label">${esc(node[0])}</div><div class="node-progress ${running && !hasNumericProgress ? 'indeterminate' : ''}"><span style="width:${pct}%"></span></div><div class="node-detail">${statusText}</div></div>`;
    }).join('');
  }

  function readAudioDuration(plan) {
    if (!plan.audio) return Promise.resolve(null);
    return new Promise(resolve => {
      const url = URL.createObjectURL(plan.audio);
      const audio = new Audio();
      audio.preload = 'metadata';
      audio.onloadedmetadata = () => {
        plan.duration = Number.isFinite(audio.duration) ? audio.duration : null;
        URL.revokeObjectURL(url);
        resolve(plan.duration);
      };
      audio.onerror = () => { URL.revokeObjectURL(url); resolve(null); };
      audio.src = url;
    });
  }

  function renderFileSummary() {
    if (!state.batch) return;
    $('fileSummary').innerHTML = state.batch.plans.map(plan => {
      const files = [plan.audio, plan.transcript].filter(Boolean);
      const kind = plan.mode === 'both' ? 'Audio + transcript' : plan.mode === 'audio' ? 'Audio' : 'Transcript';
      const durationText = plan.duration ? ` · ${formatDuration(plan.duration)}` : '';
      const size = files.reduce((sum, file) => sum + (file.size || 0), 0);
      const duplicate = (state.batch.duplicateMatches || []).find(item => item.planId === plan.id);
      const duplicateLabel = duplicate
        ? `<div class="file-kind" style="margin-top:4px;color:#9a5b00">Already in library · ${esc(duplicate.sourceName)}</div>`
        : '';
      return `<div class="file-row"><div><div class="file-kind">${kind}</div><div class="file-name">${esc(plan.name)}</div>${duplicateLabel}</div><div class="file-size">${bytes(size)}${durationText}</div></div>`;
    }).join('');
  }

  function median(values) {
    if (!values.length) return null;
    const sorted = [...values].sort((a, b) => a - b);
    const middle = Math.floor(sorted.length / 2);
    return sorted.length % 2 ? sorted[middle] : (sorted[middle - 1] + sorted[middle]) / 2;
  }

  function loadStoredRtfSamples() {
    try {
      const parsed = JSON.parse(localStorage.getItem(ETA_STORAGE_KEY) || '[]');
      return Array.isArray(parsed) ? parsed.filter(value => Number.isFinite(value) && value > 0.05 && value < 5).slice(-20) : [];
    } catch (_) { return []; }
  }

  function storeRtfSample(value) {
    if (!Number.isFinite(value) || value <= 0.05 || value >= 5) return;
    const samples = loadStoredRtfSamples();
    samples.push(value);
    try { localStorage.setItem(ETA_STORAGE_KEY, JSON.stringify(samples.slice(-20))); } catch (_) {}
  }

  function learnedRtfRange() {
    const center = median(loadStoredRtfSamples());
    if (center == null) return { low: 0.5, high: 1.0, learned: false };
    return { low: Math.max(0.1, center * 0.82), high: Math.max(center * 1.18, center * 0.82 + 0.05), learned: true };
  }

  function humanEta(seconds) {
    if (!Number.isFinite(seconds) || seconds <= 0) return '<1 min';
    if (seconds < 60) return `${Math.max(1, Math.round(seconds / 5) * 5)} sec`;
    const minutes = Math.max(1, Math.round(seconds / 60));
    if (minutes < 60) return `${minutes} min`;
    const hours = Math.floor(minutes / 60);
    const remainder = minutes % 60;
    return remainder ? `${hours}h ${remainder}m` : `${hours}h`;
  }

  function captureCompletedEtaSamples() {
    for (const job of state.audioJobs) {
      if (job.status !== 'complete' || state.capturedEtaJobs.has(job.job_id)) continue;
      const duration = Number(job.plan?.duration);
      const started = Number(job.started_at);
      const finished = Number(job.finished_at);
      if (duration > 0 && Number.isFinite(started) && Number.isFinite(finished) && finished > started) {
        storeRtfSample((finished - started) / duration);
        state.capturedEtaJobs.add(job.job_id);
      }
    }
  }

  function estimateBatchSeconds(factor, started) {
    const plans = state.batch?.plans || [];
    const rawPlans = plans.filter(plan => plan.mode === 'audio' && Number(plan.duration) > 0);
    if (!rawPlans.length) return null;
    if (!started) {
      const total = rawPlans.reduce((sum, plan) => sum + plan.duration * factor, 0);
      return total / Math.max(1, Math.min(state.ingestWorkers, rawPlans.length));
    }
    const remaining = rawPlans.map(plan => {
      const job = state.audioJobs.find(item => item.plan?.id === plan.id);
      if (!job) return plan.duration * factor;
      if (['complete', 'failed', 'cancelled'].includes(job.status)) return 0;
      if (job.status === 'running') {
        const progress = Math.max(0, Math.min(99, Number(job.overall_progress) || 0)) / 100;
        return plan.duration * factor * (1 - progress);
      }
      return plan.duration * factor;
    });
    return remaining.reduce((sum, value) => sum + value, 0) / Math.max(1, Math.min(state.ingestWorkers, remaining.filter(Boolean).length || 1));
  }

  function renderEta() {
    if (!state.batch) return;
    captureCompletedEtaSamples();
    const range = learnedRtfRange();
    const low = estimateBatchSeconds(range.low, state.batch.started);
    const high = estimateBatchSeconds(range.high, state.batch.started);
    if (low == null || high == null) {
      $('etaValue').textContent = state.batch.plans.some(plan => plan.mode === 'audio') ? 'Estimating…' : 'A few seconds';
      return;
    }
    $('etaValue').textContent = high <= 1 ? 'Finishing…' : `≈ ${humanEta(low)}–${humanEta(high)}${state.batch.started ? ' remaining' : ' total'}`;
    $('etaValue').title = range.learned ? 'Estimate adapts to completed runs on this machine.' : 'Initial heuristic; it adapts after completed runs on this machine.';
  }

  function previewBatch() {
    const batch = state.batch;
    if (!batch) return;
    const first = batch.plans[0];
    const duplicates = batch.duplicateMatches || [];
    const duplicateWarning = duplicates.length > 0;
    $('importTitle').textContent = duplicateWarning
      ? (duplicates.length === 1 ? 'Already in library' : `${duplicates.length} sources already in library`)
      : (batch.plans.length === 1 ? 'Review source' : `Review ${batch.plans.length} sources`);
    renderFileSummary();
    $('ingestionPathBadge').textContent = batch.plans.length === 1 ? first.label : `${batch.plans.length} sources`;
    $('overallProgress').textContent = '0%';
    $('dagStatus').textContent = duplicateWarning ? 'Reprocessing requires confirmation' : 'Waiting for confirmation';
    renderDag(first);
    $('importMessage').hidden = !duplicateWarning;
    $('importMessage').className = 'import-message';
    $('importMessage').textContent = duplicateWarning
      ? (duplicates.length === 1
          ? `It looks like ${duplicates[0].sourceName} is already in your library. Reprocessing will run the ingestion pipeline again and create a new source. Continue?`
          : `${duplicates.length} of these audio sources already appear in your library. Reprocessing will run the ingestion pipeline again and create new sources. Continue?`)
      : '';
    $('cancelImport').hidden = false;
    $('cancelImport').textContent = 'Cancel';
    $('cancelImport').disabled = false;
    $('addImportFiles').hidden = duplicateWarning;
    $('importAction').textContent = duplicateWarning ? 'Reprocess anyway' : 'Process';
    $('importAction').disabled = false;
    renderEta();
  }

  function currentAudioJob() {
    if (state.focusedJobId) {
      const focused = state.audioJobs.find(job => job.job_id === state.focusedJobId && !['complete', 'cancelled'].includes(job.status));
      if (focused) return focused;
      state.focusedJobId = null;
    }
    return state.audioJobs.find(job => ['running', 'cancelling'].includes(job.status))
      || state.audioJobs.find(job => job.status === 'queued')
      || state.audioJobs.find(job => job.status === 'failed')
      || null;
  }

  function batchOverall() {
    const total = state.batch?.plans.length || 1;
    const imported = state.batch?.importedCount || 0;
    const audioPoints = state.audioJobs.reduce((sum, job) => sum + Number(job.overall_progress || 0), 0);
    return Math.round((audioPoints + imported * 100) / total);
  }

  function setProcessingModal() {
    const batch = state.batch;
    if (!batch) return;
    const job = currentAudioJob();
    const totalItems = batch.plans.length;
    $('importTitle').textContent = totalItems === 1 ? 'Processing source' : `Processing ${totalItems} sources`;
    renderFileSummary();
    $('addImportFiles').hidden = true;
    $('importAction').textContent = 'Minimize';
    $('cancelImport').hidden = false;
    $('cancelImport').textContent = batch.cancelling ? 'Cancelling…' : 'Cancel processing';
    $('cancelImport').disabled = Boolean(batch.cancelling);
    $('overallProgress').textContent = `${batchOverall()}%`;
    $('importMessage').hidden = false;
    renderEta();

    if (job) {
      $('ingestionPathBadge').textContent = totalItems === 1 ? job.plan.label : `${totalItems} sources`;
      $('dagStatus').textContent = totalItems > 1 ? `${jobStageText(job)} · ${job.display_name || job.plan.name}` : jobStageText(job);
      renderDag(job.plan, job.stages || null);
      $('importMessage').className = job.status === 'failed' ? 'import-message error' : 'import-message';
      $('importMessage').textContent = job.status === 'failed'
        ? (job.error || 'Dagster materialization failed.')
        : batch.cancelling
          ? 'Stopping current processing and cancelling queued sources…'
          : 'This is a live Dagster ingestion job. You can minimize this window and keep using the rest of the library.';
      return;
    }

    const first = batch.plans[0];
    $('ingestionPathBadge').textContent = totalItems === 1 ? first.label : `${totalItems} sources`;
    $('dagStatus').textContent = state.importBusy ? 'Importing transcript' : batch.cancelling ? 'Cancelling' : 'Processing complete';
    renderDag(first);
    $('importMessage').className = 'import-message';
    $('importMessage').textContent = state.importBusy ? 'Parsing, chunking, and embedding supplied transcript data.' : batch.cancelling ? 'Cancelling processing…' : 'Processing complete.';
  }

  async function submitAudioPlan(plan) {
    const form = new FormData();
    form.append('audio', plan.audio);
    form.append('source_name', plan.name);
    const job = await api('/ingest/audio', { method: 'POST', body: form });
    job.plan = plan;
    state.audioJobs.push(job);
    return job;
  }

  async function pollAudioJobs() {
    if (!state.audioJobs.length) return;
    let changedSources = false;
    await Promise.all(state.audioJobs.map(async job => {
      if (!job.job_id || ['failed', 'complete', 'cancelled'].includes(job.status)) return;
      try {
        const updated = await api(`/ingest/jobs/${encodeURIComponent(job.job_id)}`);
        const wasComplete = job.status === 'complete';
        Object.assign(job, updated);
        if (!wasComplete && job.status === 'complete') changedSources = true;
      } catch (error) {
        job.status = 'failed';
        job.error = error.message;
      }
    }));
    if (changedSources) await loadSources();
    else renderSources();
    captureCompletedEtaSamples();
    if (!$('importModal').hidden && state.batch?.started) setProcessingModal();
    const allDone = state.audioJobs.every(job => ['complete', 'failed', 'cancelled'].includes(job.status));
    if (allDone) state.focusedJobId = null;
    if (allDone && !state.importBusy && state.audioTimer) {
      clearInterval(state.audioTimer);
      state.audioTimer = null;
      renderSources();
    }
  }

  async function startAudioPlans(plans) {
    if (!plans.length) return;
    await Promise.all(plans.map(async plan => {
      try { await submitAudioPlan(plan); }
      catch (error) { state.audioJobs.push({ plan, display_name: plan.name, status: 'failed', overall_progress: 0, stages: null, error: error.message }); }
    }));
    renderSources();
    await pollAudioJobs();
    if (state.audioJobs.some(job => job.job_id && !['complete', 'failed', 'cancelled'].includes(job.status)) && !state.audioTimer) {
      state.audioTimer = setInterval(pollAudioJobs, 1000);
    }
  }

  async function importTranscriptPlan(plan) {
    const form = new FormData();
    form.append('transcript', plan.transcript);
    if (plan.audio) form.append('audio', plan.audio);
    form.append('source_name', plan.name);
    return api('/ingest/transcript', { method: 'POST', body: form });
  }

  async function runTranscriptPlans(plans) {
    if (!plans.length) return;
    state.importBusy = true;
    try {
      for (const plan of plans) {
        if (!state.batch?.started || state.batch?.cancelling) break;
        if (!$('importModal').hidden) setProcessingModal();
        try {
          await importTranscriptPlan(plan);
          state.batch.importedCount = (state.batch.importedCount || 0) + 1;
          await loadSources();
        } catch (error) {
          state.batch.errors = state.batch.errors || [];
          state.batch.errors.push(`${plan.name}: ${error.message}`);
        }
      }
    } finally {
      state.importBusy = false;
      if (state.batch?.cancelling) {
        state.batch = null;
        state.audioJobs = [];
        state.focusedJobId = null;
        $('importModal').hidden = true;
      } else if (!$('importModal').hidden && state.batch?.started) {
        setProcessingModal();
      }
      renderSources();
    }
  }

  async function startBatch() {
    const batch = state.batch;
    if (!batch || batch.started) return;
    batch.started = true;
    batch.cancelling = false;
    batch.importedCount = 0;
    batch.errors = [];
    state.audioJobs = [];
    state.focusedJobId = null;
    setProcessingModal();
    const rawAudioPlans = batch.plans.filter(plan => plan.mode === 'audio');
    const transcriptPlans = batch.plans.filter(plan => plan.mode !== 'audio');
    await Promise.all([startAudioPlans(rawAudioPlans), runTranscriptPlans(transcriptPlans)]);
    if (!$('importModal').hidden && state.batch?.started) setProcessingModal();
  }

  async function cancelCurrentBatch() {
    const batch = state.batch;
    if (!batch?.started || batch.cancelling) return;
    batch.cancelling = true;
    setProcessingModal();
    const activeJobs = state.audioJobs.filter(job => job.job_id && !['complete', 'failed', 'cancelled'].includes(job.status));
    try {
      await Promise.all(activeJobs.map(job => api(`/ingest/jobs/${encodeURIComponent(job.job_id)}/cancel`, { method: 'POST' })));
      for (let attempt = 0; attempt < 32; attempt += 1) {
        await pollAudioJobs();
        if (state.audioJobs.every(job => ['complete', 'failed', 'cancelled'].includes(job.status))) break;
        await new Promise(resolve => setTimeout(resolve, 250));
      }
      if (state.audioTimer) {
        clearInterval(state.audioTimer);
        state.audioTimer = null;
      }
      if (state.importBusy) {
        $('importMessage').className = 'import-message';
        $('importMessage').textContent = 'Audio processing stopped. Waiting for the current transcript import to finish safely…';
        return;
      }
      state.batch = null;
      state.audioJobs = [];
      state.focusedJobId = null;
      $('importModal').hidden = true;
      renderSources();
    } catch (error) {
      batch.cancelling = false;
      $('importMessage').className = 'import-message error';
      $('importMessage').textContent = `Could not cancel processing: ${error.message}`;
      setProcessingModal();
    }
  }

  function uniqueFiles(files) {
    const seen = new Set();
    return files.filter(file => {
      const key = `${file.name}::${file.size}::${file.lastModified}`;
      if (seen.has(key)) return false;
      seen.add(key);
      return true;
    });
  }

  function prepareImport(files, append = false) {
    const existing = append && state.batch && !state.batch.started ? state.batch.supported : [];
    const mergedFiles = uniqueFiles([...existing, ...files]);
    const { supported, plans } = plansForFiles(mergedFiles);
    if (!supported.length || !plans.length) return;
    const duplicateMatches = findDuplicateSources(plans);
    state.batch = { supported, plans, duplicateMatches, started: false, cancelling: false, importedCount: 0, errors: [] };
    state.audioJobs = [];
    state.focusedJobId = null;
    $('importModal').hidden = false;
    previewBatch();
    Promise.all(plans.filter(plan => plan.audio).map(readAudioDuration)).then(() => {
      if (state.batch?.plans === plans) {
        renderFileSummary();
        renderEta();
      }
    });
  }

  function minimizeImport() {
    $('importModal').hidden = true;
    setView('ask');
  }

  function cancelPreview() {
    state.batch = null;
    $('importModal').hidden = true;
  }

  function closeImport() {
    if (state.batch?.started) minimizeImport();
    else cancelPreview();
  }

  function pickFiles(append = false) {
    const input = document.createElement('input');
    input.type = 'file';
    input.multiple = true;
    input.accept = '.mp3,.wav,.m4a,.flac,.ogg,.aac,.json,.srt,.vtt,.txt';
    input.addEventListener('change', () => {
      if (input.files?.length) prepareImport(Array.from(input.files), append);
    }, { once: true });
    input.click();
  }

  function bindEvents() {
    $('sidebarToggle')?.addEventListener('click', () => $('app').classList.toggle('sidebar-collapsed'));
    document.querySelectorAll('.nav-item[data-view]').forEach(item => item.addEventListener('click', () => setView(item.dataset.view)));
    $('queryForm')?.addEventListener('submit', event => {
      event.preventDefault();
      askQuestion($('queryInput').value);
    });
    $('queryInput')?.addEventListener('keydown', event => {
      if (event.key !== 'Enter' || event.shiftKey) return;
      event.preventDefault();
      $('queryForm').requestSubmit();
    });
    $('clearSources')?.addEventListener('click', () => {
      $('queryInput').value = '';
      $('queryInput').focus();
    });
    $('selectAllSources')?.addEventListener('click', selectAllLibrarySources);
    $('clearAllSources')?.addEventListener('click', clearLibrarySources);
    $('transcriptSourceButton')?.addEventListener('click', event => {
      event.stopPropagation();
      $('transcriptSourceMenu').hidden = !$('transcriptSourceMenu').hidden;
    });
    $('transcriptSelectAll')?.addEventListener('click', () => {
      state.transcriptSelected = new Set(state.sources.map(source => source.source_key));
      renderAuditSourcePicker();
    });
    $('transcriptClearAll')?.addEventListener('click', () => {
      state.transcriptSelected.clear();
      renderAuditSourcePicker();
    });
    document.addEventListener('click', event => {
      if (!event.target.closest('#transcriptSourcePicker')) $('transcriptSourceMenu').hidden = true;
    });
    $('transcriptSearchButton')?.addEventListener('click', loadAuditTranscript);
    $('transcriptSearch')?.addEventListener('keydown', event => {
      if (event.key === 'Enter') { event.preventDefault(); loadAuditTranscript(); }
    });
    $('closeDrawer')?.addEventListener('click', () => $('evidenceDrawer').classList.remove('open'));

    const player = $('audioPlayer');
    player?.addEventListener('timeupdate', syncClipPlayer);
    player?.addEventListener('play', syncClipPlayer);
    player?.addEventListener('pause', syncClipPlayer);
    player?.addEventListener('loadedmetadata', syncClipPlayer);
    $('clipToggle')?.addEventListener('click', () => {
      if (!state.clip.sourceKey) return;
      if (!player.paused) { player.pause(); return; }
      if (player.currentTime >= state.clip.end - 0.05 || player.currentTime < state.clip.start) player.currentTime = state.clip.start;
      player.play().catch(() => {});
    });
    $('clipBack15')?.addEventListener('click', () => jumpClip(-15));
    $('clipForward15')?.addEventListener('click', () => jumpClip(15));
    $('clipSeek')?.addEventListener('input', event => {
      if (!state.clip.sourceKey) return;
      player.currentTime = Math.max(state.clip.start, Math.min(state.clip.end, state.clip.start + Number(event.target.value || 0)));
      syncClipPlayer();
    });
    $('closeAudio')?.addEventListener('click', () => {
      player.pause();
      $('audioDock').hidden = true;
      state.clip = { start: 0, end: 0, sourceKey: null, chunkId: null };
    });

    $('addSourceButton')?.addEventListener('click', () => pickFiles(false));
    $('topAddButton')?.addEventListener('click', () => pickFiles(false));
    $('addImportFiles')?.addEventListener('click', () => pickFiles(true));
    $('closeImport')?.addEventListener('click', closeImport);
    $('cancelImport')?.addEventListener('click', () => {
      if (state.batch?.started) cancelCurrentBatch();
      else cancelPreview();
    });
    $('importAction')?.addEventListener('click', async () => {
      if (!state.batch) return;
      if (!state.batch.started) await startBatch();
      else minimizeImport();
    });
    $('importModal')?.addEventListener('click', event => {
      if (event.target === $('importModal') && state.batch?.started) minimizeImport();
    });

    let dragDepth = 0;
    window.addEventListener('dragenter', event => { event.preventDefault(); dragDepth += 1; $('dropOverlay').classList.add('visible'); });
    window.addEventListener('dragover', event => event.preventDefault());
    window.addEventListener('dragleave', event => {
      event.preventDefault();
      dragDepth = Math.max(0, dragDepth - 1);
      if (!dragDepth) $('dropOverlay').classList.remove('visible');
    });
    window.addEventListener('drop', event => {
      event.preventDefault();
      dragDepth = 0;
      $('dropOverlay').classList.remove('visible');
      const files = Array.from(event.dataTransfer?.files || []);
      if (files.length) prepareImport(files, Boolean(state.batch && !state.batch.started && !$('importModal').hidden));
    });
  }

  async function initialize() {
    bindEvents();
    try {
      const health = await api('/health');
      state.ingestWorkers = Math.max(1, Number(health.ingest_workers) || 1);
      await loadSources();
    } catch (error) {
      if ($('connectionBadge')) $('connectionBadge').innerHTML = '<span></span>API unavailable';
      $('answerState').innerHTML = `<div class="answer-card refusal">${esc(error.message)}</div>`;
    }
    updateHomeLayout();
  }

  initialize();
})();