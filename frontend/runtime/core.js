  const state = {
    sources: [],
    selected: new Set(),
    transcriptSelected: new Set(),
    transcriptInitialized: false,
    batch: null,
    pendingAppend: null,
    importBusy: false,
    transcriptQueue: [],
    transcriptDraining: false,
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
  const terminalStatuses = new Set(['complete', 'failed', 'cancelled']);
  const liveStatuses = new Set(['queued', 'running', 'cancelling']);
  const ETA_STORAGE_KEY = 'audio-search-local-rtf-v1';
  const HOME_MARKUP = '<div class="empty-orb">⌁</div><h2>Ask across hours of audio in seconds.</h2><p>Answers stay traceable to transcript chunks, speakers, timestamps, and the original recording when audio is available.</p>';

  window.__AUDIO_SEARCH_FRONTEND_BUILD__ = '20260822-live-job-state';

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

  function audioPlanFromJob(job) {
    const name = job.display_name || stem(job.filename || 'Audio source');
    return {
      id: job.job_id || `${name}-${Math.random().toString(36).slice(2, 8)}`,
      name,
      mode: 'audio',
      label: 'Raw audio pipeline',
      audio: null,
      transcript: null,
      recovered: true,
      duration: null,
      nodes: [['Normalize', '16 kHz mono'], ['Transcribe', 'faster-whisper'], ['Speakers', 'pyannote + roles'], ['Chunk', 'retrieval units'], ['Embed', 'searchable corpus']],
    };
  }

  function mergeServerJobs(serverJobs) {
    const previous = new Map(state.audioJobs.filter(job => job.job_id).map(job => [job.job_id, job]));
    state.audioJobs = (serverJobs || []).map(serverJob => {
      const current = previous.get(serverJob.job_id);
      const plan = current?.plan || audioPlanFromJob(serverJob);
      return { ...(current || {}), ...serverJob, plan };
    });
  }

  function hasLiveJobs() {
    return state.audioJobs.some(job => liveStatuses.has(job.status));
  }

  function ensureBatchForLiveJobs() {
    if (state.batch?.started) return state.batch;
    const jobs = state.audioJobs.filter(job => !['complete', 'cancelled'].includes(job.status));
    if (!jobs.length) return null;
    state.batch = {
      supported: [],
      plans: jobs.map(job => job.plan || audioPlanFromJob(job)),
      duplicateMatches: [],
      started: true,
      recovered: true,
      cancelling: false,
      importedCount: 0,
      errors: [],
    };
    return state.batch;
  }

  function startJobPollingIfNeeded() {
    if (hasLiveJobs() && !state.audioTimer) state.audioTimer = setInterval(pollAudioJobs, 1000);
    if (!hasLiveJobs() && !state.importBusy && state.audioTimer) {
      clearInterval(state.audioTimer);
      state.audioTimer = null;
    }
  }

  async function syncAudioJobsFromServer({ refreshSources = false } = {}) {
    const before = new Map(state.audioJobs.filter(job => job.job_id).map(job => [job.job_id, job.status]));
    const data = await api('/ingest/jobs');
    mergeServerJobs(data.jobs || []);
    const newlyComplete = state.audioJobs.some(job => job.status === 'complete' && before.get(job.job_id) !== 'complete');
    if (refreshSources || newlyComplete) await loadSources();
    else renderSources();
    captureCompletedEtaSamples();
    startJobPollingIfNeeded();
    return state.audioJobs;
  }

  function openProcessingJob(jobId) {
    const job = state.audioJobs.find(item => item.job_id === jobId);
    if (!job || ['complete', 'cancelled'].includes(job.status)) return;
    ensureBatchForLiveJobs();
    if (!state.batch?.started) return;
    state.focusedJobId = jobId;
    state.pendingAppend = null;
    $('importModal').hidden = false;
    setProcessingModal();
  }

  function renderSources() {
    const activeJobs = state.audioJobs.filter(job => liveStatuses.has(job.status));
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
        <div class="source-copy"><div class="source-name">${esc(job.display_name || job.plan?.name || 'Audio source')}</div><div class="source-meta">Failed</div></div>
      </div>`).join('');

    const sourceHtml = state.sources.map(source => `
      <label class="source-row" title="${esc(source.display_name)}">
        <input class="source-check" type="checkbox" data-key="${esc(source.source_key)}" ${state.selected.has(source.source_key) ? 'checked' : ''}>
        <div class="source-copy"><div class="source-name">${esc(source.display_name)}</div><div class="source-meta">${source.chunk_count} chunks · ${source.has_audio ? 'audio + transcript' : 'transcript'}</div></div>
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
      <label class="multi-select-option"><input type="checkbox" data-transcript-key="${esc(source.source_key)}" ${state.transcriptSelected.has(source.source_key) ? 'checked' : ''}><span>${esc(source.display_name)}</span></label>`).join('');
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
    // Selection is independent of ingestion state. Re-sync so a stale local UI
    // can never make live backend jobs disappear from the Processing section.
    syncAudioJobsFromServer().catch(() => {});
  }

