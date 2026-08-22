(() => {
  const state = {
    sources: [],
    selected: new Set(),
    transcriptSource: null,
    batch: null,
    importBusy: false,
    audioJobs: [],
  };

  const $ = (id) => document.getElementById(id);
  const ext = (name) => (name.split('.').pop() || '').toLowerCase();
  const stem = (name) => name.replace(/\.[^.]+$/, '');
  const transcriptExts = new Set(['json', 'srt', 'vtt', 'txt']);
  const audioExts = new Set(['mp3', 'wav', 'm4a', 'flac', 'ogg', 'aac']);
  const esc = (value = '') => String(value).replace(/[&<>'"]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;',"'":'&#39;','"':'&quot;'}[c]));

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

  async function api(path, options = {}) {
    const response = await fetch(path, options);
    if (!response.ok) {
      let message = `${response.status} ${response.statusText}`;
      try { message = (await response.json()).detail || message; } catch (_) {}
      throw new Error(message);
    }
    return response.json();
  }

  function progressBar(percent) {
    const pct = Math.max(0, Math.min(100, Number(percent) || 0));
    return `<div style="height:4px;margin-top:6px;border-radius:999px;background:#e5e5e9;overflow:hidden"><span style="display:block;height:100%;width:${pct}%;background:#171719;border-radius:999px;transition:width .25s ease"></span></div>`;
  }

  function renderSources() {
    const jobHtml = state.audioJobs.map(job => `
      <div class="source-row" title="${esc(job.name)}">
        <div style="width:8px;height:8px;border-radius:50%;background:#171719;flex:0 0 auto"></div>
        <div class="source-copy">
          <div class="source-name">${esc(job.name)}</div>
          <div class="source-meta">${esc(job.stageText)} · ${job.overall}%</div>
          ${progressBar(job.overall)}
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

    $('sourceList').innerHTML = jobHtml + sourceHtml;

    document.querySelectorAll('.source-check').forEach(input => input.addEventListener('change', () => {
      input.checked ? state.selected.add(input.dataset.key) : state.selected.delete(input.dataset.key);
      updateSelectedSummary();
    }));
  }

  function updateSelectedSummary() {
    const count = state.selected.size;
    if (!state.sources.length || count === state.sources.length) $('selectedSourceText').textContent = 'All recordings';
    else if (!count) $('selectedSourceText').textContent = 'No sources selected';
    else if (count === 1) $('selectedSourceText').textContent = state.sources.find(s => state.selected.has(s.source_key))?.display_name || '1 source';
    else $('selectedSourceText').textContent = `${count} of ${state.sources.length} sources`;
  }

  function renderTranscriptOptions() {
    $('transcriptSource').innerHTML = state.sources.map(s => `<option value="${esc(s.source_key)}">${esc(s.display_name)}</option>`).join('');
    if (!state.transcriptSource && state.sources.length) state.transcriptSource = state.sources[0].source_key;
    if (state.transcriptSource) $('transcriptSource').value = state.transcriptSource;
  }

  async function loadSources(reset = false) {
    const old = new Set(state.selected);
    state.sources = (await api('/sources')).sources || [];
    state.selected = new Set();
    state.sources.forEach(source => {
      if (reset || !old.size || old.has(source.source_key)) state.selected.add(source.source_key);
    });
    renderSources();
    renderTranscriptOptions();
    updateSelectedSummary();
    $('connectionBadge').classList.add('online');
    $('connectionBadge').innerHTML = '<span></span>API ready';
  }

  function setView(view) {
    document.querySelectorAll('.nav-item[data-view]').forEach(el => el.classList.toggle('active', el.dataset.view === view));
    $('askView').classList.toggle('view-active', view === 'ask');
    $('transcriptView').classList.toggle('view-active', view === 'transcript');
    $('viewEyebrow').textContent = view === 'ask' ? 'Search across your library' : 'Human-verifiable source of truth';
    $('viewTitle').textContent = view === 'ask' ? 'Ask your recordings' : 'Audit the transcript';
    if (view === 'transcript') loadTranscript();
  }

  function sourceForChunk(chunk) {
    return state.sources.find(source => source.source_key === chunk.source_key || source.source_id === chunk.source_id);
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
        ${source?.has_audio ? '<button id="playEvidence" class="primary-button" type="button">▶ Play audio</button>' : ''}
        <button id="viewTranscriptEvidence" class="secondary-button" type="button">View in transcript</button>
      </div>`;
    $('evidenceDrawer').classList.add('open');
    if ($('playEvidence')) $('playEvidence').addEventListener('click', () => playAudio(source, chunk));
    $('viewTranscriptEvidence').addEventListener('click', () => {
      if (!source) return;
      state.transcriptSource = source.source_key;
      renderTranscriptOptions();
      $('transcriptSearch').value = `chunk ${chunk.chunk_id}`;
      setView('transcript');
      loadTranscript(chunk.chunk_id);
    });
  }

  function playAudio(source, chunk) {
    $('audioDock').hidden = false;
    $('audioDockTitle').textContent = source.display_name;
    $('audioDockSubtitle').textContent = `Chunk ${chunk.chunk_id} · ${formatTime(chunk.start)}`;
    const player = $('audioPlayer');
    player.src = `/sources/${encodeURIComponent(source.source_key)}/audio`;
    const seek = () => {
      player.currentTime = Math.max(0, Number(chunk.start) || 0);
      player.play().catch(() => {});
      player.removeEventListener('loadedmetadata', seek);
    };
    player.addEventListener('loadedmetadata', seek);
    player.load();
  }

  async function askQuestion(question) {
    const query = question.trim();
    if (!query) return;
    const sourceKeys = state.selected.size === state.sources.length || state.selected.size === 0 ? null : Array.from(state.selected);
    const payload = {query, top_k: 6, source_keys: sourceKeys, retrieval_mode: 'global', top_k_per_source: 3};
    $('askButton').disabled = true;
    $('answerState').className = 'answer-state';
    $('answerState').innerHTML = '<div class="loading-card"><div class="loading-line"></div><div class="loading-line"></div><div class="loading-line"></div></div>';

    const [searchResult, answerResult] = await Promise.allSettled([
      api('/search', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(payload)}),
      api('/answer', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(payload)})
    ]);

    $('askButton').disabled = false;
    if (answerResult.status === 'rejected') {
      $('answerState').innerHTML = `<div class="answer-card refusal"><div class="answer-kicker">LLM unavailable</div><div class="answer-text">${esc(answerResult.reason.message)}. Retrieval remains available through the API.</div></div>`;
      return;
    }

    const answer = answerResult.value;
    const search = searchResult.status === 'fulfilled' ? searchResult.value : {results: []};
    const lookup = new Map((search.results || []).map(chunk => [`${chunk.source_id}:${chunk.chunk_id}`, chunk]));
    const cited = (answer.citations || []).map(citation => lookup.get(citation.citation_id) || citation);

    let html = `<div class="answer-card ${answer.answerable ? '' : 'refusal'}"><div class="answer-kicker">${answer.answerable ? 'Grounded answer' : 'Not enough evidence'}</div><div class="answer-text">${esc(answer.answer)}</div></div>`;
    if (cited.length) {
      html += `<div class="evidence-section"><div class="evidence-title">Evidence · ${cited.length}</div><div class="evidence-grid">${cited.map((chunk, i) => {
        const source = sourceForChunk(chunk);
        return `<button class="evidence-card" data-i="${i}" type="button"><div class="evidence-card-head"><div class="evidence-source">${esc(source?.display_name || chunk.source_id)}</div><div class="evidence-time">${formatTime(chunk.start)}${Number(chunk.end) >= 0 ? `–${formatTime(chunk.end)}` : ''}</div></div><div class="evidence-excerpt">${esc(chunk.speaker_text || chunk.text || 'Open to inspect source evidence.')}</div></button>`;
      }).join('')}</div></div>`;
    }
    $('answerState').innerHTML = html;
    document.querySelectorAll('[data-i]').forEach(button => button.addEventListener('click', () => openEvidence(cited[Number(button.dataset.i)])));
  }

  async function loadTranscript(forcedChunkId = null) {
    if (!state.transcriptSource) return;
    const raw = $('transcriptSearch').value.trim();
    let chunkId = forcedChunkId;
    let query = null;
    const match = raw.match(/^(?:chunk\s*)?#?(\d+)$/i);
    if (chunkId == null && match) chunkId = Number(match[1]);
    else if (raw && chunkId == null) query = raw;

    const params = new URLSearchParams({limit:'800'});
    if (chunkId != null) params.set('chunk_id', chunkId);
    if (query) params.set('query', query);
    $('transcriptList').innerHTML = '<div class="loading-card"><div class="loading-line"></div><div class="loading-line"></div></div>';

    try {
      const data = await api(`/sources/${encodeURIComponent(state.transcriptSource)}/chunks?${params}`);
      $('transcriptMeta').textContent = `${data.total} chunk${data.total === 1 ? '' : 's'} · ${data.source.has_audio ? 'audio-backed' : 'transcript-only'} · ${data.source.has_timestamps ? 'timestamps available' : 'untimed transcript'}`;
      $('transcriptList').innerHTML = (data.chunks || []).map((chunk, i) => `<article class="chunk-row" data-chunk="${i}"><div class="chunk-time">${formatTime(chunk.start)}</div><div class="chunk-text">${esc(chunk.speaker_text || chunk.text)}</div><div class="chunk-id">chunk ${chunk.chunk_id}</div></article>`).join('') || '<div class="empty-state"><h2>No matching chunks</h2><p>Try text, speaker, or a numeric chunk ID.</p></div>';
      document.querySelectorAll('[data-chunk]').forEach(row => row.addEventListener('click', () => openEvidence(data.chunks[Number(row.dataset.chunk)])));
    } catch (error) {
      $('transcriptList').innerHTML = `<div class="answer-card refusal">${esc(error.message)}</div>`;
    }
  }

  function makePlan(audio, transcript, suffix = '') {
    const baseName = stem((transcript || audio).name) + suffix;
    if (audio && transcript) {
      return {
        id:`${baseName}-${Math.random().toString(36).slice(2, 8)}`,
        name:baseName,
        mode:'both',
        label:'Audio + supplied transcript',
        audio,
        transcript,
        nodes:[['Validate','audio + transcript'],['Parse','timestamps / speakers'],['Attach audio','retain source audio'],['Chunk','retrieval units'],['Embed','searchable corpus']],
        duration:null,
      };
    }
    if (transcript) {
      return {
        id:`${baseName}-${Math.random().toString(36).slice(2, 8)}`,
        name:baseName,
        mode:'transcript',
        label:'Supplied transcript',
        audio:null,
        transcript,
        nodes:[['Validate','transcript supplied'],['Parse','timestamps / speakers'],['Skip ASR','Whisper not needed','skip'],['Chunk','retrieval units'],['Embed','searchable corpus']],
        duration:null,
      };
    }
    return {
      id:`${baseName}-${Math.random().toString(36).slice(2, 8)}`,
      name:baseName,
      mode:'audio',
      label:'Raw audio pipeline',
      audio,
      transcript:null,
      nodes:[['Normalize','16 kHz mono'],['Transcribe','faster-whisper'],['Speakers','pyannote + roles'],['Chunk','retrieval units'],['Embed','searchable corpus']],
      duration:null,
    };
  }

  function plansForFiles(files) {
    const supported = files.filter(file => transcriptExts.has(ext(file.name)) || audioExts.has(ext(file.name)));
    const groups = new Map();

    for (const file of supported) {
      const key = stem(file.name).toLowerCase();
      if (!groups.has(key)) groups.set(key, {audios: [], transcripts: []});
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

    return {supported, plans};
  }

  function renderDag(plan, active = -1, completed = -1, progress = 0) {
    $('dagGraph').innerHTML = plan.nodes.map((node, index) => {
      const skipped = node[2] === 'skip';
      const done = !skipped && index <= completed;
      const running = !skipped && index === active;
      const pct = done ? 100 : running ? progress : 0;
      return `<div class="dag-node ${done ? 'complete' : ''} ${running ? 'active' : ''} ${skipped ? 'skipped' : ''}"><div class="node-dot">${done ? '✓' : skipped ? '↷' : index + 1}</div><div class="node-label">${esc(node[0])}</div><div class="node-progress"><span style="width:${pct}%"></span></div><div class="node-detail">${running ? `${pct}% · ` : ''}${esc(node[1])}</div></div>`;
    }).join('');
  }

  function readAudioDuration(plan) {
    if (!plan.audio) return Promise.resolve(null);
    return new Promise(resolve => {
      const url = URL.createObjectURL(plan.audio);
      const audio = new Audio();
      audio.preload = 'metadata';
      audio.onloadedmetadata = () => {
        const duration = Number.isFinite(audio.duration) ? audio.duration : null;
        plan.duration = duration;
        URL.revokeObjectURL(url);
        resolve(duration);
      };
      audio.onerror = () => {
        URL.revokeObjectURL(url);
        resolve(null);
      };
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
      return `<div class="file-row"><div><div class="file-kind">${kind}</div><div class="file-name">${esc(plan.name)}</div></div><div class="file-size">${bytes(size)}${durationText}</div></div>`;
    }).join('');
  }

  function previewBatch() {
    const batch = state.batch;
    if (!batch) return;
    const first = batch.plans[0];

    $('importTitle').textContent = batch.plans.length === 1 ? 'Ingestion preview' : `Review ${batch.plans.length} sources`;
    renderFileSummary();
    $('ingestionPathBadge').textContent = batch.plans.length === 1 ? first.label : `${batch.plans.length} sources`;
    $('etaValue').textContent = 'Not calibrated';
    $('overallProgress').textContent = '0%';
    $('dagStatus').textContent = 'Ready to process';
    renderDag(first);
    $('importMessage').className = 'import-message';
    $('importMessage').textContent = batch.plans.length === 1
      ? 'Review the source and processing path, then start ingestion.'
      : `${batch.plans.length} sources will be added as separate ingestion jobs. Files with matching base names are paired as audio + transcript.`;
    $('cancelImport').textContent = 'Cancel';
    $('importAction').textContent = 'Process';
    $('importAction').disabled = false;
    $('cancelImport').disabled = false;
  }

  function currentAudioJob() {
    return state.audioJobs.find(job => !job.cancelled) || null;
  }

  function setProcessingModal() {
    const batch = state.batch;
    if (!batch) return;
    const job = currentAudioJob();
    const completedImports = batch.importedCount || 0;
    const totalItems = batch.plans.length;

    $('importTitle').textContent = totalItems === 1 ? 'Processing source' : `Processing ${totalItems} sources`;
    renderFileSummary();
    $('cancelImport').textContent = 'Cancel processing';
    $('importAction').textContent = 'Minimize';

    if (job) {
      $('ingestionPathBadge').textContent = totalItems === 1 ? job.plan.label : `${totalItems} sources`;
      $('etaValue').textContent = 'Not calibrated';
      $('overallProgress').textContent = `${Math.round(state.audioJobs.reduce((sum, item) => sum + item.overall, 0) / Math.max(1, state.audioJobs.length))}%`;
      $('dagStatus').textContent = totalItems > 1 ? `${job.stageText} · ${job.name}` : job.stageText;
      renderDag(job.plan, job.stageIndex, job.stageIndex - 1, job.stageProgress);
      $('importMessage').className = 'import-message';
      $('importMessage').textContent = 'Processing continues in the background while you work with the rest of your library.';
      return;
    }

    const first = batch.plans[0];
    $('ingestionPathBadge').textContent = totalItems === 1 ? first.label : `${totalItems} sources`;
    $('etaValue').textContent = 'Not calibrated';
    $('overallProgress').textContent = totalItems ? `${Math.round((completedImports / totalItems) * 100)}%` : '0%';
    $('dagStatus').textContent = state.importBusy ? 'Importing transcript' : 'Ready';
    renderDag(first, state.importBusy ? 1 : -1, state.importBusy ? 0 : -1, state.importBusy ? 50 : 0);
    $('importMessage').textContent = state.importBusy ? 'Parsing, chunking, and embedding supplied transcript data.' : 'Processing complete.';
  }

  function tickAudioJobs() {
    const stageDurations = [5, 480, 240, 25, 35];
    const stageWeights = [3, 55, 30, 4, 8];
    const stageNames = ['Normalizing audio', 'Transcribing', 'Identifying speakers', 'Building chunks', 'Generating embeddings'];

    for (const job of state.audioJobs) {
      if (job.cancelled) continue;
      let elapsed = (Date.now() - job.stageStartedAt) / 1000;
      let duration = stageDurations[job.stageIndex];
      if (elapsed >= duration && job.stageIndex < stageDurations.length - 1) {
        job.stageIndex += 1;
        job.stageStartedAt = Date.now();
        elapsed = 0;
        duration = stageDurations[job.stageIndex];
      }
      job.stageProgress = Math.min(99, Math.max(1, Math.round((elapsed / duration) * 100)));
      const completedWeight = stageWeights.slice(0, job.stageIndex).reduce((sum, value) => sum + value, 0);
      job.overall = Math.min(99, Math.round(completedWeight + stageWeights[job.stageIndex] * (job.stageProgress / 100)));
      job.stageText = stageNames[job.stageIndex];
    }

    renderSources();
    if (!$('importModal').hidden && state.batch?.started) setProcessingModal();
  }

  function startAudioPlans(plans) {
    const now = Date.now();
    for (const plan of plans) {
      state.audioJobs.push({
        id:plan.id,
        name:plan.name,
        plan,
        stageIndex:0,
        stageStartedAt:now,
        stageProgress:1,
        overall:1,
        stageText:'Normalizing audio',
        cancelled:false,
      });
    }
    if (state.audioJobs.length && !state.audioTimer) {
      state.audioTimer = setInterval(tickAudioJobs, 500);
    }
    tickAudioJobs();
  }

  async function importTranscriptPlan(plan) {
    const form = new FormData();
    form.append('transcript', plan.transcript);
    if (plan.audio) form.append('audio', plan.audio);
    form.append('source_name', plan.name);
    return api('/ingest/transcript', {method:'POST', body:form});
  }

  async function runTranscriptPlans(plans) {
    if (!plans.length) return;
    state.importBusy = true;
    try {
      for (const plan of plans) {
        if (!state.batch?.started) break;
        setProcessingModal();
        try {
          await importTranscriptPlan(plan);
          state.batch.importedCount = (state.batch.importedCount || 0) + 1;
          await loadSources(true);
        } catch (error) {
          state.batch.errors = state.batch.errors || [];
          state.batch.errors.push(`${plan.name}: ${error.message}`);
        }
      }
    } finally {
      state.importBusy = false;
      if (!$('importModal').hidden && state.batch?.started) setProcessingModal();
    }
  }

  function startBatch() {
    const batch = state.batch;
    if (!batch || batch.started) return;
    batch.started = true;
    batch.importedCount = 0;
    batch.errors = [];

    const rawAudioPlans = batch.plans.filter(plan => plan.mode === 'audio');
    const transcriptPlans = batch.plans.filter(plan => plan.mode !== 'audio');
    startAudioPlans(rawAudioPlans);
    runTranscriptPlans(transcriptPlans);
    setProcessingModal();
  }

  function prepareImport(files) {
    const {supported, plans} = plansForFiles(files);
    if (!supported.length || !plans.length) return;

    state.batch = {
      supported,
      plans,
      started:false,
      importedCount:0,
      errors:[],
    };

    $('importModal').hidden = false;
    previewBatch();

    Promise.all(plans.filter(plan => plan.audio).map(readAudioDuration)).then(() => {
      if (state.batch?.plans === plans) renderFileSummary();
    });
  }

  function minimizeImport() {
    $('importModal').hidden = true;
    setView('ask');
  }

  function cancelBatch() {
    if (state.importBusy) return;
    if (state.batch?.started) {
      const ids = new Set(state.batch.plans.map(plan => plan.id));
      state.audioJobs = state.audioJobs.filter(job => !ids.has(job.id));
      if (!state.audioJobs.length && state.audioTimer) {
        clearInterval(state.audioTimer);
        state.audioTimer = null;
      }
      renderSources();
    }
    state.batch = null;
    $('importModal').hidden = true;
  }

  function closeImport() {
    if (state.batch?.started) minimizeImport();
    else cancelBatch();
  }

  function pickFiles() {
    const input = document.createElement('input');
    input.type = 'file';
    input.multiple = true;
    input.accept = '.mp3,.wav,.m4a,.flac,.ogg,.aac,.json,.srt,.vtt,.txt';
    input.addEventListener('change', () => {
      if (input.files?.length) prepareImport(Array.from(input.files));
    }, {once:true});
    input.click();
  }

  $('sidebarToggle').addEventListener('click', () => $('app').classList.toggle('sidebar-collapsed'));
  document.querySelectorAll('.nav-item[data-view]').forEach(item => item.addEventListener('click', () => setView(item.dataset.view)));
  $('queryForm').addEventListener('submit', event => { event.preventDefault(); askQuestion($('queryInput').value); });
  document.querySelectorAll('[data-query]').forEach(button => button.addEventListener('click', () => { $('queryInput').value = button.dataset.query; askQuestion(button.dataset.query); }));
  $('clearSources').addEventListener('click', () => { state.selected = new Set(state.sources.map(s => s.source_key)); renderSources(); updateSelectedSummary(); });
  $('transcriptSource').addEventListener('change', () => { state.transcriptSource = $('transcriptSource').value; loadTranscript(); });
  $('transcriptSearchButton').addEventListener('click', () => loadTranscript());
  $('transcriptSearch').addEventListener('keydown', event => { if (event.key === 'Enter') loadTranscript(); });
  $('closeDrawer').addEventListener('click', () => $('evidenceDrawer').classList.remove('open'));
  $('closeAudio').addEventListener('click', () => { $('audioPlayer').pause(); $('audioDock').hidden = true; });
  $('addSourceButton').addEventListener('click', pickFiles);
  $('topAddButton').addEventListener('click', pickFiles);
  $('closeImport').addEventListener('click', closeImport);
  $('cancelImport').addEventListener('click', cancelBatch);
  $('importAction').addEventListener('click', () => {
    if (!state.batch) return;
    if (!state.batch.started) startBatch();
    else minimizeImport();
  });

  let dragDepth = 0;
  window.addEventListener('dragenter', event => {
    event.preventDefault();
    dragDepth += 1;
    $('dropOverlay').classList.add('visible');
  });
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
    if (files.length) prepareImport(files);
  });

  loadSources().catch(error => {
    $('connectionBadge').innerHTML = '<span></span>API unavailable';
    $('answerState').innerHTML = `<div class="answer-card refusal">${esc(error.message)}</div>`;
  });
})();
