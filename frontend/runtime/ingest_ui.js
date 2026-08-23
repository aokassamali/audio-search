  function makePlan(audio, transcript, suffix = '') {
    const baseName = stem((transcript || audio).name) + suffix;
    if (audio && transcript) return { id: `${baseName}-${Math.random().toString(36).slice(2, 8)}`, name: baseName, mode: 'both', label: 'Audio + supplied transcript', audio, transcript, nodes: [['Validate', 'audio + transcript'], ['Parse', 'timestamps / speakers'], ['Attach audio', 'retain source audio'], ['Chunk', 'retrieval units'], ['Embed', 'searchable corpus']], duration: null };
    if (transcript) return { id: `${baseName}-${Math.random().toString(36).slice(2, 8)}`, name: baseName, mode: 'transcript', label: 'Supplied transcript', audio: null, transcript, nodes: [['Validate', 'transcript supplied'], ['Parse', 'timestamps / speakers'], ['Skip ASR', 'Whisper not needed', 'skip'], ['Chunk', 'retrieval units'], ['Embed', 'searchable corpus']], duration: null };
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
    return String(value || '').trim().toLowerCase().replace(/_/g, ' ').replace(/\s+/g, ' ');
  }

  function findDuplicateSources(plans, { includeActive = false } = {}) {
    const readyAudioSources = state.sources.filter(source => source.has_audio);
    const activeNames = includeActive
      ? new Set((state.batch?.plans || []).filter(plan => plan.audio || plan.recovered).map(plan => normalizeSourceName(plan.audio ? stem(plan.audio.name) : plan.name)))
      : new Set();
    const matches = [];
    for (const plan of plans) {
      if (!plan.audio) continue;
      const incoming = normalizeSourceName(stem(plan.audio.name));
      const source = readyAudioSources.find(item => normalizeSourceName(item.display_name) === incoming);
      if (source) {
        matches.push({ planId: plan.id, planName: plan.name, sourceKey: source.source_key, sourceName: source.display_name });
      } else if (activeNames.has(incoming)) {
        matches.push({ planId: plan.id, planName: plan.name, sourceKey: null, sourceName: plan.name });
      }
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
      audio.onloadedmetadata = () => { plan.duration = Number.isFinite(audio.duration) ? audio.duration : null; URL.revokeObjectURL(url); resolve(plan.duration); };
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
      const duplicateLabel = duplicate ? `<div class="file-kind" style="margin-top:4px;color:#9a5b00">Already in library · ${esc(duplicate.sourceName)}</div>` : '';
      const detail = files.length ? `${bytes(size)}${durationText}` : (plan.recovered ? 'Live job' : durationText.replace(/^ · /, ''));
      return `<div class="file-row"><div><div class="file-kind">${kind}</div><div class="file-name">${esc(plan.name)}</div>${duplicateLabel}</div><div class="file-size">${detail}</div></div>`;
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
    if (!started) return rawPlans.reduce((sum, plan) => sum + plan.duration * factor, 0) / Math.max(1, Math.min(state.ingestWorkers, rawPlans.length));
    const remaining = rawPlans.map(plan => {
      const job = state.audioJobs.find(item => item.plan?.id === plan.id);
      if (!job) return plan.duration * factor;
      if (terminalStatuses.has(job.status)) return 0;
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
      $('etaValue').textContent = state.batch.plans.some(plan => plan.mode === 'audio' && !plan.recovered) ? 'Estimating…' : 'In progress';
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
    $('importTitle').textContent = duplicateWarning ? (duplicates.length === 1 ? 'Already in library' : `${duplicates.length} sources already in library`) : (batch.plans.length === 1 ? 'Review source' : `Review ${batch.plans.length} sources`);
    renderFileSummary();
    $('ingestionPathBadge').textContent = batch.plans.length === 1 ? first.label : `${batch.plans.length} sources`;
    $('overallProgress').textContent = '0%';
    $('dagStatus').textContent = duplicateWarning ? 'Reprocessing requires confirmation' : 'Waiting for confirmation';
    renderDag(first);
    $('importMessage').hidden = !duplicateWarning;
    $('importMessage').className = 'import-message';
    $('importMessage').textContent = duplicateWarning
      ? (duplicates.length === 1 ? `It looks like ${duplicates[0].sourceName} is already in your library. Reprocessing will run the ingestion pipeline again and create a new source. Continue?` : `${duplicates.length} of these audio sources already appear in your library. Reprocessing will run the ingestion pipeline again and create new sources. Continue?`)
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
    return state.audioJobs.find(job => ['running', 'cancelling'].includes(job.status)) || state.audioJobs.find(job => job.status === 'queued') || state.audioJobs.find(job => job.status === 'failed') || null;
  }

  function batchOverall() {
    const plans = state.batch?.plans || [];
    const total = plans.length || 1;
    const imported = state.batch?.importedCount || 0;
    const planIds = new Set(plans.map(plan => plan.id));
    const audioPoints = state.audioJobs
      .filter(job => planIds.has(job.plan?.id))
      .reduce((sum, job) => sum + Number(job.overall_progress || 0), 0);
    return Math.min(100, Math.round((audioPoints + imported * 100) / total));
  }

  function showPendingAppendWarning() {
    const pending = state.pendingAppend;
    if (!pending) return;
    const duplicates = pending.duplicateMatches || [];
    $('importTitle').textContent = duplicates.length === 1 ? 'Already in library' : `${duplicates.length} files already exist`;
    $('fileSummary').innerHTML = pending.plans.map(plan => {
      const duplicate = duplicates.find(item => item.planId === plan.id);
      return `<div class="file-row"><div><div class="file-kind">${plan.mode === 'audio' ? 'Audio' : plan.mode === 'both' ? 'Audio + transcript' : 'Transcript'}</div><div class="file-name">${esc(plan.name)}</div>${duplicate ? `<div class="file-kind" style="margin-top:4px;color:#9a5b00">Already exists · ${esc(duplicate.sourceName)}</div>` : ''}</div></div>`;
    }).join('');
    $('ingestionPathBadge').textContent = 'Add to active batch';
    $('overallProgress').textContent = `${batchOverall()}%`;
    $('dagStatus').textContent = 'Reprocessing requires confirmation';
    renderDag(pending.plans[0]);
    $('importMessage').hidden = false;
    $('importMessage').className = 'import-message';
    $('importMessage').textContent = duplicates.length === 1
      ? `That audio is already in the library or current processing queue. Reprocessing will run the full pipeline again and create a new source. Add it anyway?`
      : `${duplicates.length} of these audio files are already in the library or current processing queue. Add them and run the pipeline again?`;
    $('cancelImport').hidden = false;
    $('cancelImport').textContent = 'Cancel';
    $('cancelImport').disabled = false;
    $('addImportFiles').hidden = true;
    $('importAction').textContent = 'Reprocess anyway';
    $('importAction').disabled = false;
  }

  function setProcessingModal() {
    const batch = ensureBatchForLiveJobs() || state.batch;
    if (!batch) return;
    if (state.pendingAppend) { showPendingAppendWarning(); return; }
    const job = currentAudioJob();
    const totalItems = batch.plans.length;
    $('importTitle').textContent = totalItems === 1 ? 'Processing source' : `Processing ${totalItems} sources`;
    renderFileSummary();
    $('addImportFiles').hidden = Boolean(batch.cancelling);
    $('addImportFiles').textContent = 'Add files';
    $('importAction').textContent = 'Minimize';
    $('cancelImport').hidden = false;
    $('cancelImport').textContent = batch.cancelling ? 'Cancelling…' : 'Cancel processing';
    $('cancelImport').disabled = Boolean(batch.cancelling);
    $('overallProgress').textContent = `${batchOverall()}%`;
    $('importMessage').hidden = false;
    renderEta();

    if (job) {
      const plan = job.plan || audioPlanFromJob(job);
      $('ingestionPathBadge').textContent = totalItems === 1 ? plan.label : `${totalItems} sources`;
      $('dagStatus').textContent = totalItems > 1 ? `${jobStageText(job)} · ${job.display_name || plan.name}` : jobStageText(job);
      renderDag(plan, job.stages || null);
      $('importMessage').className = job.status === 'failed' ? 'import-message error' : 'import-message';
      $('importMessage').textContent = job.status === 'failed'
        ? (job.error || 'Dagster materialization failed.')
        : batch.cancelling
          ? 'Stopping current processing and cancelling queued sources…'
          : 'This is a live Dagster ingestion job. You can minimize this window, add more files, and keep using the rest of the library.';
      return;
    }
    const first = batch.plans[0];
    $('ingestionPathBadge').textContent = totalItems === 1 ? first.label : `${totalItems} sources`;
    $('dagStatus').textContent = state.importBusy ? 'Importing transcript' : batch.cancelling ? 'Cancelling' : 'Processing complete';
    renderDag(first);
    $('importMessage').className = 'import-message';
    $('importMessage').textContent = state.importBusy ? 'Parsing, chunking, and embedding supplied transcript data.' : batch.cancelling ? 'Cancelling processing…' : 'Processing complete.';
  }
