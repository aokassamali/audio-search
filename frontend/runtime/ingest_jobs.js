  async function submitAudioPlan(plan) {
    const form = new FormData();
    form.append('audio', plan.audio);
    form.append('source_name', plan.name);
    const job = await api('/ingest/audio', { method: 'POST', body: form });
    job.plan = plan;
    const existing = state.audioJobs.findIndex(item => item.job_id === job.job_id);
    if (existing >= 0) state.audioJobs[existing] = job;
    else state.audioJobs.push(job);
    return job;
  }

  async function pollAudioJobs() {
    try {
      await syncAudioJobsFromServer();
      if (!$('importModal').hidden && state.batch?.started) setProcessingModal();
      if (!hasLiveJobs()) state.focusedJobId = null;
    } catch (_) {
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
    startJobPollingIfNeeded();
  }

  async function importTranscriptPlan(plan) {
    const form = new FormData();
    form.append('transcript', plan.transcript);
    if (plan.audio) form.append('audio', plan.audio);
    form.append('source_name', plan.name);
    return api('/ingest/transcript', { method: 'POST', body: form });
  }

  async function drainTranscriptQueue() {
    if (state.transcriptDraining) return;
    state.transcriptDraining = true;
    state.importBusy = true;
    try {
      while (state.transcriptQueue.length) {
        if (!state.batch?.started || state.batch?.cancelling) break;
        const plan = state.transcriptQueue.shift();
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
      state.transcriptDraining = false;
      if (state.batch?.cancelling) {
        state.transcriptQueue = [];
        if (!hasLiveJobs()) {
          state.batch = null;
          state.focusedJobId = null;
          $('importModal').hidden = true;
        }
      } else if (!$('importModal').hidden && state.batch?.started) setProcessingModal();
      renderSources();
    }
  }

  async function enqueueTranscriptPlans(plans) {
    if (!plans.length) return;
    state.transcriptQueue.push(...plans);
    await drainTranscriptQueue();
  }

  async function startBatch() {
    const batch = state.batch;
    if (!batch || batch.started) return;
    batch.started = true;
    batch.cancelling = false;
    batch.importedCount = 0;
    batch.errors = [];
    state.pendingAppend = null;
    state.audioJobs = [];
    state.focusedJobId = null;
    setProcessingModal();
    const rawAudioPlans = batch.plans.filter(plan => plan.mode === 'audio');
    const transcriptPlans = batch.plans.filter(plan => plan.mode !== 'audio');
    await Promise.all([startAudioPlans(rawAudioPlans), enqueueTranscriptPlans(transcriptPlans)]);
    if (!$('importModal').hidden && state.batch?.started) setProcessingModal();
  }

  async function cancelCurrentBatch() {
    const batch = ensureBatchForLiveJobs() || state.batch;
    if (!batch?.started || batch.cancelling) return;
    batch.cancelling = true;
    state.pendingAppend = null;
    state.transcriptQueue = [];
    setProcessingModal();
    const activeJobs = state.audioJobs.filter(job => job.job_id && !terminalStatuses.has(job.status));
    try {
      await Promise.all(activeJobs.map(job => api(`/ingest/jobs/${encodeURIComponent(job.job_id)}/cancel`, { method: 'POST' })));
      for (let attempt = 0; attempt < 32; attempt += 1) {
        await pollAudioJobs();
        if (!hasLiveJobs()) break;
        await new Promise(resolve => setTimeout(resolve, 250));
      }
      if (state.importBusy) {
        $('importMessage').className = 'import-message';
        $('importMessage').textContent = 'Audio processing stopped. Waiting for the current transcript import to finish safely…';
        return;
      }
      state.batch = null;
      state.audioJobs = state.audioJobs.filter(job => !['cancelled'].includes(job.status));
      state.focusedJobId = null;
      $('importModal').hidden = true;
      renderSources();
      startJobPollingIfNeeded();
    } catch (error) {
      batch.cancelling = false;
      $('importMessage').className = 'import-message error';
      $('importMessage').textContent = `Could not cancel processing: ${error.message}`;
      setProcessingModal();
    }
  }

  function fileIdentity(file) {
    return `${file.name}::${file.size}::${file.lastModified}`;
  }

  function uniqueFiles(files) {
    const seen = new Set();
    return files.filter(file => {
      const key = fileIdentity(file);
      if (seen.has(key)) return false;
      seen.add(key);
      return true;
    });
  }

  function activeBatchIsRunning() {
    return Boolean(state.batch?.started && (hasLiveJobs() || state.importBusy || state.transcriptQueue.length));
  }

  function prepareImport(files, append = false) {
    if (activeBatchIsRunning()) {
      appendFilesToActiveBatch(files);
      return;
    }
    const existing = append && state.batch && !state.batch.started ? state.batch.supported : [];
    const mergedFiles = uniqueFiles([...existing, ...files]);
    const { supported, plans } = plansForFiles(mergedFiles);
    if (!supported.length || !plans.length) return;
    const duplicateMatches = findDuplicateSources(plans);
    state.batch = { supported, plans, duplicateMatches, started: false, cancelling: false, importedCount: 0, errors: [] };
    state.pendingAppend = null;
    state.audioJobs = state.audioJobs.filter(job => liveStatuses.has(job.status));
    state.focusedJobId = null;
    $('importModal').hidden = false;
    previewBatch();
    Promise.all(plans.filter(plan => plan.audio).map(readAudioDuration)).then(() => {
      if (state.batch?.plans === plans) { renderFileSummary(); renderEta(); }
    });
  }

  async function commitAppend(pending) {
    const batch = ensureBatchForLiveJobs() || state.batch;
    if (!batch?.started || batch.cancelling) return;
    state.pendingAppend = null;
    const existingIds = new Set((batch.supported || []).map(fileIdentity));
    const newFiles = pending.supported.filter(file => !existingIds.has(fileIdentity(file)));
    batch.supported = [...(batch.supported || []), ...newFiles];
    batch.plans.push(...pending.plans);
    batch.duplicateMatches = [];
    await Promise.all(pending.plans.filter(plan => plan.audio).map(readAudioDuration));
    if (!$('importModal').hidden) setProcessingModal();
    const rawAudioPlans = pending.plans.filter(plan => plan.mode === 'audio');
    const transcriptPlans = pending.plans.filter(plan => plan.mode !== 'audio');
    await Promise.all([startAudioPlans(rawAudioPlans), enqueueTranscriptPlans(transcriptPlans)]);
    if (!$('importModal').hidden) setProcessingModal();
  }

  function appendFilesToActiveBatch(files) {
    const batch = ensureBatchForLiveJobs();
    if (!batch?.started || batch.cancelling) return;
    const currentIdentities = new Set((batch.supported || []).map(fileIdentity));
    const incoming = uniqueFiles(files).filter(file => !currentIdentities.has(fileIdentity(file)));
    const { supported, plans } = plansForFiles(incoming);
    if (!supported.length || !plans.length) return;
    const duplicateMatches = findDuplicateSources(plans, { includeActive: true });
    const pending = { supported, plans, duplicateMatches };
    if (duplicateMatches.length) {
      state.pendingAppend = pending;
      $('importModal').hidden = false;
      showPendingAppendWarning();
      return;
    }
    commitAppend(pending).catch(error => {
      $('importModal').hidden = false;
      $('importMessage').hidden = false;
      $('importMessage').className = 'import-message error';
      $('importMessage').textContent = `Could not add files: ${error.message}`;
    });
  }

  function cancelPendingAppend() {
    state.pendingAppend = null;
    if (state.batch?.started) setProcessingModal();
    else $('importModal').hidden = true;
  }

  function minimizeImport() {
    state.pendingAppend = null;
    $('importModal').hidden = true;
    setView('ask');
  }

  function cancelPreview() {
    state.pendingAppend = null;
    state.batch = null;
    $('importModal').hidden = true;
  }

  function closeImport() {
    if (state.pendingAppend) { cancelPendingAppend(); return; }
    if (state.batch?.started) minimizeImport();
    else cancelPreview();
  }

  function pickFiles(append = false) {
    const input = document.createElement('input');
    input.type = 'file';
    input.multiple = true;
    input.accept = '.mp3,.wav,.m4a,.flac,.ogg,.aac,.json,.srt,.vtt,.txt';
    input.addEventListener('change', () => {
      if (!input.files?.length) return;
      const files = Array.from(input.files);
      if (activeBatchIsRunning() || (append && state.batch?.started)) appendFilesToActiveBatch(files);
      else prepareImport(files, append);
    }, { once: true });
    input.click();
  }
