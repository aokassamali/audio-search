  function bindEvents() {
    $('sidebarToggle')?.addEventListener('click', () => $('app').classList.toggle('sidebar-collapsed'));
    document.querySelectorAll('.nav-item[data-view]').forEach(item => item.addEventListener('click', () => setView(item.dataset.view)));
    $('queryForm')?.addEventListener('submit', event => { event.preventDefault(); askQuestion($('queryInput').value); });
    $('queryInput')?.addEventListener('keydown', event => { if (event.key === 'Enter' && !event.shiftKey) { event.preventDefault(); $('queryForm').requestSubmit(); } });
    $('clearSources')?.addEventListener('click', () => { $('queryInput').value = ''; $('queryInput').focus(); });
    $('selectAllSources')?.addEventListener('click', selectAllLibrarySources);
    $('clearAllSources')?.addEventListener('click', clearLibrarySources);
    $('transcriptSourceButton')?.addEventListener('click', event => { event.stopPropagation(); $('transcriptSourceMenu').hidden = !$('transcriptSourceMenu').hidden; });
    $('transcriptSelectAll')?.addEventListener('click', () => { state.transcriptSelected = new Set(state.sources.map(source => source.source_key)); renderAuditSourcePicker(); });
    $('transcriptClearAll')?.addEventListener('click', () => { state.transcriptSelected.clear(); renderAuditSourcePicker(); });
    document.addEventListener('click', event => { if (!event.target.closest('#transcriptSourcePicker')) $('transcriptSourceMenu').hidden = true; });
    $('transcriptSearchButton')?.addEventListener('click', loadAuditTranscript);
    $('transcriptSearch')?.addEventListener('keydown', event => { if (event.key === 'Enter') { event.preventDefault(); loadAuditTranscript(); } });
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
    $('clipSeek')?.addEventListener('input', event => { if (state.clip.sourceKey) { player.currentTime = Math.max(state.clip.start, Math.min(state.clip.end, state.clip.start + Number(event.target.value || 0))); syncClipPlayer(); } });
    $('closeAudio')?.addEventListener('click', () => { player.pause(); $('audioDock').hidden = true; state.clip = { start: 0, end: 0, sourceKey: null, chunkId: null }; });

    $('addSourceButton')?.addEventListener('click', () => pickFiles(activeBatchIsRunning()));
    $('topAddButton')?.addEventListener('click', () => pickFiles(activeBatchIsRunning()));
    $('addImportFiles')?.addEventListener('click', () => pickFiles(true));
    $('closeImport')?.addEventListener('click', closeImport);
    $('cancelImport')?.addEventListener('click', () => {
      if (state.pendingAppend) cancelPendingAppend();
      else if (state.batch?.started) cancelCurrentBatch();
      else cancelPreview();
    });
    $('importAction')?.addEventListener('click', async () => {
      if (state.pendingAppend) { await commitAppend(state.pendingAppend); return; }
      if (!state.batch) return;
      if (!state.batch.started) await startBatch();
      else minimizeImport();
    });
    $('importModal')?.addEventListener('click', event => { if (event.target === $('importModal') && state.batch?.started && !state.pendingAppend) minimizeImport(); });

    let dragDepth = 0;
    window.addEventListener('dragenter', event => { event.preventDefault(); dragDepth += 1; $('dropOverlay').classList.add('visible'); });
    window.addEventListener('dragover', event => event.preventDefault());
    window.addEventListener('dragleave', event => { event.preventDefault(); dragDepth = Math.max(0, dragDepth - 1); if (!dragDepth) $('dropOverlay').classList.remove('visible'); });
    window.addEventListener('drop', event => {
      event.preventDefault();
      dragDepth = 0;
      $('dropOverlay').classList.remove('visible');
      const files = Array.from(event.dataTransfer?.files || []);
      if (!files.length) return;
      if (activeBatchIsRunning() || hasLiveJobs()) appendFilesToActiveBatch(files);
      else prepareImport(files, Boolean(state.batch && !state.batch.started && !$('importModal').hidden));
    });
  }

  async function initialize() {
    bindEvents();
    try {
      const health = await api('/health');
      state.ingestWorkers = Math.max(1, Number(health.ingest_workers) || 1);
      await loadSources();
      await syncAudioJobsFromServer();
      if (hasLiveJobs()) ensureBatchForLiveJobs();
    } catch (error) {
      if ($('connectionBadge')) $('connectionBadge').innerHTML = '<span></span>API unavailable';
      $('answerState').innerHTML = `<div class="answer-card refusal">${esc(error.message)}</div>`;
    }
    updateHomeLayout();
  }

  initialize();
