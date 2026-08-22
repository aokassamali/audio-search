(() => {
  const $ = (id) => document.getElementById(id);

  window.__AUDIO_SEARCH_FRONTEND_BUILD__ = '20260822-clarify-intent';

  function showTranscriptViewWithoutLoading() {
    document.querySelectorAll('.nav-item[data-view="transcript"]').forEach(item => {
      item.classList.toggle('active', item.dataset.view === 'transcript');
    });
    $('askView')?.classList.remove('view-active');
    $('transcriptView')?.classList.add('view-active');
    if ($('viewEyebrow')) $('viewEyebrow').textContent = 'Human-verifiable source of truth';
    if ($('viewTitle')) $('viewTitle').textContent = 'Audit the transcript';
  }

  // demo_patch.js intentionally navigates to Transcripts before it renders
  // surrounding citation context. The original app's click handler also loads
  // the entire transcript on that same synthetic click, which races and can
  // overwrite the context view. Synthetic navigation is therefore view-only;
  // normal user clicks still use the ordinary transcript loader.
  window.addEventListener('click', event => {
    const transcriptNav = event.target.closest?.('.nav-item[data-view="transcript"]');
    if (!transcriptNav || event.isTrusted) return;
    event.preventDefault();
    event.stopImmediatePropagation();
    showTranscriptViewWithoutLoading();
  }, true);

  function clearLibrarySources() {
    document.querySelectorAll('.source-check').forEach(input => {
      if (input.checked) {
        input.checked = false;
        input.dispatchEvent(new Event('change', {bubbles:true}));
      }
    });
  }

  $('clearAllSources')?.addEventListener('click', event => {
    event.preventDefault();
    clearLibrarySources();
  });

  function relabelClarification() {
    const answerState = $('answerState');
    if (!answerState) return;
    const card = answerState.querySelector('.answer-card.refusal');
    const text = card?.querySelector('.answer-text')?.textContent?.trim() || '';
    if (!text.startsWith("I don't understand the question well enough to answer it reliably.")) return;
    const kicker = card.querySelector('.answer-kicker');
    if (kicker) kicker.textContent = 'Clarify question';
  }

  if ($('answerState')) {
    new MutationObserver(relabelClarification).observe($('answerState'), {
      childList: true,
      subtree: true,
    });
  }

  // Clicking the backdrop while a live ingestion is running should behave like
  // Minimize. The pre-processing confirmation remains explicit and is not
  // dismissed by an accidental outside click.
  $('importModal')?.addEventListener('click', event => {
    if (event.target !== $('importModal')) return;
    const action = $('importAction');
    if (action?.textContent?.trim() === 'Minimize') action.click();
  });

  const ETA_STORAGE_KEY = 'audio-search-local-rtf-v1';
  let ingestWorkers = 1;
  let latestJobs = [];
  const capturedEtaJobs = new Set();

  function median(values) {
    if (!values.length) return null;
    const sorted = [...values].sort((a, b) => a - b);
    const middle = Math.floor(sorted.length / 2);
    return sorted.length % 2
      ? sorted[middle]
      : (sorted[middle - 1] + sorted[middle]) / 2;
  }

  function loadStoredRtfSamples() {
    try {
      const parsed = JSON.parse(localStorage.getItem(ETA_STORAGE_KEY) || '[]');
      return Array.isArray(parsed)
        ? parsed.filter(value => Number.isFinite(value) && value > 0.05 && value < 5).slice(-20)
        : [];
    } catch (_) {
      return [];
    }
  }

  function storeRtfSample(value) {
    if (!Number.isFinite(value) || value <= 0.05 || value >= 5) return;
    const samples = loadStoredRtfSamples();
    samples.push(value);
    try {
      localStorage.setItem(ETA_STORAGE_KEY, JSON.stringify(samples.slice(-20)));
    } catch (_) {}
  }

  function parseDisplayedDuration(text) {
    const value = String(text || '').toLowerCase();
    const hourMatch = value.match(/(\d+(?:\.\d+)?)\s*h(?:our|ours)?\b/);
    const minuteMatch = value.match(/(\d+(?:\.\d+)?)\s*min(?:ute|utes)?\b/);
    let seconds = 0;
    if (hourMatch) seconds += Number(hourMatch[1]) * 3600;
    if (minuteMatch) seconds += Number(minuteMatch[1]) * 60;
    return seconds > 0 ? seconds : null;
  }

  function displayedDurations() {
    const map = new Map();
    document.querySelectorAll('#fileSummary .file-row').forEach(row => {
      const name = row.querySelector('.file-name')?.textContent?.trim();
      const duration = parseDisplayedDuration(row.querySelector('.file-size')?.textContent);
      if (name && duration) map.set(name, duration);
    });
    return map;
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

  function learnedRtfRange() {
    const samples = loadStoredRtfSamples();
    const center = median(samples);
    if (center == null) {
      // Broad initial heuristic for the full local GPU pipeline. It is
      // deliberately a range; completed runs replace it with machine-specific
      // measurements automatically.
      return {low: 0.5, high: 1.0, learned: false};
    }
    return {
      low: Math.max(0.1, center * 0.82),
      high: Math.max(center * 1.18, center * 0.82 + 0.05),
      learned: true,
    };
  }

  function captureCompletedSamples(durations) {
    latestJobs.forEach(job => {
      if (job.status !== 'complete' || capturedEtaJobs.has(job.job_id)) return;
      const duration = durations.get(job.display_name);
      const started = Number(job.started_at);
      const finished = Number(job.finished_at);
      if (duration && Number.isFinite(started) && Number.isFinite(finished) && finished > started) {
        storeRtfSample((finished - started) / duration);
        capturedEtaJobs.add(job.job_id);
      }
    });
  }

  function estimateWorkSeconds(factor, durations, started) {
    const rows = [...durations.entries()];
    if (!rows.length) return null;

    if (!started) {
      const total = rows.reduce((sum, [, duration]) => sum + duration * factor, 0);
      return total / Math.max(1, Math.min(ingestWorkers, rows.length));
    }

    const remaining = [];
    for (const [name, duration] of rows) {
      const job = latestJobs.find(item => item.display_name === name);
      if (!job) {
        remaining.push(duration * factor);
        continue;
      }
      if (job.status === 'complete' || job.status === 'failed') continue;
      if (job.status === 'running') {
        const progress = Math.max(0, Math.min(99, Number(job.overall_progress) || 0)) / 100;
        remaining.push(duration * factor * (1 - progress));
      } else {
        remaining.push(duration * factor);
      }
    }

    if (!remaining.length) return 0;
    return remaining.reduce((sum, value) => sum + value, 0)
      / Math.max(1, Math.min(ingestWorkers, remaining.length));
  }

  function renderEta() {
    const eta = $('etaValue');
    if (!eta || $('importModal')?.hidden) return;
    const durations = displayedDurations();
    if (!durations.size) {
      eta.textContent = 'Estimating…';
      return;
    }

    captureCompletedSamples(durations);
    const range = learnedRtfRange();
    const started = $('importAction')?.textContent?.trim() === 'Minimize';
    const low = estimateWorkSeconds(range.low, durations, started);
    const high = estimateWorkSeconds(range.high, durations, started);
    if (low == null || high == null) return;

    if (high <= 1) {
      eta.textContent = 'Finishing…';
    } else {
      eta.textContent = `≈ ${humanEta(low)}–${humanEta(high)}${started ? ' remaining' : ' total'}`;
    }
    eta.title = range.learned
      ? 'Estimate adapts to completed processing runs on this machine.'
      : 'Initial heuristic; it automatically adapts after completed runs on this machine.';
  }

  async function refreshEtaData() {
    try {
      const [healthResponse, jobsResponse] = await Promise.all([
        fetch('/health'),
        fetch('/ingest/jobs'),
      ]);
      if (healthResponse.ok) {
        const health = await healthResponse.json();
        ingestWorkers = Math.max(1, Number(health.ingest_workers) || 1);
      }
      if (jobsResponse.ok) {
        const payload = await jobsResponse.json();
        latestJobs = payload.jobs || [];
      }
    } catch (_) {}
    renderEta();
  }

  // app.js owns the modal and may temporarily write "Not calibrated" during
  // polling. This observer immediately replaces it with the adaptive estimate.
  const etaNode = $('etaValue');
  if (etaNode) {
    new MutationObserver(() => {
      if (etaNode.textContent.trim() === 'Not calibrated') {
        queueMicrotask(renderEta);
      }
    }).observe(etaNode, {childList:true, characterData:true, subtree:true});
  }

  setInterval(refreshEtaData, 1500);
  refreshEtaData();
})();
