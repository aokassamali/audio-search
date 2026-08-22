(() => {
  const $ = (id) => document.getElementById(id);

  window.__AUDIO_SEARCH_FRONTEND_BUILD__ = '20260822-final-evidence-ux';

  function showTranscriptViewWithoutLoading() {
    document.querySelectorAll('.nav-item[data-view]').forEach(item => {
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
})();
