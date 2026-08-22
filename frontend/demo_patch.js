(() => {
  const $ = (id) => document.getElementById(id);
  const esc = (value = '') => String(value).replace(/[&<>'"]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;',"'":'&#39;','"':'&quot;'}[c]));

  window.__AUDIO_SEARCH_FRONTEND_BUILD__ = '20260822-demo-ux';

  const homeMarkup = '<div class="empty-orb">⌁</div><h2>Ask across hours of audio in seconds.</h2><p>Answers stay traceable to transcript chunks, speakers, timestamps, and the original recording when audio is available.</p>';

  const audit = {
    sources: [],
    selected: new Set(),
    initialized: false,
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

  function sourceByKey(key) {
    return audit.sources.find(source => source.source_key === key || source.source_id === key);
  }

  function sourceForChunk(chunk) {
    return sourceByKey(chunk.source_key) || sourceByKey(chunk.source_id);
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

  function showHome() {
    const answer = $('answerState');
    if (!answer) return;
    answer.className = 'answer-state empty-state';
    answer.innerHTML = homeMarkup;
    document.querySelector('.ask-stage')?.classList.add('home-layout');
  }

  function updateHomeLayout() {
    const answer = $('answerState');
    const stage = document.querySelector('.ask-stage');
    if (!answer || !stage) return;
    stage.classList.toggle('home-layout', answer.classList.contains('empty-state'));
  }

  function selectAllLibrarySources() {
    document.querySelectorAll('.source-check').forEach(input => {
      if (!input.checked) {
        input.checked = true;
        input.dispatchEvent(new Event('change', {bubbles:true}));
      }
    });
  }

  function playSourceAudio(source, chunk) {
    if (!source?.has_audio) return;
    const dock = $('audioDock');
    const player = $('audioPlayer');
    if (!dock || !player) return;
    dock.hidden = false;
    $('audioDockTitle').textContent = source.display_name;
    $('audioDockSubtitle').textContent = `Chunk ${chunk.chunk_id} · ${formatTime(chunk.start)}`;
    player.src = `/sources/${encodeURIComponent(source.source_key)}/audio`;
    const seek = () => {
      player.currentTime = Math.max(0, Number(chunk.start) || 0);
      player.play().catch(() => {});
      player.removeEventListener('loadedmetadata', seek);
    };
    player.addEventListener('loadedmetadata', seek);
    player.load();
  }

  function openAuditEvidence(chunk) {
    const source = sourceForChunk(chunk);
    if (!source) return;
    $('drawerTitle').textContent = source.display_name;
    $('drawerBody').innerHTML = `
      <div class="drawer-meta">
        <span class="meta-chip">Chunk ${esc(chunk.chunk_id)}</span>
        <span class="meta-chip">${formatTime(chunk.start)}${Number(chunk.end) >= 0 ? `–${formatTime(chunk.end)}` : ''}</span>
        <span class="meta-chip">${source.has_audio ? 'Audio attached' : 'Transcript only'}</span>
      </div>
      <div class="drawer-transcript">${esc(chunk.speaker_text || chunk.text || 'Transcript excerpt unavailable.')}</div>
      <div class="drawer-actions">
        ${source.has_audio ? '<button id="auditPlayEvidence" class="primary-button" type="button">▶ Play audio</button>' : ''}
        <button id="auditLocateEvidence" class="secondary-button" type="button">View in transcript</button>
      </div>`;
    $('evidenceDrawer').classList.add('open');
    $('auditPlayEvidence')?.addEventListener('click', () => playSourceAudio(source, chunk));
    $('auditLocateEvidence')?.addEventListener('click', () => {
      audit.selected = new Set([source.source_key]);
      renderAuditSourcePicker();
      $('transcriptSearch').value = `chunk ${chunk.chunk_id}`;
      $('evidenceDrawer').classList.remove('open');
      loadAuditTranscript(chunk.chunk_id);
    });
  }

  async function refreshAuditSources() {
    const data = await api('/sources');
    const previous = new Set(audit.selected);
    audit.sources = data.sources || [];
    const keys = new Set(audit.sources.map(source => source.source_key));
    audit.selected = new Set([...previous].filter(key => keys.has(key)));

    if (!audit.initialized) {
      const legacy = $('transcriptSource')?.value;
      const fallback = audit.sources[0]?.source_key;
      if (legacy && keys.has(legacy)) audit.selected.add(legacy);
      else if (fallback) audit.selected.add(fallback);
    }
    audit.initialized = true;
    renderAuditSourcePicker();
  }

  function renderAuditSourcePicker() {
    const button = $('transcriptSourceButton');
    const options = $('transcriptSourceOptions');
    if (!button || !options) return;

    const selectedSources = audit.sources.filter(source => audit.selected.has(source.source_key));
    if (!selectedSources.length) button.querySelector('.picker-value').textContent = 'Select sources';
    else if (selectedSources.length === audit.sources.length && audit.sources.length > 1) button.querySelector('.picker-value').textContent = 'All sources';
    else if (selectedSources.length === 1) button.querySelector('.picker-value').textContent = selectedSources[0].display_name;
    else button.querySelector('.picker-value').textContent = `${selectedSources.length} sources`;

    options.innerHTML = audit.sources.map(source => `
      <label class="multi-select-option">
        <input type="checkbox" data-transcript-key="${esc(source.source_key)}" ${audit.selected.has(source.source_key) ? 'checked' : ''}>
        <span>${esc(source.display_name)}</span>
      </label>`).join('');

    options.querySelectorAll('[data-transcript-key]').forEach(input => {
      input.addEventListener('change', () => {
        input.checked ? audit.selected.add(input.dataset.transcriptKey) : audit.selected.delete(input.dataset.transcriptKey);
        renderAuditSourcePicker();
      });
    });
  }

  function renderAuditChunks(chunks) {
    const list = $('transcriptList');
    if (!list) return;
    list.innerHTML = chunks.map((chunk, index) => {
      const source = sourceForChunk(chunk);
      return `<article class="chunk-row" data-audit-chunk="${index}">
        <div class="chunk-time">${formatTime(chunk.start)}</div>
        <div class="chunk-text">${esc(chunk.speaker_text || chunk.text || '')}</div>
        <div class="chunk-id"><div>chunk ${esc(chunk.chunk_id)}</div><div class="chunk-source">${esc(source?.display_name || chunk.source_id || '')}</div></div>
      </article>`;
    }).join('') || '<div class="empty-state"><h2>No matching chunks</h2><p>Try a different query or source selection.</p></div>';

    list.querySelectorAll('[data-audit-chunk]').forEach(row => {
      row.addEventListener('click', () => openAuditEvidence(chunks[Number(row.dataset.auditChunk)]));
    });
  }

  async function loadAuditTranscript(forcedChunkId = null) {
    try {
      await refreshAuditSources();
      const keys = [...audit.selected];
      if (!keys.length) {
        $('transcriptMeta').textContent = 'Select at least one source.';
        $('transcriptList').innerHTML = '';
        return;
      }

      const raw = $('transcriptSearch').value.trim();
      const numeric = raw.match(/^(?:chunk\s*)?#?(\d+)$/i);
      const chunkId = forcedChunkId != null ? Number(forcedChunkId) : numeric ? Number(numeric[1]) : null;
      $('transcriptList').innerHTML = '<div class="loading-card"><div class="loading-line"></div><div class="loading-line"></div></div>';

      let chunks = [];
      let mode = 'full transcript';

      if (chunkId != null) {
        const results = await Promise.all(keys.map(async key => {
          const params = new URLSearchParams({limit:'100', chunk_id:String(chunkId)});
          const data = await api(`/sources/${encodeURIComponent(key)}/chunks?${params}`);
          return data.chunks || [];
        }));
        chunks = results.flat();
        mode = 'chunk lookup';
      } else if (raw) {
        const [hybridResult, exactResults] = await Promise.all([
          api('/search', {
            method:'POST',
            headers:{'Content-Type':'application/json'},
            body:JSON.stringify({
              query:raw,
              top_k:20,
              source_keys:keys,
              retrieval_mode:'global',
              top_k_per_source:5,
            }),
          }),
          Promise.all(keys.map(async key => {
            const params = new URLSearchParams({limit:'50', query:raw});
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

  function syncAuditToLegacySource() {
    const legacy = $('transcriptSource')?.value;
    if (!legacy) return;
    audit.selected = new Set([legacy]);
    renderAuditSourcePicker();
  }

  function enhanceEvidenceCards() {
    document.querySelectorAll('.evidence-card:not([data-direct-actions])').forEach(card => {
      card.dataset.directActions = '1';
      const actions = document.createElement('span');
      actions.className = 'evidence-direct-actions';
      actions.innerHTML = '<span role="button" tabindex="0" data-evidence-action="transcript">View transcript</span><span role="button" tabindex="0" data-evidence-action="play">▶ Play</span>';
      card.appendChild(actions);

      actions.querySelectorAll('[data-evidence-action]').forEach(action => {
        const run = event => {
          event.preventDefault();
          event.stopPropagation();
          const kind = action.dataset.evidenceAction;
          card.click();
          setTimeout(() => {
            if (kind === 'play') {
              const play = $('playEvidence');
              if (play) {
                play.click();
                $('evidenceDrawer')?.classList.remove('open');
              }
            } else {
              const locate = $('viewTranscriptEvidence');
              if (locate) {
                locate.click();
                setTimeout(() => {
                  syncAuditToLegacySource();
                  loadAuditTranscript();
                  $('evidenceDrawer')?.classList.remove('open');
                }, 0);
              }
            }
          }, 0);
        };
        action.addEventListener('click', run);
        action.addEventListener('keydown', event => {
          if (event.key === 'Enter' || event.key === ' ') run(event);
        });
      });
    });
  }

  $('queryInput')?.addEventListener('keydown', event => {
    if (event.key !== 'Enter' || event.shiftKey) return;
    event.preventDefault();
    const value = $('queryInput').value.trim();
    if (!value) showHome();
    else $('queryForm').requestSubmit();
  }, true);

  document.addEventListener('submit', event => {
    if (event.target?.id !== 'queryForm') return;
    if ($('queryInput').value.trim()) return;
    event.preventDefault();
    event.stopImmediatePropagation();
    showHome();
  }, true);

  document.addEventListener('click', event => {
    if (event.target.closest('#clearSources')) {
      event.preventDefault();
      event.stopImmediatePropagation();
      $('queryInput').value = '';
      $('queryInput').focus();
      return;
    }
    if (event.target.closest('#selectAllSources')) {
      event.preventDefault();
      selectAllLibrarySources();
    }
  }, true);

  $('transcriptSourceButton')?.addEventListener('click', event => {
    event.stopPropagation();
    const menu = $('transcriptSourceMenu');
    menu.hidden = !menu.hidden;
  });
  $('transcriptSelectAll')?.addEventListener('click', () => {
    audit.selected = new Set(audit.sources.map(source => source.source_key));
    renderAuditSourcePicker();
  });
  $('transcriptClearAll')?.addEventListener('click', () => {
    audit.selected.clear();
    renderAuditSourcePicker();
  });
  document.addEventListener('click', event => {
    if (!event.target.closest('#transcriptSourcePicker')) $('transcriptSourceMenu').hidden = true;
  });

  document.addEventListener('click', event => {
    if (!event.target.closest('#transcriptSearchButton')) return;
    event.preventDefault();
    event.stopImmediatePropagation();
    loadAuditTranscript();
  }, true);
  $('transcriptSearch')?.addEventListener('keydown', event => {
    if (event.key !== 'Enter') return;
    event.preventDefault();
    event.stopImmediatePropagation();
    loadAuditTranscript();
  }, true);

  document.querySelector('[data-view="transcript"]')?.addEventListener('click', () => {
    setTimeout(() => loadAuditTranscript(), 0);
  });

  document.addEventListener('click', event => {
    if (!event.target.closest('#viewTranscriptEvidence')) return;
    setTimeout(() => {
      syncAuditToLegacySource();
      loadAuditTranscript();
    }, 0);
  });

  const answer = $('answerState');
  if (answer) {
    new MutationObserver(() => {
      updateHomeLayout();
      enhanceEvidenceCards();
    }).observe(answer, {childList:true, subtree:true, attributes:true, attributeFilter:['class']});
  }

  refreshAuditSources().catch(() => {});
  updateHomeLayout();
  enhanceEvidenceCards();
})();
