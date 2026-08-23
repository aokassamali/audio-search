  function renderAuditChunks(chunks, focusKey = null) {
    const list = $('transcriptList');
    if (!list) return;
    list.innerHTML = chunks.map((chunk, index) => {
      const source = sourceForChunk(chunk);
      const key = `${chunk.source_key || source?.source_key || chunk.source_id}:${chunk.chunk_id}`;
      return `<article class="chunk-row ${focusKey === key ? 'context-focus' : ''}" data-audit-chunk="${index}"><div class="chunk-time">${formatTime(chunk.start)}</div><div class="chunk-text">${esc(chunk.speaker_text || chunk.text || '')}</div><div class="chunk-id"><div>chunk ${esc(chunk.chunk_id)}</div><div class="chunk-source">${esc(source?.display_name || chunk.source_id || '')}</div></div></article>`;
    }).join('') || '<div class="empty-state"><h2>No matching chunks</h2><p>Try a different query or source selection.</p></div>';
    list.querySelectorAll('[data-audit-chunk]').forEach(row => row.addEventListener('click', () => openEvidence(chunks[Number(row.dataset.auditChunk)])));
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
          api('/search', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ query: raw, top_k: 20, source_keys: keys, top_k_per_source: 5 }) }),
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
        const results = await Promise.all(keys.map(async key => (await api(`/sources/${encodeURIComponent(key)}/chunks?limit=800`)).chunks || []));
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
    const startPlayback = () => { player.currentTime = state.clip.start; player.play().catch(() => {}); };
    if (player.dataset.sourceKey !== source.source_key) {
      player.dataset.sourceKey = source.source_key;
      player.src = expectedSrc;
      player.addEventListener('loadedmetadata', startPlayback, { once: true });
      player.load();
    } else startPlayback();
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
    $('drawerBody').innerHTML = `<div class="drawer-meta"><span class="meta-chip">Chunk ${esc(chunk.chunk_id)}</span><span class="meta-chip">${formatTime(chunk.start)}${Number(chunk.end) >= 0 ? `–${formatTime(chunk.end)}` : ''}</span><span class="meta-chip">${source?.has_audio ? 'Audio attached' : 'Transcript only'}</span></div><div class="drawer-transcript">${esc(chunk.speaker_text || chunk.text || 'Transcript excerpt unavailable.')}</div><div class="drawer-actions">${source?.has_audio ? '<button id="drawerPlayEvidence" class="primary-button" type="button">▶ Play audio</button>' : ''}<button id="drawerViewTranscript" class="secondary-button" type="button">View in transcript</button></div>`;
    $('evidenceDrawer').classList.add('open');
    $('drawerPlayEvidence')?.addEventListener('click', () => playSourceAudio(source, chunk));
    $('drawerViewTranscript')?.addEventListener('click', () => {
      $('evidenceDrawer').classList.remove('open');
      if (!source) return;
      showAuditContext(source.source_key, chunk.chunk_id).catch(error => { $('transcriptList').innerHTML = `<div class="answer-card refusal">${esc(error.message)}</div>`; });
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
      } catch (_) { return citation; }
    }));
  }

  async function askQuestion(question) {
    const query = question.trim();
    if (!query) { showHome(); return; }
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
        return `<article class="evidence-card" data-evidence-index="${i}"><div class="evidence-card-head"><div class="evidence-source">${esc(source?.display_name || chunk.source_id || 'Source')}</div><div class="evidence-time">${formatTime(chunk.start)}${Number(chunk.end) >= 0 ? `–${formatTime(chunk.end)}` : ''}</div></div><div class="evidence-excerpt">${esc(chunk.speaker_text || chunk.text || 'Open to inspect source evidence.')}</div><div class="evidence-direct-actions"><button type="button" data-evidence-action="transcript">View transcript</button>${source?.has_audio ? '<button type="button" data-evidence-action="play">▶ Play</button>' : ''}</div></article>`;
      }).join('')}</div></div>`;
    }
    $('answerState').innerHTML = html;
    updateHomeLayout();
    document.querySelectorAll('[data-evidence-index]').forEach(card => {
      const chunk = cited[Number(card.dataset.evidenceIndex)];
      card.addEventListener('click', event => { if (!event.target.closest('[data-evidence-action]')) openEvidence(chunk); });
      card.querySelector('[data-evidence-action="transcript"]')?.addEventListener('click', () => {
        const source = sourceForChunk(chunk);
        if (source) showAuditContext(source.source_key, chunk.chunk_id).catch(error => { $('transcriptList').innerHTML = `<div class="answer-card refusal">${esc(error.message)}</div>`; });
      });
      card.querySelector('[data-evidence-action="play"]')?.addEventListener('click', () => {
        const source = sourceForChunk(chunk);
        if (source) playSourceAudio(source, chunk);
      });
    });
  }

