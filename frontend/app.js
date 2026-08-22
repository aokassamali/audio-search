(() => {
  const state = {sources: [], selected: new Set(), transcriptSource: null, importPlan: null, importResult: null, importBusy: false};
  const $ = (id) => document.getElementById(id);
  const ext = (name) => (name.split('.').pop() || '').toLowerCase();
  const transcriptExts = new Set(['json','srt','vtt','txt']);
  const audioExts = new Set(['mp3','wav','m4a','flac','ogg','aac']);
  const esc = (value='') => String(value).replace(/[&<>'"]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;',"'":'&#39;','"':'&quot;'}[c]));
  const time = (value) => {
    if (value == null || Number(value) < 0) return 'Untimed';
    const total = Math.floor(Number(value));
    const h = Math.floor(total / 3600), m = Math.floor((total % 3600) / 60), s = total % 60;
    return h ? `${h}:${String(m).padStart(2,'0')}:${String(s).padStart(2,'0')}` : `${m}:${String(s).padStart(2,'0')}`;
  };
  const bytes = (value) => {
    if (!value) return '0 B';
    const units = ['B','KB','MB','GB'];
    const i = Math.min(Math.floor(Math.log(value) / Math.log(1024)), units.length - 1);
    return `${(value / Math.pow(1024, i)).toFixed(i > 1 ? 1 : 0)} ${units[i]}`;
  };
  async function api(path, options={}) {
    const response = await fetch(path, options);
    if (!response.ok) {
      let message = `${response.status} ${response.statusText}`;
      try { message = (await response.json()).detail || message; } catch (_) {}
      throw new Error(message);
    }
    return response.json();
  }

  function renderSources() {
    $('sourceList').innerHTML = state.sources.map(source => `
      <label class="source-row" title="${esc(source.display_name)}">
        <input class="source-check" type="checkbox" data-key="${esc(source.source_key)}" ${state.selected.has(source.source_key) ? 'checked' : ''}>
        <div class="source-copy"><div class="source-name">${esc(source.display_name)}</div><div class="source-meta">${source.chunk_count} chunks · ${source.has_audio ? 'audio + transcript' : 'transcript'}</div></div>
      </label>`).join('');
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

  async function loadSources(reset=false) {
    const old = new Set(state.selected);
    state.sources = (await api('/sources')).sources || [];
    state.selected = new Set();
    state.sources.forEach(source => {
      if (reset || !old.size || old.has(source.source_key)) state.selected.add(source.source_key);
    });
    renderSources(); renderTranscriptOptions(); updateSelectedSummary();
    $('connectionBadge').classList.add('online'); $('connectionBadge').innerHTML = '<span></span>API ready';
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
      <div class="drawer-meta"><span class="meta-chip">Chunk ${esc(chunk.chunk_id)}</span><span class="meta-chip">${time(chunk.start)}${Number(chunk.end) >= 0 ? `–${time(chunk.end)}` : ''}</span><span class="meta-chip">${source?.has_audio ? 'Audio attached' : 'Transcript only'}</span></div>
      <div class="drawer-transcript">${esc(chunk.speaker_text || chunk.text || 'Transcript excerpt unavailable.')}</div>
      <div class="drawer-actions">${source?.has_audio ? '<button id="playEvidence" class="primary-button" type="button">▶ Play audio</button>' : ''}<button id="viewTranscriptEvidence" class="secondary-button" type="button">View in transcript</button></div>`;
    $('evidenceDrawer').classList.add('open');
    if ($('playEvidence')) $('playEvidence').addEventListener('click', () => playAudio(source, chunk));
    $('viewTranscriptEvidence').addEventListener('click', () => {
      if (!source) return;
      state.transcriptSource = source.source_key; renderTranscriptOptions();
      $('transcriptSearch').value = `chunk ${chunk.chunk_id}`; setView('transcript'); loadTranscript(chunk.chunk_id);
    });
  }

  function playAudio(source, chunk) {
    $('audioDock').hidden = false;
    $('audioDockTitle').textContent = source.display_name;
    $('audioDockSubtitle').textContent = `Chunk ${chunk.chunk_id} · ${time(chunk.start)}`;
    const player = $('audioPlayer');
    player.src = `/sources/${encodeURIComponent(source.source_key)}/audio`;
    const seek = () => { player.currentTime = Math.max(0, Number(chunk.start) || 0); player.play().catch(() => {}); player.removeEventListener('loadedmetadata', seek); };
    player.addEventListener('loadedmetadata', seek); player.load();
  }

  async function askQuestion(question) {
    const query = question.trim(); if (!query) return;
    const sourceKeys = state.selected.size === state.sources.length || state.selected.size === 0 ? null : Array.from(state.selected);
    const payload = {query, top_k: 6, source_keys: sourceKeys, retrieval_mode: 'global', top_k_per_source: 3};
    $('askButton').disabled = true;
    $('answerState').className = 'answer-state';
    $('answerState').innerHTML = '<div class="loading-card"><div class="loading-line"></div><div class="loading-line"></div><div class="loading-line"></div></div>';
    const [searchResult, answerResult] = await Promise.allSettled([
      api('/search',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(payload)}),
      api('/answer',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(payload)})
    ]);
    $('askButton').disabled = false;
    if (answerResult.status === 'rejected') {
      $('answerState').innerHTML = `<div class="answer-card refusal"><div class="answer-kicker">LLM unavailable</div><div class="answer-text">${esc(answerResult.reason.message)}. Retrieval remains available through the API.</div></div>`;
      return;
    }
    const answer = answerResult.value;
    const search = searchResult.status === 'fulfilled' ? searchResult.value : {results:[]};
    const lookup = new Map((search.results || []).map(chunk => [`${chunk.source_id}:${chunk.chunk_id}`, chunk]));
    const cited = (answer.citations || []).map(citation => lookup.get(citation.citation_id) || citation);
    let html = `<div class="answer-card ${answer.answerable ? '' : 'refusal'}"><div class="answer-kicker">${answer.answerable ? 'Grounded answer' : 'Not enough evidence'}</div><div class="answer-text">${esc(answer.answer)}</div></div>`;
    if (cited.length) html += `<div class="evidence-section"><div class="evidence-title">Evidence · ${cited.length}</div><div class="evidence-grid">${cited.map((chunk,i) => { const source = sourceForChunk(chunk); return `<button class="evidence-card" data-i="${i}" type="button"><div class="evidence-card-head"><div class="evidence-source">${esc(source?.display_name || chunk.source_id)}</div><div class="evidence-time">${time(chunk.start)}${Number(chunk.end)>=0?`–${time(chunk.end)}`:''}</div></div><div class="evidence-excerpt">${esc(chunk.speaker_text || chunk.text || 'Open to inspect source evidence.')}</div></button>`; }).join('')}</div></div>`;
    $('answerState').innerHTML = html;
    document.querySelectorAll('[data-i]').forEach(button => button.addEventListener('click', () => openEvidence(cited[Number(button.dataset.i)])));
  }

  async function loadTranscript(forcedChunkId=null) {
    if (!state.transcriptSource) return;
    const raw = $('transcriptSearch').value.trim();
    let chunkId = forcedChunkId, query = null;
    const match = raw.match(/^(?:chunk\s*)?#?(\d+)$/i);
    if (chunkId == null && match) chunkId = Number(match[1]); else if (raw && chunkId == null) query = raw;
    const params = new URLSearchParams({limit:'800'}); if (chunkId != null) params.set('chunk_id', chunkId); if (query) params.set('query', query);
    $('transcriptList').innerHTML = '<div class="loading-card"><div class="loading-line"></div><div class="loading-line"></div></div>';
    try {
      const data = await api(`/sources/${encodeURIComponent(state.transcriptSource)}/chunks?${params}`);
      $('transcriptMeta').textContent = `${data.total} chunk${data.total===1?'':'s'} · ${data.source.has_audio?'audio-backed':'transcript-only'} · ${data.source.has_timestamps?'timestamps available':'untimed transcript'}`;
      $('transcriptList').innerHTML = (data.chunks || []).map((chunk,i) => `<article class="chunk-row" data-chunk="${i}"><div class="chunk-time">${time(chunk.start)}</div><div class="chunk-text">${esc(chunk.speaker_text || chunk.text)}</div><div class="chunk-id">chunk ${chunk.chunk_id}</div></article>`).join('') || '<div class="empty-state"><h2>No matching chunks</h2><p>Try text, speaker, or a numeric chunk ID.</p></div>';
      document.querySelectorAll('[data-chunk]').forEach(row => row.addEventListener('click', () => openEvidence(data.chunks[Number(row.dataset.chunk)])));
    } catch (error) { $('transcriptList').innerHTML = `<div class="answer-card refusal">${esc(error.message)}</div>`; }
  }

  function pipelineFor(files) {
    const transcript = files.find(file => transcriptExts.has(ext(file.name)));
    const audio = files.find(file => audioExts.has(ext(file.name)));
    if (transcript && audio) return {mode:'both',label:'Audio + supplied transcript',action:'Import transcript',transcript,audio,nodes:[['Validate','audio + transcript'],['Parse','timestamps / speakers'],['Attach audio','retain source audio'],['Chunk','retrieval units'],['Embed','searchable corpus']]};
    if (transcript) return {mode:'transcript',label:'Supplied transcript',action:'Import transcript',transcript,audio:null,nodes:[['Validate','transcript supplied'],['Parse','timestamps / speakers'],['Skip ASR','Whisper not needed','skip'],['Chunk','retrieval units'],['Embed','searchable corpus']]};
    if (audio) return {mode:'audio',label:'Raw audio pipeline',action:'Open processed example',transcript:null,audio,nodes:[['Normalize','16 kHz mono'],['Transcribe','faster-whisper'],['Speakers','pyannote + roles'],['Chunk','retrieval units'],['Embed','searchable corpus']]};
    return null;
  }

  function renderDag(plan, active=-1, completed=-1, progress=0) {
    $('dagGraph').innerHTML = plan.nodes.map((node,index) => {
      const skipped = node[2] === 'skip', done = !skipped && index <= completed, running = !skipped && index === active, pct = done ? 100 : running ? progress : 0;
      return `<div class="dag-node ${done?'complete':''} ${running?'active':''} ${skipped?'skipped':''}"><div class="node-dot">${done?'✓':skipped?'↷':index+1}</div><div class="node-label">${esc(node[0])}</div><div class="node-progress"><span style="width:${pct}%"></span></div><div class="node-detail">${esc(node[1])}</div></div>`;
    }).join('');
  }

  function prepareImport(files) {
    const supported = files.filter(file => transcriptExts.has(ext(file.name)) || audioExts.has(ext(file.name)));
    const plan = pipelineFor(supported); if (!plan) return;
    state.importPlan = plan; state.importResult = null;
    $('fileSummary').innerHTML = supported.map(file => `<div class="file-row"><div><div class="file-kind">${transcriptExts.has(ext(file.name))?'Transcript':'Audio'}</div><div class="file-name">${esc(file.name)}</div></div><div class="file-size">${bytes(file.size)}</div></div>`).join('');
    $('ingestionPathBadge').textContent = plan.label; $('importAction').textContent = plan.action; $('overallProgress').textContent = '0%'; $('dagStatus').textContent = 'Ready to process'; renderDag(plan);
    $('etaValue').textContent = plan.mode === 'audio' ? 'Estimating from audio length…' : plan.mode === 'both' ? '~15–45 seconds' : '~10–30 seconds';
    $('importMessage').className = 'import-message';
    $('importMessage').textContent = plan.mode === 'audio' ? 'This is the real ingestion path. For the live demo, skip the multi-minute run and open a preprocessed example.' : 'Existing transcript detected. Speech recognition is skipped; parsing, chunking, and embedding make it searchable.';
    $('importModal').hidden = false;
    if (plan.mode === 'audio') estimateAudio(plan.audio);
  }

  function estimateAudio(file) {
    const url = URL.createObjectURL(file), audio = new Audio();
    audio.preload = 'metadata';
    audio.onloadedmetadata = () => { const minutes = audio.duration / 60, low = Math.max(1,Math.round(minutes*.12)), high = Math.max(low+1,Math.round(minutes*.2)); $('etaValue').textContent = `~${low}–${high} min`; URL.revokeObjectURL(url); };
    audio.onerror = () => { $('etaValue').textContent = 'A few minutes'; URL.revokeObjectURL(url); };
    audio.src = url;
  }

  async function runImport(plan) {
    if (state.importBusy) return; state.importBusy = true; $('importAction').disabled = true; $('cancelImport').disabled = true;
    const form = new FormData(); form.append('transcript', plan.transcript); if (plan.audio) form.append('audio', plan.audio); form.append('source_name', plan.transcript.name.replace(/\.[^.]+$/,''));
    let step = 0; const stages = [0,1,3,4]; renderDag(plan,stages[0],-1,35); $('overallProgress').textContent = '8%';
    const timer = setInterval(() => { step = Math.min(step+1, stages.length-1); const active = stages[step]; renderDag(plan,active,active-1,55); $('overallProgress').textContent = `${Math.min(85,18+step*22)}%`; $('dagStatus').textContent = plan.nodes[active][0]; }, 800);
    try {
      const result = await api('/ingest/transcript',{method:'POST',body:form}); clearInterval(timer); renderDag(plan,-1,plan.nodes.length-1,100); $('overallProgress').textContent='100%'; $('dagStatus').textContent='Ready to search'; $('importMessage').className='import-message success'; $('importMessage').textContent=`${result.source.display_name} is now in the searchable corpus.`; $('importAction').textContent='Open source'; $('importAction').disabled=false; state.importResult=result;
    } catch (error) { clearInterval(timer); $('importMessage').className='import-message error'; $('importMessage').textContent=error.message; $('dagStatus').textContent='Import failed'; $('importAction').disabled=false; $('cancelImport').disabled=false; }
    finally { state.importBusy=false; }
  }

  function closeImport() { if (state.importBusy) return; $('importModal').hidden = true; $('importAction').disabled=false; $('cancelImport').disabled=false; state.importPlan=null; state.importResult=null; }
  function pickFiles() { const input=document.createElement('input'); input.type='file'; input.multiple=true; input.accept='.mp3,.wav,.m4a,.flac,.ogg,.aac,.json,.srt,.vtt,.txt'; input.addEventListener('change',()=>{if(input.files?.length)prepareImport(Array.from(input.files));},{once:true}); input.click(); }

  $('sidebarToggle').addEventListener('click',()=> $('app').classList.toggle('sidebar-collapsed'));
  document.querySelectorAll('.nav-item[data-view]').forEach(item=>item.addEventListener('click',()=>setView(item.dataset.view)));
  $('queryForm').addEventListener('submit',event=>{event.preventDefault();askQuestion($('queryInput').value)});
  document.querySelectorAll('[data-query]').forEach(button=>button.addEventListener('click',()=>{$('queryInput').value=button.dataset.query;askQuestion(button.dataset.query)}));
  $('clearSources').addEventListener('click',()=>{state.selected=new Set(state.sources.map(s=>s.source_key));renderSources();updateSelectedSummary()});
  $('transcriptSource').addEventListener('change',()=>{state.transcriptSource=$('transcriptSource').value;loadTranscript()});
  $('transcriptSearchButton').addEventListener('click',()=>loadTranscript());
  $('transcriptSearch').addEventListener('keydown',event=>{if(event.key==='Enter')loadTranscript()});
  $('closeDrawer').addEventListener('click',()=> $('evidenceDrawer').classList.remove('open'));
  $('closeAudio').addEventListener('click',()=>{$('audioPlayer').pause();$('audioDock').hidden=true});
  $('addSourceButton').addEventListener('click',pickFiles); $('topAddButton').addEventListener('click',pickFiles);
  $('closeImport').addEventListener('click',closeImport); $('cancelImport').addEventListener('click',closeImport);
  $('importAction').addEventListener('click', async()=>{
    if (state.importResult) { const result=state.importResult; await loadSources(true); state.transcriptSource=result.source.source_key; closeImport(); setView('transcript'); return; }
    if (!state.importPlan) return;
    if (state.importPlan.mode === 'audio') { closeImport(); setView('ask'); return; }
    runImport(state.importPlan);
  });

  let dragDepth=0;
  window.addEventListener('dragenter',event=>{event.preventDefault();dragDepth++;$('dropOverlay').classList.add('visible')});
  window.addEventListener('dragover',event=>event.preventDefault());
  window.addEventListener('dragleave',event=>{event.preventDefault();dragDepth=Math.max(0,dragDepth-1);if(!dragDepth)$('dropOverlay').classList.remove('visible')});
  window.addEventListener('drop',event=>{event.preventDefault();dragDepth=0;$('dropOverlay').classList.remove('visible');const files=Array.from(event.dataTransfer?.files||[]);if(files.length)prepareImport(files)});

  loadSources().catch(error=>{$('connectionBadge').innerHTML='<span></span>API unavailable';$('answerState').innerHTML=`<div class="answer-card refusal">${esc(error.message)}</div>`});
})();
