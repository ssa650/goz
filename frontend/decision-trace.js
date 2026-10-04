// Public local records only. All untrusted strings are rendered as text.
export function filteredTraces(records, sessionId) {
  return sessionId ? records.filter(record => record.sessionId === sessionId) : records;
}

export function withTrackingDiagnostics(records, diagnostics) {
  return records.map(record => ({...record, trackingDiagnostics: {
    note: 'Saved detector and browser history can include events after the decision freeze.',
    records: diagnostics.filter(event => event.sessionId === record.sessionId &&
      [record.clipId,record.references?.sourceClipId].includes(event.clipId))
  }}));
}

export function renderTraces(container, records, documentObject = document) {
  container.replaceChildren();
  for (const trace of records) {
    const details = documentObject.createElement('details');
    const summary = documentObject.createElement('summary');
    summary.textContent = `Scene ${trace.sceneIndex + 1} · ${trace.action ?? 'unavailable'} · ${trace.submission} · session ${trace.sessionId}`;
    const pre = documentObject.createElement('pre');
    pre.textContent = JSON.stringify(trace, null, 2);
    details.append(summary, pre);
    container.append(details);
  }
  if (!records.length) container.textContent = 'No saved decision traces for this session.';
}

export function setupTracePanel(panel, currentSession) {
  const filter = panel.querySelector('select');
  const list = panel.querySelector('[data-trace-list]');
  const status = panel.querySelector('[role="status"]');
  let records = [], loading = false, showVersion = 0;
  const tracking = new Map();
  const enriched = () => withTrackingDiagnostics(filteredTraces(records,filter.value), [...tracking.values()].flat());
  async function show() {
    const version=++showVersion, selected=filter.value;
    if (selected && !tracking.has(selected)) {
      try {
        const response=await fetch(`/api/adaptive/tracking-diagnostics?session_id=${encodeURIComponent(selected)}`);
        if (!response.ok) throw new Error('Tracking diagnostics unavailable');
        tracking.set(selected,(await response.json()).records);
      } catch { if (version===showVersion) status.textContent='Saved decisions loaded; tracking diagnostics unavailable for this run.'; }
    }
    if (version===showVersion) renderTraces(list,enriched());
  }
  async function load() {
    if (loading) return;
    loading = true;
    status.textContent = 'Loading local decision history…';
    try {
      const response = await fetch('/api/adaptive/decision-traces');
      if (!response.ok) throw new Error('Local trace history unavailable.');
      const result = await response.json();
      records = result.records; tracking.clear();
      const selected = filter.value || currentSession() || '';
      filter.replaceChildren();
      for (const id of ['', ...new Set(records.map(record => record.sessionId))]) {
        const option = document.createElement('option');
        option.value = id; option.textContent = id || 'All runs'; filter.append(option);
      }
      filter.value = [...filter.options].some(option => option.value === selected) ? selected : '';
      status.textContent = result.warning || 'Select a run to include saved tracking and overlay history. History can extend after the decision freeze. Refresh after a run finishes.';
      await show();
    } catch (error) { status.textContent = 'Local trace history unavailable. Try Refresh.'; }
    finally { loading = false; }
  }
  panel.addEventListener('toggle', () => { if (panel.open) void load(); });
  panel.querySelector('[data-trace-refresh]').addEventListener('click', load);
  filter.addEventListener('change', show);
  panel.querySelector('[data-trace-download]').addEventListener('click', () => {
    const blob = new Blob([JSON.stringify(enriched(), null, 2)], {type:'application/json'});
    const url = URL.createObjectURL(blob), anchor = document.createElement('a');
    anchor.href = url; anchor.download = 'decision-traces.json'; anchor.click();
    URL.revokeObjectURL(url);
  });
}
