// ============================================================
//  SentinelAI â€” Dashboard (Timeline + Heatmap + Evidence)
// ============================================================

const API = 'http://localhost:8000';
let inferenceEnabled = true, backendConnected = false, heatmapVisible = false;
let currentDetections = [];

function utcNow() { return new Date().toISOString().replace('T', ' ').slice(0, 19) + ' UTC'; }
async function api(path) {
  try { const r = await fetch(API + path); if (!r.ok) throw 0; return await r.json(); }
  catch (e) { return null; }
}
function setM(id, v, ph = false) { const e = document.getElementById(id); e.textContent = v; e.classList.toggle('placeholder', ph); }

// ============================================================
// 1. HEALTH POLL
// ============================================================
async function pollHealth() {
  const d = await api('/api/v1/health');
  const chip = document.getElementById('systemStatusChip'), txt = document.getElementById('systemStatusText');
  const mc = document.getElementById('modelChip'), st = document.getElementById('streamStatus');
  if (d) {
    backendConnected = true;
    chip.className = 'system-status-chip healthy'; txt.textContent = 'Healthy';
    mc.textContent = `AI: Connected Â· ${d.model}`;
    st.textContent = 'â— STREAMING (WebRTC)'; st.className = 'stream-chip streaming';
    document.getElementById('fpsValue').textContent = d.fps || '--';
    document.getElementById('auditModelInfo').textContent = `Model: ${d.model} Â· Classes: ${Object.values(d.model_classes).join(', ')} Â· Uptime: ${d.uptime_seconds}s`;
  } else {
    backendConnected = false;
    chip.className = 'system-status-chip offline'; txt.textContent = 'Offline';
    mc.textContent = 'AI: Disconnected';
    st.textContent = 'â— DISCONNECTED'; st.className = 'stream-chip';
    document.getElementById('fpsValue').textContent = '--';
    ['metricStreams', 'metricAlerts', 'metricLatency', 'metricDetections', 'metricEvidence'].forEach(id => setM(id, '--', true));
  }
  document.getElementById('lastUpdated').textContent = 'Last updated: ' + utcNow();
}
setInterval(pollHealth, 3000); pollHealth();

// ============================================================
// 2. CCTV FEED (MJPEG)
// ============================================================
const cctvCanvas = document.getElementById('cctvCanvas'), ctx = cctvCanvas.getContext('2d');
const streamImg = new Image(); let streamOk = false;
function resizeCCTV() { const p = cctvCanvas.parentElement; cctvCanvas.width = p.clientWidth; cctvCanvas.height = p.clientHeight; }
resizeCCTV(); window.addEventListener('resize', resizeCCTV);

function startStream() { streamImg.src = API + '/api/v1/stream?' + Date.now(); streamImg.onload = () => { streamOk = true }; streamImg.onerror = () => { streamOk = false; setTimeout(startStream, 5000) }; }
startStream();

function drawFeed() {
  const W = cctvCanvas.width, H = cctvCanvas.height;
  if (streamOk && streamImg.complete && streamImg.naturalWidth > 0) {
    ctx.drawImage(streamImg, 0, 0, W, H);
  } else {
    const bg = ctx.createRadialGradient(W / 2, H / 2, 50, W / 2, H / 2, W * 0.7);
    bg.addColorStop(0, '#12161f'); bg.addColorStop(1, '#090b12');
    ctx.fillStyle = bg; ctx.fillRect(0, 0, W, H);
    ctx.strokeStyle = 'rgba(0,180,255,0.04)'; ctx.lineWidth = 0.5;
    for (let x = 0; x < W; x += 40) { ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, H); ctx.stroke(); }
    for (let y = 0; y < H; y += 40) { ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(W, y); ctx.stroke(); }
    ctx.font = '13px "Inter",sans-serif'; ctx.fillStyle = 'rgba(200,220,255,0.2)'; ctx.textAlign = 'center';
    ctx.fillText('Waiting for camera stream...', W / 2, H / 2);
    ctx.font = '10px "JetBrains Mono",monospace';
    ctx.fillText('Run: python server.py', W / 2, H / 2 + 20);
    ctx.textAlign = 'left';
  }

  // ── Draw detection boxes directly on canvas ──
  if (currentDetections.length > 0) {
    const natW = streamImg.naturalWidth || W;
    const natH = streamImg.naturalHeight || H;
    const sx = W / natW, sy = H / natH;
    currentDetections.forEach(det => {
      const x = det.x1 * sx, y = det.y1 * sy;
      const bw = (det.x2 - det.x1) * sx, bh = (det.y2 - det.y1) * sy;
      const cls = det.class.toLowerCase();
      let color = '#ff3b5c';
      if (cls.includes('not_danger')) color = '#00e676';
      else if (cls === 'knife' || cls === 'pistol') color = '#ffb020';

      // Thick box
      ctx.strokeStyle = color;
      ctx.lineWidth = 3;
      ctx.strokeRect(x, y, bw, bh);

      // Corner accents
      const cl = Math.min(bw, bh) * 0.2;
      ctx.lineWidth = 5;
      ctx.beginPath(); ctx.moveTo(x, y + cl); ctx.lineTo(x, y); ctx.lineTo(x + cl, y); ctx.stroke();
      ctx.beginPath(); ctx.moveTo(x + bw - cl, y); ctx.lineTo(x + bw, y); ctx.lineTo(x + bw, y + cl); ctx.stroke();
      ctx.beginPath(); ctx.moveTo(x, y + bh - cl); ctx.lineTo(x, y + bh); ctx.lineTo(x + cl, y + bh); ctx.stroke();
      ctx.beginPath(); ctx.moveTo(x + bw - cl, y + bh); ctx.lineTo(x + bw, y + bh); ctx.lineTo(x + bw, y + bh - cl); ctx.stroke();

      // Label
      const conf = det.confidence > 0 ? ` ${(det.confidence * 100).toFixed(1)}%` : '';
      const label = det.class + conf;
      ctx.font = 'bold 13px "Inter", sans-serif';
      const tw = ctx.measureText(label).width;
      const lx = Math.min(x, W - tw - 14);
      const ly = y > 28 ? y - 6 : y + bh + 22;
      ctx.fillStyle = color;
      ctx.fillRect(lx, ly - 18, tw + 12, 22);
      ctx.fillStyle = '#050608';
      ctx.fillText(label, lx + 6, ly - 3);
    });
  }

  const ts = new Date().toISOString().replace('T', ' ').slice(0, 19);
  ctx.font = '10px "JetBrains Mono",monospace'; ctx.fillStyle = 'rgba(200,220,255,0.3)';
  ctx.fillText(`CAM-01 | ${ts} UTC | AI: ${inferenceEnabled ? 'ON' : 'OFF'}`, 10, H - 10);
  document.getElementById('feedTimestamp').textContent = `${ts} UTC · CAM-01`;
  requestAnimationFrame(drawFeed);
}
drawFeed();

// ============================================================
// 3. DETECTIONS POLL
// ============================================================
async function pollDetections() {
  if (!backendConnected) { currentDetections = []; setM('metricDetections', '--', true); return; }
  const d = await api('/api/v1/detections');
  if (!d || !d.detections) return;
  currentDetections = d.detections;
  setM('metricDetections', d.detections.length, false);
}
setInterval(pollDetections, 400);

// ============================================================
// 4. EVENTS / ALERTS
// ============================================================
async function pollEvents() {
  const d = await api('/api/v1/events');
  const list = document.getElementById('alertsList');
  if (!d || !d.events) {
    if (!backendConnected) list.innerHTML = '<div style="text-align:center;padding:24px;color:var(--text-dim);font-size:11px;">Backend offline</div>';
    return;
  }
  const ac = d.events.filter(e => e.status === 'Active').length;
  setM('metricAlerts', ac, false); setM('metricStreams', backendConnected ? '1' : '--', !backendConnected);
  document.getElementById('alertCountBadge').textContent = ac;
  list.innerHTML = '';
  if (d.events.length === 0) { list.innerHTML = '<div style="text-align:center;padding:24px;color:var(--text-dim);font-size:11px;">No events yet</div>'; return; }
  d.events.slice(0, 10).forEach((a, i) => {
    const act = a.status === 'Active';
    const c = document.createElement('div');
    c.className = `glass-panel alert-card ${act ? 'active-alert' : 'resolved-alert'} float-d${i % 4}`;
    c.innerHTML = `<div class="alert-event-id">${a.id}</div>
      <div class="alert-type-row"><span class="alert-status-dot ${act ? 'active' : 'resolved'}"></span>
      <span class="alert-type-label">${a.type}</span>
      <span class="alert-status-badge ${act ? 'active' : 'resolved'}">${a.status}</span></div>
      <div class="alert-meta-grid">
        <span class="meta-label">Camera</span><span class="meta-value">${a.camera}</span>
        <span class="meta-label">Zone</span><span class="meta-value">${a.zone}</span>
        <span class="meta-label">Class</span><span class="meta-value">${a.class || '--'}</span>
        <span class="meta-label">Confidence</span><span class="meta-value">${a.confidence ? (a.confidence * 100).toFixed(1) + '%' : '--'}</span>
      </div><div class="alert-timestamp">${a.timestamp}</div>`;
    list.appendChild(c);
  });
}
setInterval(pollEvents, 2000); pollEvents();

// ============================================================
// 5. ACTIVITY TIMELINE CHART
// ============================================================
const tlCanvas = document.getElementById('timelineCanvas'), tlCtx = tlCanvas.getContext('2d');
function resizeTl() { const p = tlCanvas.parentElement; tlCanvas.width = p.clientWidth - 32; tlCanvas.height = p.clientHeight - 28; }
resizeTl(); window.addEventListener('resize', resizeTl);

async function drawTimeline() {
  const d = await api('/api/v1/analytics/timeline');
  const W = tlCanvas.width, H = tlCanvas.height;
  tlCtx.clearRect(0, 0, W, H);
  const emp = document.getElementById('timelineEmpty');
  if (!d || !d.buckets || d.buckets.length < 2) { emp.style.display = 'flex'; return; }
  emp.style.display = 'none';
  const buckets = d.buckets;
  const maxC = Math.max(...buckets.map(b => b.count), 1);
  const stepX = W / (buckets.length - 1);

  // Grid
  tlCtx.strokeStyle = 'rgba(0,180,255,0.06)'; tlCtx.lineWidth = 0.5;
  for (let i = 0; i < 5; i++) { const y = H * (i / 4); tlCtx.beginPath(); tlCtx.moveTo(0, y); tlCtx.lineTo(W, y); tlCtx.stroke(); }

  // Area fill
  tlCtx.beginPath(); tlCtx.moveTo(0, H);
  buckets.forEach((b, i) => {
    const x = i * stepX, y = H - (b.count / maxC) * H * 0.85;
    if (i === 0) tlCtx.lineTo(x, y); else {
      const px = (i - 1) * stepX, py = H - (buckets[i - 1].count / maxC) * H * 0.85;
      const cx = (px + x) / 2; tlCtx.bezierCurveTo(cx, py, cx, y, x, y);
    }
  });
  tlCtx.lineTo(W, H); tlCtx.closePath();
  const grd = tlCtx.createLinearGradient(0, 0, 0, H);
  grd.addColorStop(0, 'rgba(0,180,255,0.15)'); grd.addColorStop(1, 'rgba(0,180,255,0)');
  tlCtx.fillStyle = grd; tlCtx.fill();

  // Line
  tlCtx.beginPath();
  buckets.forEach((b, i) => {
    const x = i * stepX, y = H - (b.count / maxC) * H * 0.85;
    if (i === 0) tlCtx.moveTo(x, y); else {
      const px = (i - 1) * stepX, py = H - (buckets[i - 1].count / maxC) * H * 0.85;
      const cx = (px + x) / 2; tlCtx.bezierCurveTo(cx, py, cx, y, x, y);
    }
  });
  tlCtx.strokeStyle = 'rgba(0,180,255,0.7)'; tlCtx.lineWidth = 2; tlCtx.stroke();

  // Dots
  buckets.forEach((b, i) => {
    const x = i * stepX, y = H - (b.count / maxC) * H * 0.85;
    tlCtx.beginPath(); tlCtx.arc(x, y, 3, 0, Math.PI * 2);
    tlCtx.fillStyle = '#00b4ff'; tlCtx.fill();
  });

  // Labels
  tlCtx.font = '9px "JetBrains Mono",monospace'; tlCtx.fillStyle = 'rgba(200,220,255,0.3)';
  tlCtx.fillText(`max: ${maxC}`, 4, 12);
  tlCtx.fillText(buckets[0].t.slice(11, 19), 4, H - 3);
  const last = buckets[buckets.length - 1].t.slice(11, 19);
  tlCtx.fillText(last, W - 55, H - 3);
}
setInterval(drawTimeline, 5000); drawTimeline();

// ============================================================
// 6. HEATMAP OVERLAY
// ============================================================
const hmCanvas = document.getElementById('heatmapCanvas'), hmCtx = hmCanvas.getContext('2d');
function resizeHm() { const p = hmCanvas.parentElement; hmCanvas.width = p.clientWidth; hmCanvas.height = p.clientHeight; }
resizeHm(); window.addEventListener('resize', resizeHm);

async function drawHeatmap() {
  if (!heatmapVisible || !backendConnected) { return; }
  const d = await api('/api/v1/analytics/heatmap');
  const W = hmCanvas.width, H = hmCanvas.height;
  hmCtx.clearRect(0, 0, W, H);
  if (!d || !d.points || d.points.length === 0) return;
  const sx = W / (streamImg.naturalWidth || W), sy = H / (streamImg.naturalHeight || H);
  d.points.forEach(p => {
    const x = p.x * sx, y = p.y * sy;
    const grd = hmCtx.createRadialGradient(x, y, 2, x, y, 30);
    grd.addColorStop(0, 'rgba(255,60,60,0.3)'); grd.addColorStop(0.5, 'rgba(255,160,0,0.12)'); grd.addColorStop(1, 'rgba(0,0,0,0)');
    hmCtx.fillStyle = grd; hmCtx.fillRect(x - 30, y - 30, 60, 60);
  });
}
setInterval(drawHeatmap, 2000);

document.getElementById('heatmapBtn').addEventListener('click', () => {
  heatmapVisible = !heatmapVisible;
  document.getElementById('heatmapBtn').classList.toggle('active', heatmapVisible);
  document.getElementById('heatmapCanvas').style.opacity = heatmapVisible ? '0.7' : '0';
  if (!heatmapVisible) hmCtx.clearRect(0, 0, hmCanvas.width, hmCanvas.height);
  else drawHeatmap();
});

// ============================================================
// 7. EVIDENCE SNAPSHOTS
// ============================================================
async function pollEvidence() {
  const d = await api('/api/v1/evidence');
  const list = document.getElementById('evidenceList');
  if (!d || !d.evidence) { setM('metricEvidence', '--', true); return; }
  setM('metricEvidence', d.total, false);
  document.getElementById('evidenceCountBadge').textContent = d.total;
  list.innerHTML = '';
  if (d.evidence.length === 0) { list.innerHTML = '<div style="text-align:center;padding:24px;color:var(--text-dim);font-size:11px;">No evidence captured yet</div>'; return; }
  d.evidence.forEach((e, i) => {
    const card = document.createElement('div');
    card.className = `glass-panel evidence-card float-d${i % 4}`;
    card.innerHTML = `<img class="evidence-thumb" src="${API}/api/v1/evidence/${e.id}" alt="Evidence ${e.id}" />
      <div class="evidence-meta">${e.id} Â· ${e.class} Â· ${e.confidence ? (e.confidence * 100).toFixed(1) + '%' : '--'}</div>
      <div style="font-size:9px;color:var(--text-dim);font-family:'JetBrains Mono',monospace;margin-top:3px">${e.timestamp}</div>`;
    card.addEventListener('click', () => showEvidence(e));
    list.appendChild(card);
  });
}
setInterval(pollEvidence, 3000); pollEvidence();

function showEvidence(e) {
  document.getElementById('evidenceModalTitle').textContent = e.id + ' â€” ' + e.event_id;
  document.getElementById('evidenceModalImg').src = API + '/api/v1/evidence/' + e.id;
  document.getElementById('evidenceModalMeta').textContent = `Class: ${e.class} Â· Confidence: ${e.confidence ? (e.confidence * 100).toFixed(1) + '%' : '--'} Â· ${e.timestamp}`;
  document.getElementById('evidenceModal').style.display = 'flex';
}
document.getElementById('closeEvidence').addEventListener('click', () => { document.getElementById('evidenceModal').style.display = 'none'; });
document.getElementById('evidenceModal').addEventListener('click', e => { if (e.target === document.getElementById('evidenceModal')) document.getElementById('evidenceModal').style.display = 'none'; });

// ============================================================
// 8. SIDEBAR TABS (Alerts / Evidence)
// ============================================================
document.querySelectorAll('.sidebar-tab').forEach(tab => {
  tab.addEventListener('click', () => {
    document.querySelectorAll('.sidebar-tab').forEach(t => t.classList.remove('active'));
    tab.classList.add('active');
    const which = tab.dataset.tab;
    document.getElementById('alertsTab').style.display = which === 'alerts' ? 'block' : 'none';
    document.getElementById('evidenceTab').style.display = which === 'evidence' ? 'block' : 'none';
  });
});

// ============================================================
// 9. AI SERVICES
// ============================================================
const services = [
  { name: 'unattended-object-service', endpoint: 'POST /api/v1/inference/unattended' },
  { name: 'stream-gateway', endpoint: 'GET  /api/v1/stream' },
  { name: 'event-processor', endpoint: 'GET  /api/v1/events' },
  { name: 'evidence-service', endpoint: 'GET  /api/v1/evidence' },
];
function renderServices() {
  const g = document.getElementById('servicesGrid'); g.innerHTML = '';
  services.forEach((s, i) => {
    const st = backendConnected ? 'healthy' : 'down', sl = backendConnected ? 'Healthy' : 'Down';
    const c = document.createElement('div'); c.className = `glass-panel service-card float-d${i % 4}`;
    c.innerHTML = `<div class="service-name">${s.name}</div><div class="service-endpoint">${s.endpoint}</div>
      <div class="service-meta"><div><div class="service-stat-label">Status</div>
      <div class="service-stat-value"><span class="status-orb ${st}"></span>${sl}</div></div>
      <div><div class="service-stat-label">Avg Latency</div><div class="service-stat-value">-- ms</div></div>
      <div><div class="service-stat-label">Queue</div><div class="service-stat-value">--</div></div></div>`;
    g.appendChild(c);
  });
}
renderServices(); setInterval(renderServices, 3000);

// ============================================================
// 10. BACKGROUND PARTICLES
// ============================================================
const bgC = document.getElementById('dataFlowCanvas'), bgX = bgC.getContext('2d');
function resizeBg() { bgC.width = window.innerWidth; bgC.height = window.innerHeight; }
resizeBg(); window.addEventListener('resize', resizeBg);
const pts = Array.from({ length: 35 }, () => ({ x: Math.random() * innerWidth, y: Math.random() * innerHeight, r: Math.random() * 1.2 + .3, vx: (Math.random() - .5) * .2, vy: (Math.random() - .5) * .2, a: Math.random() * .12 + .03 }));
function drawP() {
  bgX.clearRect(0, 0, bgC.width, bgC.height);
  pts.forEach(p => {
    p.x += p.vx; p.y += p.vy; if (p.x < 0) p.x = bgC.width; if (p.x > bgC.width) p.x = 0; if (p.y < 0) p.y = bgC.height; if (p.y > bgC.height) p.y = 0;
    bgX.beginPath(); bgX.arc(p.x, p.y, p.r, 0, Math.PI * 2); bgX.fillStyle = `rgba(0,180,255,${p.a})`; bgX.fill();
  });
  for (let i = 0; i < pts.length; i++)for (let j = i + 1; j < pts.length; j++) {
    const dx = pts[i].x - pts[j].x, dy = pts[i].y - pts[j].y, d = Math.sqrt(dx * dx + dy * dy);
    if (d < 100) { bgX.beginPath(); bgX.moveTo(pts[i].x, pts[i].y); bgX.lineTo(pts[j].x, pts[j].y); bgX.strokeStyle = `rgba(0,180,255,${.04 * (1 - d / 100)})`; bgX.stroke(); }
  }
  requestAnimationFrame(drawP);
} drawP();

// ============================================================
// 11. INFERENCE TOGGLE
// ============================================================
document.getElementById('inferenceToggle').addEventListener('click', async () => {
  if (backendConnected) await fetch(API + '/api/v1/inference/toggle', { method: 'POST' });
  inferenceEnabled = !inferenceEnabled;
  document.getElementById('inferenceToggle').classList.toggle('off', !inferenceEnabled);
  document.getElementById('inferenceLabel').textContent = inferenceEnabled ? 'Enabled' : 'Disabled';
  document.getElementById('modelChip').textContent = inferenceEnabled ? 'AI: Connected' : 'AI: Inference Paused';
});

// ============================================================
// 12. AUDIT LOG MODAL
// ============================================================
const modal = document.getElementById('auditModal');
document.getElementById('auditLogBtn').addEventListener('click', async () => {
  const d = await api('/api/v1/events'); const body = document.getElementById('auditBody'); body.innerHTML = '';
  if (d && d.events) {
    d.events.forEach(e => {
      const tr = document.createElement('tr'); tr.style.borderBottom = '1px solid var(--glass-border)';
      tr.innerHTML = `<td style="padding:5px 8px">${e.id}</td><td style="padding:5px 8px">${e.timestamp}</td><td style="padding:5px 8px">${e.class || '--'}</td><td style="padding:5px 8px">${e.confidence ? (e.confidence * 100).toFixed(1) + '%' : '--'}</td>`;
      body.appendChild(tr);
    });
  }
  else body.innerHTML = '<tr><td colspan="4" style="padding:12px;color:var(--text-dim);text-align:center">No data</td></tr>';
  modal.style.display = 'flex';
});
document.getElementById('closeAudit').addEventListener('click', () => { modal.style.display = 'none'; });
modal.addEventListener('click', e => { if (e.target === modal) modal.style.display = 'none'; });

// ============================================================
// 13. RIGHT PANEL TAB SWITCHING
// ============================================================

// Toast notification helper
function showToast(msg) {
  let t = document.createElement('div');
  t.textContent = msg;
  t.style.cssText = 'position:fixed;bottom:24px;left:50%;transform:translateX(-50%);padding:10px 24px;border-radius:10px;background:rgba(15,18,40,0.9);border:1px solid rgba(0,180,255,0.25);color:#00e5ff;font-size:12px;font-family:"Inter",sans-serif;backdrop-filter:blur(10px);z-index:9999;animation:fadeInUp 0.3s ease;';
  document.body.appendChild(t);
  setTimeout(() => { t.style.opacity = '0'; t.style.transition = 'opacity 0.3s'; setTimeout(() => t.remove(), 300); }, 2000);
}

setInterval(() => { document.getElementById('lastUpdated').textContent = 'Last sync: ' + utcNow(); }, 5000);


// ============================================================
// 15. LOGOUT
// ============================================================
document.getElementById('logoutBtn').addEventListener('click', () => {
  sessionStorage.removeItem('capcam_auth');
  window.location.href = 'login.html';
});

// Show logged-in user in header
(function () {
  try {
    const auth = JSON.parse(sessionStorage.getItem('capcam_auth'));
    if (auth && auth.user) {
      const badge = document.createElement('span');
      badge.style.cssText = 'font-size:9px;color:var(--cyan);font-family:"JetBrains Mono",monospace;padding:3px 10px;border-radius:6px;border:1px solid rgba(0,212,255,0.2);background:rgba(0,212,255,0.06);';
      badge.textContent = '● ' + auth.user;
      document.getElementById('logoutBtn').before(badge);
    }
  } catch (e) {}
})();

// Threat level auto-update based on active alerts
function updateThreatLevel() {
  const n = parseInt(document.getElementById('metricAlerts').textContent) || 0;
  const fill = document.getElementById('threatFill');
  const val  = document.getElementById('threatValue');
  if (!fill || !val) return;
  if (n === 0) {
    fill.style.width = '15%'; val.textContent = 'MONITORING';
    val.style.color = 'var(--green)';
  } else if (n <= 2) {
    fill.style.width = '45%'; val.textContent = 'ELEVATED';
    val.style.color = 'var(--amber)';
  } else if (n <= 5) {
    fill.style.width = '72%'; val.textContent = 'HIGH';
    val.style.color = 'var(--red)';
  } else {
    fill.style.width = '100%'; val.textContent = 'CRITICAL';
    val.style.color = '#ff0033';
  }
}
setInterval(updateThreatLevel, 2500); updateThreatLevel();
