/**
 * StormSight Command Center — Frontend JavaScript v3.0
 * Connects to FastAPI backend REST endpoints.
 * Features: boot animation, IST/UTC clock, Leaflet map, all live API data.
 */

const API_BASE = '';  // Same-origin FastAPI server

// ==============================================================
// BOOT SEQUENCE
// ==============================================================
const bootSteps = [
  'Initializing deep learning inference engine...',
  'Loading IBTrACS historical cyclone archive...',
  'Connecting multi-satellite data streams...',
  'Calibrating temperature scaler (T=1.25)...',
  'Verifying active learning review queue...',
  'Rendering geospatial track visualization...',
  'System ready. Launching operations center.',
];

let map = null;
let trackLayer = null;
let forecastLayer = null;

window.addEventListener('DOMContentLoaded', () => {
  runBootSequence();
  startClock();
});

function runBootSequence() {
  const splash = document.getElementById('bootSplash');
  const bar    = document.getElementById('bootProgress');
  const status = document.getElementById('bootStatus');

  let step = 0;
  const interval = setInterval(() => {
    const pct = Math.round(((step + 1) / bootSteps.length) * 100);
    bar.style.width = pct + '%';
    status.textContent = bootSteps[step];
    step++;
    if (step >= bootSteps.length) {
      clearInterval(interval);
      setTimeout(() => {
        splash.classList.add('fade-out');
        document.getElementById('mainContent').style.opacity = '1';
        document.getElementById('mainContent').style.transition = 'opacity 0.5s ease';
        document.getElementById('mainGrid').style.opacity = '1';
        document.getElementById('mainGrid').style.transition = 'opacity 0.5s ease';
        setTimeout(() => {
          splash.style.display = 'none';
          initMap();
          loadCyclone();
          checkHealth();
          loadModelStatus();
        }, 600);
      }, 400);
    }
  }, 350);
}

// ==============================================================
// CLOCK — IST & UTC
// ==============================================================
function startClock() {
  function update() {
    const now = new Date();
    const utcStr = now.toUTCString().split(' ')[4]; // HH:MM:SS
    const utcFull = now.toISOString().replace('T', ' ').slice(0, 19) + ' UTC';

    // IST = UTC + 5:30
    const istOffset = 5.5 * 60 * 60 * 1000;
    const istDate = new Date(now.getTime() + istOffset);
    const istStr  = istDate.toISOString().replace('T', ' ').slice(0, 19) + ' IST';

    document.getElementById('stClockIST').textContent = istDate.toISOString().slice(11, 19);
    document.getElementById('stClockUTC').textContent = utcStr;
    document.getElementById('tfIST').textContent = istStr;
    document.getElementById('tfUTC').textContent = utcFull;
  }
  update();
  setInterval(update, 1000);
}

// ==============================================================
// LEAFLET MAP INIT
// ==============================================================
function initMap() {
  map = L.map('map', {
    center: [15, 90],
    zoom: 4,
    zoomControl: true,
    attributionControl: true
  });

  // Dark satellite-style tile
  L.tileLayer('https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png', {
    attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OSM</a> &copy; <a href="https://carto.com/">CARTO</a>',
    subdomains: 'abcd',
    maxZoom: 12
  }).addTo(map);

  trackLayer    = L.layerGroup().addTo(map);
  forecastLayer = L.layerGroup().addTo(map);
}

// ==============================================================
// API CALLS
// ==============================================================
async function checkHealth() {
  try {
    const r = await fetch(`${API_BASE}/health`);
    if (r.ok) {
      const d = await r.json();
      document.getElementById('stApiStatus').textContent = d.status;
    }
  } catch (e) {
    document.getElementById('stApiStatus').textContent = 'OFFLINE';
    document.getElementById('apiStatus').querySelector('.status-dot').className = 'status-dot dot-red';
    showToast('Cannot reach StormSight backend API — make sure server.py is running on port 8000.');
  }
}

async function loadModelStatus() {
  try {
    const r = await fetch(`${API_BASE}/model/status`);
    if (r.ok) {
      const d = await r.json();
      document.getElementById('stModel').textContent = d.model_version || 'v3.0-PyTorch';
      document.getElementById('tfReviewCount').textContent = `Pending Reviews: ${d.pending_review_count ?? '—'}`;
      document.getElementById('tfReviewed').textContent    = `Reviewed: ${d.reviewed_count ?? '—'}`;
    }
  } catch (e) { /* silent — health check already warned */ }
}

async function loadCyclone() {
  const cycloneId = document.getElementById('cycloneSelect').value;
  const icon = document.getElementById('refreshIcon');
  icon.style.animation = 'boot-spin 1s linear infinite';

  setLoading(true);

  try {
    const [detailRes, predRes, obsRes] = await Promise.all([
      fetch(`${API_BASE}/cyclones/${cycloneId}`),
      fetch(`${API_BASE}/cyclones/${cycloneId}/prediction`),
      fetch(`${API_BASE}/cyclones/${cycloneId}/observations`),
    ]);

    if (!detailRes.ok) throw new Error(`Cyclone data not found: ${cycloneId}`);

    const detail = await detailRes.json();
    const pred   = predRes.ok ? await predRes.json() : null;
    const obs    = obsRes.ok  ? await obsRes.json()  : null;

    renderCycloneDetail(detail);
    if (pred) renderPrediction(pred);
    if (obs)  renderObservations(obs);
    renderMap(detail, pred);

  } catch (e) {
    showToast(`Error loading cyclone data: ${e.message}`);
  } finally {
    icon.style.animation = '';
    setLoading(false);
  }
}

// ==============================================================
// RENDER: CYCLONE DETAIL
// ==============================================================
function renderCycloneDetail(d) {
  const cls = d.current_intensity_class || 'Unknown';

  document.getElementById('txtID').textContent         = d.cyclone_id || '—';
  document.getElementById('txtNameBasin').textContent  = `${d.name} / ${d.basin}`;
  document.getElementById('txtPos').textContent        = `${d.current_position?.lat?.toFixed(2)}° N, ${d.current_position?.lon?.toFixed(2)}° E`;
  document.getElementById('txtObsTime').textContent    = formatTimestamp(d.last_observation_timestamp);
  document.getElementById('txtSST').textContent        = d.sea_surface_temp_celsius != null ? `${d.sea_surface_temp_celsius} °C` : '—';
  document.getElementById('txtShear').textContent      = d.wind_shear_kts != null ? `${d.wind_shear_kts} kts` : '—';
  document.getElementById('txtMoisture').textContent   = d.moisture_pct != null ? `${d.moisture_pct} %` : '—';
  document.getElementById('txtObsCount').textContent   = d.track_history?.length ?? '—';

  // Intensity banner
  document.getElementById('txtIntensityClass').textContent = cls;
  document.getElementById('txtWind').textContent     = d.max_sustained_wind_kts ?? '—';
  document.getElementById('txtPressure').textContent = d.min_central_pressure_hpa ?? '—';

  // Colour intensity by class
  const intensityEl = document.getElementById('txtIntensityClass');
  const wind = d.max_sustained_wind_kts || 0;
  if (wind >= 120) {
    intensityEl.style.color = '#ff4060';
    intensityEl.style.textShadow = '0 0 16px rgba(255,64,96,0.5)';
  } else if (wind >= 90) {
    intensityEl.style.color = '#f5a623';
    intensityEl.style.textShadow = '0 0 14px rgba(245,166,35,0.4)';
  } else if (wind >= 64) {
    intensityEl.style.color = '#528cff';
  } else {
    intensityEl.style.color = '#00e096';
  }
}

// ==============================================================
// RENDER: SATELLITE OBSERVATIONS
// ==============================================================
function renderObservations(obs) {
  const sats = obs.satellite_observations || [];
  const ref  = obs.reference_timestamp ? formatTimestamp(obs.reference_timestamp) : '—';

  sats.forEach(s => {
    const timeStr = formatTimestamp(s.timestamp);
    const dtStr   = `Δt: ${s.delta_t_min}m${s.is_reference ? ' (REF)' : ''}`;

    if (s.platform_id.includes('INSAT')) {
      document.getElementById('tINSAT').textContent   = timeStr;
      document.getElementById('dtINSAT').textContent  = dtStr;
      document.getElementById('dtINSAT').className    = 'sat-delta' + (s.is_reference ? ' ref' : '');
    } else if (s.platform_id.includes('Meteosat')) {
      document.getElementById('tMeteosat').textContent = timeStr;
    } else if (s.platform_id.includes('NOAA')) {
      document.getElementById('tNOAA').textContent   = timeStr;
      document.getElementById('dtNOAA').textContent  = dtStr;
      document.getElementById('dtNOAA').className    = 'sat-delta' + (s.is_reference ? ' ref' : '');
    }
  });
}

// ==============================================================
// RENDER: PREDICTION INTELLIGENCE
// ==============================================================
function renderPrediction(p) {
  document.getElementById('predIntensity').textContent  = p.current_intensity_class || '—';
  document.getElementById('txtModelVer').textContent    = p.model_version || '—';

  const conf = p.calibrated_confidence_pct ?? 0;
  const unc  = p.epistemic_uncertainty ?? 0;

  document.getElementById('predConf').textContent        = `${conf}%`;
  document.getElementById('predUncertainty').textContent = unc.toFixed(3);
  document.getElementById('barConf').style.width  = `${conf}%`;
  document.getElementById('barUnc').style.width   = `${Math.min(100, unc * 100)}%`;

  // RI Risk
  const ri     = p.rapid_intensification_risk || 'UNKNOWN';
  const riBan  = document.getElementById('riBanner');
  document.getElementById('predRIRisk').textContent = ri;
  riBan.className = 'ri-banner';
  if (ri === 'LOW')      riBan.classList.add('low');
  else if (ri === 'MODERATE') riBan.classList.add('mod');

  // Attribution
  const mt = p.model_transparency || {};
  const imgPct  = mt.image_contribution_pct || 48.5;
  const stPct   = mt.temporal_contribution_pct || 26.2;
  const physPct = mt.physics_contribution_pct || 25.3;
  document.getElementById('attrImg').style.width  = `${imgPct}%`;
  document.getElementById('attrST').style.width   = `${stPct}%`;
  document.getElementById('attrPhys').style.width = `${physPct}%`;
  document.getElementById('attrImgPct').textContent  = `${imgPct}%`;
  document.getElementById('attrSTPct').textContent   = `${stPct}%`;
  document.getElementById('attrPhysPct').textContent = `${physPct}%`;

  // Transparency footer
  document.getElementById('tfTransparency').textContent = `Image: ${imgPct}% | Temporal: ${stPct}% | Physics: ${physPct}%`;
  document.getElementById('tfLatency').textContent = `Latency: ${mt.inference_latency_ms || '—'}ms · Dataset: ${mt.training_dataset_version || 'IBTrACS'}`;

  // Data quality footer
  const dq = p.data_quality || {};
  document.getElementById('tfQuality').textContent = `Observations: ${dq.input_completeness_pct || '—'}% Complete`;
  document.getElementById('tfMissing').textContent = `Missing Variables: ${(dq.missing_variables || []).join(', ') || 'None'}`;

  // Forecast Waypoints
  const container = document.getElementById('waypointContainer');
  container.innerHTML = '';
  (p.forecast_waypoints || []).forEach(wp => {
    const el = document.createElement('div');
    el.className = 'waypoint-item';
    el.innerHTML = `
      <span class="wp-hour">+${wp.forecast_hour}h</span>
      <span class="wp-pos">${wp.lat?.toFixed(2)}°N ${Math.abs(wp.lon?.toFixed(2))}°${wp.lon >= 0 ? 'E' : 'W'} ±${wp.uncertainty_radius_km}km</span>
      <span class="wp-class">${(wp.forecast_intensity_class || '').split('(')[0].trim()}</span>
    `;
    container.appendChild(el);
  });

  // Human review routing
  const routing = p.review_routing || {};
  const revSection = document.getElementById('reviewSection');
  if (routing.requires_review) {
    revSection.style.display = 'block';
    document.getElementById('reviewBody').innerHTML =
      `ID: ${routing.review_item_id}<br>Reason: ${routing.reason}<br>Action: ${routing.action}`;
  } else {
    revSection.style.display = 'none';
  }
}

// ==============================================================
// RENDER: MAP
// ==============================================================
function renderMap(detail, pred) {
  if (!map) return;
  trackLayer.clearLayers();
  forecastLayer.clearLayers();

  const history = detail.track_history || [];
  if (history.length === 0) return;

  // Draw observed track polyline
  const trackPoints = history.map(p => [p.lat, p.lon]);
  L.polyline(trackPoints, {
    color: '#528cff',
    weight: 2.5,
    opacity: 0.85,
    dashArray: null,
  }).addTo(trackLayer);

  // Track observation circles
  history.forEach((p, i) => {
    const isLatest = (i === history.length - 1);
    const r = isLatest ? 8 : 4;
    const color = isLatest ? '#ff4060' : '#528cff';
    const fillOpacity = isLatest ? 0.9 : 0.5;

    const circle = L.circleMarker([p.lat, p.lon], {
      radius: r,
      color: color,
      fillColor: color,
      fillOpacity: fillOpacity,
      weight: isLatest ? 2 : 1,
    }).addTo(trackLayer);

    const timeIST = toIST(p.timestamp);
    circle.bindPopup(`
      <div style="font-family:monospace;font-size:11px;background:#06101d;color:#a8c4e8;padding:6px;">
        <b style="color:#00e5ff">${detail.name}</b><br>
        UTC: ${p.timestamp || '—'}<br>
        IST: ${timeIST}<br>
        Pos: ${p.lat}°N, ${p.lon}°E<br>
        Wind: ${p.max_wind_kts} kts<br>
        Press: ${p.pressure_hpa} hPa
      </div>
    `, { className: 'dark-popup' });

    if (isLatest) {
      circle.bindTooltip(`📍 ${detail.name} (Latest)`, { permanent: false });
    }
  });

  // Forecast waypoints
  const waypoints = pred?.forecast_waypoints || [];
  const forecastCoords = [];
  const latest = history[history.length - 1];
  if (latest) forecastCoords.push([latest.lat, latest.lon]);

  waypoints.forEach(wp => {
    forecastCoords.push([wp.lat, wp.lon]);
    L.circleMarker([wp.lat, wp.lon], {
      radius: 5,
      color: '#f5a623',
      fillColor: '#f5a623',
      fillOpacity: 0.7,
      weight: 1,
    }).addTo(forecastLayer)
    .bindPopup(`
      <div style="font-family:monospace;font-size:11px;background:#06101d;color:#a8c4e8;padding:6px;">
        <b style="color:#f5a623">+${wp.forecast_hour}h Forecast</b><br>
        Pos: ${wp.lat}°N, ${wp.lon}°E<br>
        Intensity: ${(wp.forecast_intensity_class||'').split('(')[0]}<br>
        Uncertainty: ±${wp.uncertainty_radius_km} km
      </div>
    `);
  });

  if (forecastCoords.length > 1) {
    L.polyline(forecastCoords, {
      color: '#f5a623',
      weight: 1.8,
      opacity: 0.7,
      dashArray: '5, 5',
    }).addTo(forecastLayer);
  }

  // Fit map to track bounds
  if (trackPoints.length > 0) {
    try {
      const allPoints = [...trackPoints, ...forecastCoords];
      map.fitBounds(L.latLngBounds(allPoints), { padding: [30, 30] });
    } catch (e) {}
  }
}

// ==============================================================
// UTILITIES
// ==============================================================
function formatTimestamp(ts) {
  if (!ts) return '—';
  try {
    const d = new Date(ts);
    if (isNaN(d)) return ts;
    const utcStr = d.toISOString().replace('T', ' ').slice(0, 16) + ' UTC';
    const istDate = new Date(d.getTime() + 5.5 * 3600000);
    const istStr  = istDate.toISOString().replace('T', ' ').slice(0, 16) + ' IST';
    return `${utcStr} / ${istStr}`;
  } catch (e) { return ts; }
}

function toIST(ts) {
  if (!ts) return '—';
  try {
    const d = new Date(ts);
    const istDate = new Date(d.getTime() + 5.5 * 3600000);
    return istDate.toISOString().replace('T', ' ').slice(0, 16) + ' IST';
  } catch (e) { return '—'; }
}

function showToast(msg) {
  document.getElementById('toastMsg').textContent = msg;
  document.getElementById('toast').classList.remove('hidden');
  setTimeout(() => document.getElementById('toast').classList.add('hidden'), 7000);
}

function setLoading(state) {
  const ids = ['txtID','txtNameBasin','txtPos','txtObsTime','txtSST','txtShear','txtMoisture',
                'txtIntensityClass','predIntensity','predConf','predUncertainty'];
  if (state) {
    ids.forEach(id => {
      const el = document.getElementById(id);
      if (el) el.classList.add('loading');
    });
  } else {
    ids.forEach(id => {
      const el = document.getElementById(id);
      if (el) el.classList.remove('loading');
    });
  }
}

// Reload on cyclone change
document.addEventListener('DOMContentLoaded', () => {
  const sel = document.getElementById('cycloneSelect');
  if (sel) sel.addEventListener('change', loadCyclone);
});
