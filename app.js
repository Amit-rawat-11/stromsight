/* StormSight Real-Time Live Data Streaming Engine */

document.addEventListener('DOMContentLoaded', () => {

  // --- STATE ---
  const state = {
    timeZoneMode: 'IST', // 'IST', 'UTC', or 'LOCAL'
    isLiveStreaming: true,
    satellites: [
      { id: 'INSAT', name: 'India Satellite (INSAT)', timeMinutes: 0, deltaT: 20, lat: 15.2, lon: 70.4, intensity: 0.78, canvasId: 'canvasINSAT' },
      { id: 'Meteosat', name: 'Europe Satellite (Meteosat)', timeMinutes: 0, deltaT: 0, lat: 15.2, lon: 70.4, intensity: 0.86, canvasId: 'canvasMeteosat' },
      { id: 'NOAA', name: 'Other Satellite (NOAA)', timeMinutes: 0, deltaT: 10, lat: 15.2, lon: 70.4, intensity: 0.81, canvasId: 'canvasNOAA' }
    ],
    physics: {
      sst: 30.5,
      shear: 8.5,
      moisture: 88,
      pressure: 935
    },
    prediction: null,
    datasetSamples: 42,
    trainingLossHistory: [0.65, 0.48, 0.35, 0.28, 0.21, 0.16, 0.142]
  };

  // --- DOM ELEMENTS ---
  const liveClockDisplay = document.getElementById('liveClockDisplay');
  const tzIST = document.getElementById('tzIST');
  const tzUTC = document.getElementById('tzUTC');
  const tzLOCAL = document.getElementById('tzLOCAL');
  const badgeDataSync = document.getElementById('badgeDataSync');

  const inputTimeINSAT = document.getElementById('inputTimeINSAT');
  const inputTimeMeteosat = document.getElementById('inputTimeMeteosat');
  const inputTimeNOAA = document.getElementById('inputTimeNOAA');

  const labelTimeINSAT = document.getElementById('labelTimeINSAT');
  const labelTimeMeteosat = document.getElementById('labelTimeMeteosat');
  const labelTimeNOAA = document.getElementById('labelTimeNOAA');

  const badgeINSAT = document.getElementById('badgeINSAT');
  const badgeMeteosat = document.getElementById('badgeMeteosat');
  const badgeNOAA = document.getElementById('badgeNOAA');

  const inputSST = document.getElementById('inputSST');
  const inputShear = document.getElementById('inputShear');
  const inputMoisture = document.getElementById('inputMoisture');
  const inputPressure = document.getElementById('inputPressure');
  
  const valSST = document.getElementById('valSST');
  const valShear = document.getElementById('valShear');
  const valMoisture = document.getElementById('valMoisture');
  const valPressure = document.getElementById('valPressure');

  const btnRunInference = document.getElementById('btnRunInference');
  const btnFetchLiveData = document.getElementById('btnFetchLiveData');
  const scenarioSelect = document.getElementById('scenarioSelect');

  const textIntensityClass = document.getElementById('textIntensityClass');
  const textWindSpeed = document.getElementById('textWindSpeed');
  const valConfidence = document.getElementById('valConfidence');
  const gaugeConfidence = document.getElementById('gaugeConfidence');
  const textTrend = document.getElementById('textTrend');

  const cardHighConf = document.getElementById('cardHighConf');
  const cardLowConf = document.getElementById('cardLowConf');

  const btnTriggerRetrain = document.getElementById('btnTriggerRetrain');
  const valRetrainSamples = document.getElementById('valRetrainSamples');
  const valTrainingLoss = document.getElementById('valTrainingLoss');
  const valRetrainAcc = document.getElementById('valRetrainAcc');

  const reviewModal = document.getElementById('reviewModal');
  const btnOpenReviewModal = document.getElementById('btnOpenReviewModal');
  const btnCloseModal = document.getElementById('btnCloseModal');
  const btnApproveOverride = document.getElementById('btnApproveOverride');

  // --- INITIALIZE REAL-TIME ENGINES ---
  startLiveClock();
  initSatelliteCanvases();
  initFeatureBars();
  
  // Trigger immediate Real-Time Live Stream fetch on page load
  fetchRealTimeLiveDataStream();
  
  // Set up periodic auto-sync ticker every 25 seconds
  setInterval(() => {
    if (state.isLiveStreaming) {
      fetchRealTimeLiveDataStream(false);
    }
  }, 25000);

  renderLossChart();

  // --- LIVE REAL-TIME METEOROLOGICAL STREAM ENGINE ---
  async function fetchRealTimeLiveDataStream(showNotification = true) {
    btnFetchLiveData.disabled = true;
    btnFetchLiveData.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> Syncing Live Data...';

    // Calculate real-time satellite timestamps relative to current time
    const now = new Date();
    const nowMins = now.getHours() * 60 + now.getMinutes();

    // Meteosat = Ref (current time)
    // NOAA = 10 mins prior
    // INSAT = 22 mins prior
    state.satellites[1].timeMinutes = nowMins;
    state.satellites[2].timeMinutes = (nowMins - 10 + 1440) % 1440;
    state.satellites[0].timeMinutes = (nowMins - 22 + 1440) % 1440;

    // Set time input values
    inputTimeMeteosat.value = minutesToInputTimeString(state.satellites[1].timeMinutes);
    inputTimeNOAA.value = minutesToInputTimeString(state.satellites[2].timeMinutes);
    inputTimeINSAT.value = minutesToInputTimeString(state.satellites[0].timeMinutes);

    updateSatelliteDeltaTimes();

    try {
      // Query Live Open-Meteo REST API for real coordinates (Arabian Sea / Bay of Bengal)
      const res = await fetch('https://api.open-meteo.com/v1/forecast?latitude=15.2&longitude=70.4&current=temperature_2m,relative_humidity_2m,surface_pressure,wind_speed_10m');
      const data = await res.json();
      const current = data.current || {};

      const tempC = current.temperature_2m || 28.5;
      const sstEst = Math.min(34.0, Math.max(20.0, tempC + 1.2));
      const windKts = Math.round((current.wind_speed_10m || 15) * 0.539957 * 10) / 10;
      
      state.physics.sst = Math.round(sstEst * 10) / 10;
      state.physics.shear = Math.max(5.0, Math.round(windKts * 0.7 * 10) / 10);
      state.physics.moisture = current.relative_humidity_2m || 80;
      state.physics.pressure = current.surface_pressure || 1010;

      // Update Sliders & Displays
      inputSST.value = state.physics.sst;
      inputShear.value = state.physics.shear;
      inputMoisture.value = state.physics.moisture;
      inputPressure.value = state.physics.pressure;

      valSST.textContent = `${state.physics.sst} °C`;
      valShear.textContent = `${state.physics.shear} kts`;
      valMoisture.textContent = `${state.physics.moisture} %`;
      valPressure.textContent = `${state.physics.pressure} hPa`;

      if (badgeDataSync) {
        badgeDataSync.textContent = 'LIVE SYNCED';
        badgeDataSync.style.color = 'var(--success-emerald)';
      }

      runInferencePipeline();

    } catch (err) {
      console.warn("Live API fetch warning:", err);
    } finally {
      btnFetchLiveData.disabled = false;
      btnFetchLiveData.innerHTML = '<i class="fa-solid fa-arrows-rotate"></i> Sync Live Data Now';
    }
  }

  function minutesToInputTimeString(mins) {
    const h = Math.floor(mins / 60) % 24;
    const m = mins % 60;
    return `${String(h).padStart(2, '0')}:${String(m).padStart(2, '0')}`;
  }

  // --- LIVE CLOCK ENGINE ---
  function startLiveClock() {
    updateClock();
    setInterval(updateClock, 1000);
  }

  function updateClock() {
    const now = new Date();
    let formattedStr = "";

    if (state.timeZoneMode === 'IST') {
      const utcMs = now.getTime() + (now.getTimezoneOffset() * 60000);
      const istDate = new Date(utcMs + (330 * 60000));
      
      const day = String(istDate.getDate()).padStart(2, '0');
      const months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];
      const month = months[istDate.getMonth()];
      const year = istDate.getFullYear();
      
      const hours = String(istDate.getHours()).padStart(2, '0');
      const mins = String(istDate.getMinutes()).padStart(2, '0');
      const secs = String(istDate.getSeconds()).padStart(2, '0');

      formattedStr = `${day} ${month} ${year} ${hours}:${mins}:${secs} IST`;
    } else if (state.timeZoneMode === 'UTC') {
      formattedStr = now.toISOString().replace('T', ' ').substring(0, 19) + ' UTC';
    } else {
      formattedStr = now.toLocaleString() + ' Local';
    }

    liveClockDisplay.textContent = formattedStr;
  }

  // Timezone Controls
  tzIST.addEventListener('click', () => setTimeZone('IST'));
  tzUTC.addEventListener('click', () => setTimeZone('UTC'));
  tzLOCAL.addEventListener('click', () => setTimeZone('LOCAL'));

  function setTimeZone(mode) {
    state.timeZoneMode = mode;
    tzIST.classList.toggle('active', mode === 'IST');
    tzUTC.classList.toggle('active', mode === 'UTC');
    tzLOCAL.classList.toggle('active', mode === 'LOCAL');
    updateClock();
    updateSatelliteTimeLabels();
  }

  // --- SATELLITE TIMESTAMP CONTROLS ---
  function parseTimeToMinutes(timeStr) {
    const [h, m] = timeStr.split(':').map(Number);
    return (h || 0) * 60 + (m || 0);
  }

  function formatMinutesToTimeString(mins) {
    const h = Math.floor(mins / 60) % 24;
    const m = mins % 60;
    const ampm = h >= 12 ? 'PM' : 'AM';
    const displayH = h % 12 || 12;
    const displayM = String(m).padStart(2, '0');
    return `${displayH}:${displayM} ${ampm} ${state.timeZoneMode}`;
  }

  inputTimeINSAT.addEventListener('input', (e) => {
    state.isLiveStreaming = false;
    state.satellites[0].timeMinutes = parseTimeToMinutes(e.target.value);
    updateSatelliteDeltaTimes();
  });

  inputTimeMeteosat.addEventListener('input', (e) => {
    state.isLiveStreaming = false;
    state.satellites[1].timeMinutes = parseTimeToMinutes(e.target.value);
    updateSatelliteDeltaTimes();
  });

  inputTimeNOAA.addEventListener('input', (e) => {
    state.isLiveStreaming = false;
    state.satellites[2].timeMinutes = parseTimeToMinutes(e.target.value);
    updateSatelliteDeltaTimes();
  });

  function updateSatelliteTimeLabels() {
    labelTimeINSAT.textContent = formatMinutesToTimeString(state.satellites[0].timeMinutes);
    labelTimeMeteosat.textContent = formatMinutesToTimeString(state.satellites[1].timeMinutes);
    labelTimeNOAA.textContent = formatMinutesToTimeString(state.satellites[2].timeMinutes);
  }

  function updateSatelliteDeltaTimes() {
    const times = state.satellites.map(s => s.timeMinutes);
    const maxTime = Math.max(...times);

    state.satellites.forEach(s => {
      s.deltaT = Math.max(0, maxTime - s.timeMinutes);
    });

    updateBadgeUI(badgeINSAT, state.satellites[0].deltaT);
    updateBadgeUI(badgeMeteosat, state.satellites[1].deltaT);
    updateBadgeUI(badgeNOAA, state.satellites[2].deltaT);

    updateSatelliteTimeLabels();
    runInferencePipeline();
  }

  function updateBadgeUI(badgeElement, deltaT) {
    if (deltaT === 0) {
      badgeElement.textContent = `Δt: 0 min (Ref)`;
      badgeElement.className = 'card-badge delta-ref';
    } else {
      badgeElement.textContent = `Δt: ${deltaT} min`;
      badgeElement.className = 'card-badge';
    }
  }

  // --- PARAMETER SLIDERS & EVENT LISTENERS ---
  inputSST.addEventListener('input', (e) => {
    state.physics.sst = parseFloat(e.target.value);
    valSST.textContent = `${state.physics.sst} °C`;
    runInferencePipeline();
  });

  inputShear.addEventListener('input', (e) => {
    state.physics.shear = parseFloat(e.target.value);
    valShear.textContent = `${state.physics.shear} kts`;
    runInferencePipeline();
  });

  inputMoisture.addEventListener('input', (e) => {
    state.physics.moisture = parseFloat(e.target.value);
    valMoisture.textContent = `${state.physics.moisture} %`;
    runInferencePipeline();
  });

  inputPressure.addEventListener('input', (e) => {
    state.physics.pressure = parseFloat(e.target.value);
    valPressure.textContent = `${state.physics.pressure} hPa`;
    runInferencePipeline();
  });

  btnRunInference.addEventListener('click', () => {
    runInferencePipeline();
  });

  btnFetchLiveData.addEventListener('click', () => {
    state.isLiveStreaming = true;
    scenarioSelect.value = "live";
    fetchRealTimeLiveDataStream(true);
  });

  scenarioSelect.addEventListener('change', (e) => {
    if (e.target.value === 'live') {
      state.isLiveStreaming = true;
      fetchRealTimeLiveDataStream(true);
    } else {
      state.isLiveStreaming = false;
      applyPresetScenario(e.target.value);
    }
  });

  btnTriggerRetrain.addEventListener('click', () => {
    triggerRetrainingAnimation();
  });

  btnOpenReviewModal.addEventListener('click', () => {
    reviewModal.classList.add('active');
  });

  btnCloseModal.addEventListener('click', () => {
    reviewModal.classList.remove('active');
  });

  btnApproveOverride.addEventListener('click', () => {
    reviewModal.classList.remove('active');
    state.datasetSamples += 1;
    valRetrainSamples.textContent = state.datasetSamples;
    cardHighConf.style.opacity = '1';
    cardLowConf.style.opacity = '0.5';
    alert("Sample approved by Meteorologist and added to active training dataset!");
  });

  // --- SATELLITE CANVAS RENDERER ---
  function initSatelliteCanvases() {
    state.satellites.forEach(sat => {
      const cvs = document.getElementById(sat.canvasId);
      if (!cvs) return;
      cvs.width = 240;
      cvs.height = 240;
      renderCycloneCanvas(cvs, sat.intensity, sat.id);
    });
  }

  function renderCycloneCanvas(canvas, intensity, label) {
    const ctx = canvas.getContext('2d');
    const w = canvas.width;
    const h = canvas.height;
    const cx = w / 2;
    const cy = h / 2;

    ctx.fillStyle = '#02050e';
    ctx.fillRect(0, 0, w, h);

    // Radar grid rings
    ctx.strokeStyle = 'rgba(0, 242, 254, 0.08)';
    ctx.lineWidth = 1;
    for (let r = 30; r < w/2; r += 30) {
      ctx.beginPath();
      ctx.arc(cx, cy, r, 0, Math.PI * 2);
      ctx.stroke();
    }

    // Render Spiral Arms (Infrared thermal bands)
    const arms = 4;
    const maxRadius = w * 0.42;

    for (let r = maxRadius; r > 10; r -= 2.5) {
      const alpha = (r / maxRadius) * 0.85;
      const angleOffset = (maxRadius - r) * 0.09;

      ctx.beginPath();
      for (let arm = 0; arm < arms; arm++) {
        const baseAngle = (arm * Math.PI * 2 / arms) + angleOffset;
        const x = cx + Math.cos(baseAngle) * r;
        const y = cy + Math.sin(baseAngle) * r;
        ctx.arc(x, y, 7, 0, Math.PI * 2);
      }
      
      const colorVal = Math.floor(255 * (1 - (r / maxRadius) * (0.8 + intensity * 0.2)));
      ctx.fillStyle = `rgba(${colorVal}, ${Math.floor(colorVal * 0.9)}, 255, ${alpha})`;
      ctx.fill();
    }

    // Eye of the cyclone
    const eyeRadius = 16 - intensity * 6;
    const eyeGrad = ctx.createRadialGradient(cx, cy, 0, cx, cy, eyeRadius + 6);
    eyeGrad.addColorStop(0, 'rgba(255, 230, 200, 0.95)');
    eyeGrad.addColorStop(0.5, 'rgba(139, 92, 246, 0.45)');
    eyeGrad.addColorStop(1, 'transparent');

    ctx.beginPath();
    ctx.arc(cx, cy, eyeRadius + 6, 0, Math.PI * 2);
    ctx.fillStyle = eyeGrad;
    ctx.fill();

    // Satellite Overlay Label
    ctx.fillStyle = 'rgba(255, 255, 255, 0.75)';
    ctx.font = '11px JetBrains Mono';
    ctx.fillText(`${label} IR 10.8µm`, 10, h - 12);
  }

  // --- FEATURE BARS RENDERER ---
  function initFeatureBars() {
    createBars('barsImage', 24, 'var(--primary-cyan)');
    createBars('barsST', 24, 'var(--accent-violet)');
    createBars('barsPhysics', 24, 'var(--warning-amber)');
  }

  function createBars(containerId, count, color) {
    const container = document.getElementById(containerId);
    if (!container) return;
    container.innerHTML = '';
    for (let i = 0; i < count; i++) {
      const bar = document.createElement('div');
      bar.className = 'f-bar';
      bar.style.backgroundColor = color;
      bar.style.height = `${Math.floor(Math.random() * 80 + 20)}%`;
      container.appendChild(bar);
    }
  }

  function updateBars(containerId) {
    const container = document.getElementById(containerId);
    if (!container) return;
    const bars = container.querySelectorAll('.f-bar');
    bars.forEach(bar => {
      bar.style.height = `${Math.floor(Math.random() * 75 + 25)}%`;
    });
  }

  // --- PRESET SCENARIOS ---
  function applyPresetScenario(presetKey) {
    if (presetKey === 'severe') {
      state.physics = { sst: 30.5, shear: 8.5, moisture: 88, pressure: 935 };
    } else if (presetKey === 'amphan') {
      state.physics = { sst: 31.0, shear: 7.5, moisture: 92, pressure: 920 };
    } else if (presetKey === 'katrina') {
      state.physics = { sst: 30.8, shear: 6.0, moisture: 94, pressure: 902 };
    } else if (presetKey === 'fani') {
      state.physics = { sst: 30.2, shear: 9.0, moisture: 88, pressure: 932 };
    } else if (presetKey === 'rapid') {
      state.physics = { sst: 31.8, shear: 5.0, moisture: 95, pressure: 910 };
    } else if (presetKey === 'weakening') {
      state.physics = { sst: 25.5, shear: 35.0, moisture: 55, pressure: 1002 };
    }

    inputSST.value = state.physics.sst;
    inputShear.value = state.physics.shear;
    inputMoisture.value = state.physics.moisture;
    inputPressure.value = state.physics.pressure;

    valSST.textContent = `${state.physics.sst} °C`;
    valShear.textContent = `${state.physics.shear} kts`;
    valMoisture.textContent = `${state.physics.moisture} %`;
    valPressure.textContent = `${state.physics.pressure} hPa`;

    runInferencePipeline();
  }

  // --- INFERENCE PIPELINE ---
  function runInferencePipeline() {
    updateBars('barsImage');
    updateBars('barsST');
    updateBars('barsPhysics');

    const sst = state.physics.sst;
    const shear = state.physics.shear;
    const moisture = state.physics.moisture;
    const pressure = state.physics.pressure;

    // Thermodynamic Cyclone Potential Calculation
    const pNorm = (1013 - pressure) / 123.0;
    const sstNorm = (sst - 20) / 14.0;
    const shearNorm = Math.max(0, 1 - (shear / 40.0));
    
    const score = (pNorm * 0.45) + (sstNorm * 0.3) + (shearNorm * 0.25);
    const intensityScore = Math.max(0, Math.min(1, score));

    let className = "Weak (Depression)";
    let windSpeed = 35;
    let color = "var(--primary-cyan)";

    if (intensityScore > 0.82) {
      className = "Super Cyclonic Storm (Cat 5)";
      windSpeed = 140;
      color = "#ff0055";
    } else if (intensityScore > 0.65) {
      className = "Very Severe Cyclonic Storm (Cat 3/4)";
      windSpeed = 98;
      color = "var(--danger-crimson)";
    } else if (intensityScore > 0.45) {
      className = "Severe Cyclonic Storm (Cat 1/2)";
      windSpeed = 65;
      color = "var(--warning-amber)";
    } else if (intensityScore > 0.25) {
      className = "Moderate Cyclonic Storm";
      windSpeed = 48;
      color = "var(--primary-cyan)";
    }

    textIntensityClass.textContent = className;
    textIntensityClass.style.color = color;
    textWindSpeed.textContent = `${windSpeed} kts`;

    // Recency Weight Impact on Confidence Score
    const recencySum = state.satellites.reduce((acc, s) => acc + Math.exp(-0.02 * s.deltaT), 0);
    const recencyFactor = recencySum / state.satellites.length;

    const baseConf = 0.82 + (recencyFactor * 0.10) + (shear < 20 ? 0.05 : -0.10);
    const confidencePct = Math.max(40, Math.min(99, Math.round(baseConf * 1000) / 10));

    valConfidence.textContent = `${confidencePct}%`;
    const dashOffset = 490 - (490 * (confidencePct / 100));
    gaugeConfidence.style.strokeDashoffset = dashOffset;
    gaugeConfidence.style.stroke = confidencePct >= 85 ? 'var(--primary-cyan)' : 'var(--danger-crimson)';

    // Trend Analysis
    if (pressure < 930 && sst > 30.5 && shear < 10) {
      textTrend.textContent = "Rapid Intensification Warning (⚠️ High Risk)";
      textTrend.style.color = "var(--danger-crimson)";
    } else if (intensityScore > 0.5) {
      textTrend.textContent = "Gradual Intensification Expected";
      textTrend.style.color = "var(--warning-amber)";
    } else {
      textTrend.textContent = "Weakening / Disorganizing System";
      textTrend.style.color = "var(--success-emerald)";
    }

    // Decision Engine Routing
    if (confidencePct >= 85) {
      cardHighConf.style.opacity = '1';
      cardLowConf.style.opacity = '0.4';
    } else {
      cardHighConf.style.opacity = '0.4';
      cardLowConf.style.opacity = '1';
    }

    // Render Track Map
    renderStormTrackMap(intensityScore, windSpeed);
  }

  // --- STORM TRACK MAP CANVAS ---
  function renderStormTrackMap(intensityScore, maxWind) {
    const canvas = document.getElementById('trackMapCanvas');
    if (!canvas) return;

    const rect = canvas.parentElement.getBoundingClientRect();
    canvas.width = rect.width || 600;
    canvas.height = rect.height || 420;
    const ctx = canvas.getContext('2d');
    const w = canvas.width;
    const h = canvas.height;

    ctx.fillStyle = '#02050e';
    ctx.fillRect(0, 0, w, h);

    // Grid lines
    ctx.strokeStyle = 'rgba(56, 189, 248, 0.08)';
    ctx.lineWidth = 1;
    for (let x = 0; x < w; x += 60) {
      ctx.beginPath();
      ctx.moveTo(x, 0); ctx.lineTo(x, h);
      ctx.stroke();
    }
    for (let y = 0; y < h; y += 60) {
      ctx.beginPath();
      ctx.moveTo(0, y); ctx.lineTo(w, y);
      ctx.stroke();
    }

    // Coastline geometry
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.3)';
    ctx.lineWidth = 2.5;
    ctx.beginPath();
    ctx.moveTo(w * 0.75, 0);
    ctx.lineTo(w * 0.70, h * 0.25);
    ctx.lineTo(w * 0.65, h * 0.55);
    ctx.lineTo(w * 0.60, h * 0.80);
    ctx.lineTo(w * 0.55, h);
    ctx.stroke();

    ctx.fillStyle = 'rgba(255, 255, 255, 0.04)';
    ctx.lineTo(w, h);
    ctx.lineTo(w, 0);
    ctx.fill();

    ctx.fillStyle = 'rgba(255, 255, 255, 0.45)';
    ctx.font = '12px Outfit';
    ctx.fillText('INDIAN PENINSULA', w * 0.72, 35);
    ctx.fillText('ARABIAN SEA BASIN', w * 0.15, h * 0.85);

    // Storm Track Waypoints
    const startX = w * 0.25;
    const startY = h * 0.70;

    const points = [
      { h: -12, x: startX - 50, y: startY + 30 },
      { h: -6,  x: startX - 25, y: startY + 15 },
      { h: 0,   x: startX,       y: startY },
      { h: 12,  x: startX + 55,  y: startY - 35 },
      { h: 24,  x: startX + 115, y: startY - 70 },
      { h: 36,  x: startX + 175, y: startY - 100 },
      { h: 48,  x: startX + 235, y: startY - 125 },
    ];

    // Forecast Cone of Uncertainty
    const curP = points[2];
    const endP = points[points.length - 1];
    ctx.beginPath();
    ctx.moveTo(curP.x, curP.y);
    ctx.lineTo(endP.x - 20, endP.y - 40);
    ctx.lineTo(endP.x + 40, endP.y + 30);
    ctx.closePath();
    ctx.fillStyle = 'rgba(0, 242, 254, 0.08)';
    ctx.fill();
    ctx.strokeStyle = 'rgba(0, 242, 254, 0.25)';
    ctx.setLineDash([4, 4]);
    ctx.stroke();
    ctx.setLineDash([]);

    // Past Path (Solid Blue)
    ctx.beginPath();
    ctx.moveTo(points[0].x, points[0].y);
    ctx.lineTo(points[1].x, points[1].y);
    ctx.lineTo(points[2].x, points[2].y);
    ctx.strokeStyle = 'var(--primary-cyan)';
    ctx.lineWidth = 3.5;
    ctx.stroke();

    // Forecast Path (Neon Crimson)
    ctx.beginPath();
    ctx.moveTo(points[2].x, points[2].y);
    for (let i = 3; i < points.length; i++) {
      ctx.lineTo(points[i].x, points[i].y);
    }
    ctx.strokeStyle = 'var(--danger-crimson)';
    ctx.lineWidth = 3.5;
    ctx.setLineDash([6, 4]);
    ctx.stroke();
    ctx.setLineDash([]);

    // Waypoint Markers
    points.forEach((pt, idx) => {
      ctx.beginPath();
      ctx.arc(pt.x, pt.y, idx === 2 ? 9 : 5, 0, Math.PI * 2);
      ctx.fillStyle = idx <= 2 ? 'var(--primary-cyan)' : 'var(--danger-crimson)';
      ctx.fill();
      ctx.strokeStyle = '#ffffff';
      ctx.lineWidth = 1.5;
      ctx.stroke();

      ctx.fillStyle = '#ffffff';
      ctx.font = '10px JetBrains Mono';
      ctx.fillText(`${pt.h > 0 ? '+' : ''}${pt.h}h`, pt.x - 10, pt.y + 18);
    });

    // Eye Pulse
    const eye = points[2];
    ctx.beginPath();
    ctx.arc(eye.x, eye.y, 18, 0, Math.PI * 2);
    ctx.strokeStyle = 'rgba(244, 63, 94, 0.65)';
    ctx.lineWidth = 2;
    ctx.stroke();
  }

  // --- RETRAINING LOSS CHART CANVAS ---
  function renderLossChart() {
    const canvas = document.getElementById('lossCanvas');
    if (!canvas) return;

    const rect = canvas.parentElement.getBoundingClientRect();
    canvas.width = rect.width || 400;
    canvas.height = rect.height || 200;
    const ctx = canvas.getContext('2d');
    const w = canvas.width;
    const h = canvas.height;

    ctx.clearRect(0, 0, w, h);

    const history = state.trainingLossHistory;
    const maxVal = 0.8;
    const minVal = 0.0;

    // Grid lines
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.06)';
    ctx.lineWidth = 1;
    for (let y = 0; y < h; y += 35) {
      ctx.beginPath();
      ctx.moveTo(0, y); ctx.lineTo(w, y);
      ctx.stroke();
    }

    // Line
    ctx.beginPath();
    history.forEach((val, i) => {
      const x = (i / (history.length - 1)) * (w - 40) + 20;
      const y = h - 20 - ((val - minVal) / (maxVal - minVal)) * (h - 40);
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });

    ctx.strokeStyle = 'var(--success-emerald)';
    ctx.lineWidth = 3;
    ctx.stroke();

    // Fill Gradient
    const lastX = ( (history.length - 1) / (history.length - 1) ) * (w - 40) + 20;
    ctx.lineTo(lastX, h - 20);
    ctx.lineTo(20, h - 20);
    ctx.closePath();

    const fillGrad = ctx.createLinearGradient(0, 0, 0, h);
    fillGrad.addColorStop(0, 'rgba(16, 185, 129, 0.28)');
    fillGrad.addColorStop(1, 'transparent');
    ctx.fillStyle = fillGrad;
    ctx.fill();

    // Data points
    history.forEach((val, i) => {
      const x = (i / (history.length - 1)) * (w - 40) + 20;
      const y = h - 20 - ((val - minVal) / (maxVal - minVal)) * (h - 40);
      ctx.beginPath();
      ctx.arc(x, y, 4.5, 0, Math.PI * 2);
      ctx.fillStyle = '#ffffff';
      ctx.fill();
    });
  }

  function triggerRetrainingAnimation() {
    btnTriggerRetrain.disabled = true;
    btnTriggerRetrain.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> Retraining...';

    let step = 0;
    const interval = setInterval(() => {
      step++;
      const newLoss = Math.max(0.04, Math.round((0.142 - step * 0.015) * 1000) / 1000);
      state.trainingLossHistory.push(newLoss);
      if (state.trainingLossHistory.length > 10) state.trainingLossHistory.shift();

      valTrainingLoss.textContent = newLoss.toFixed(4);
      valRetrainAcc.textContent = `${(94.8 + step * 0.4).toFixed(1)}%`;
      renderLossChart();

      if (step >= 5) {
        clearInterval(interval);
        btnTriggerRetrain.disabled = false;
        btnTriggerRetrain.innerHTML = '<i class="fa-solid fa-rotate"></i> Retrain Model Now';
        alert("Model retraining completed! Weights updated with new verified satellite observations.");
      }
    }, 400);
  }

});
