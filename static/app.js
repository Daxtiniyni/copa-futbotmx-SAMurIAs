const state = {
  analyses: [],
  current: null,
  jobTimer: null,
  calibrationPoints: [],
  videoObjectUrl: null,
};

const analysisList = document.querySelector("#analysis-list");
const emptyState = document.querySelector("#empty-state");
const analysisContent = document.querySelector("#analysis-content");
const uploadForm = document.querySelector("#upload-form");
const videoInput = document.querySelector("#video-input");
const fileLabel = document.querySelector("#file-label");
const jobStatus = document.querySelector("#job-status");
const jobMessage = document.querySelector("#job-message");
const calibrateButton = document.querySelector("#calibrate-button");
const calibrationDialog = document.querySelector("#calibration-dialog");
const calibrationVideo = document.querySelector("#calibration-video");
const calibrationCanvas = document.querySelector("#calibration-canvas");
const calibrationProgress = document.querySelector("#calibration-progress");
const calibrationInput = document.querySelector("#calibration-points");
const calibrationStatus = document.querySelector("#calibration-status");
const saveCalibrationButton = document.querySelector("#save-calibration");

function escapeHtml(value) {
  const node = document.createElement("div");
  node.textContent = value;
  return node.innerHTML;
}

async function loadAnalyses(selectFirst = false) {
  const response = await fetch("/api/analyses");
  state.analyses = await response.json();
  renderAnalysisList();
  if ((selectFirst || !state.current) && state.analyses.length) {
    await selectAnalysis(state.analyses[0].id);
  }
}

function renderAnalysisList() {
  if (!state.analyses.length) {
    analysisList.innerHTML = `<p class="analysis-empty">Aún no hay resultados.</p>`;
    return;
  }
  analysisList.innerHTML = state.analyses.map((analysis) => `
    <button class="analysis-item ${state.current?.id === analysis.id ? "active" : ""}"
      data-analysis-id="${escapeHtml(analysis.id)}">
      <strong>${escapeHtml(analysis.name)}</strong>
      <span>${analysis.frames} muestras · ${analysis.duration.toFixed(1)} s</span>
    </button>
  `).join("");
  analysisList.querySelectorAll(".analysis-item").forEach((button) => {
    button.addEventListener("click", () => selectAnalysis(button.dataset.analysisId));
  });
}

async function selectAnalysis(analysisId) {
  const response = await fetch(`/api/analyses/${encodeURIComponent(analysisId)}`);
  if (!response.ok) return;
  state.current = await response.json();
  emptyState.hidden = true;
  analysisContent.hidden = false;
  renderAnalysisList();
  renderAnalysis(state.current);
}

function renderAnalysis(analysis) {
  document.querySelector("#analysis-title").textContent = analysis.name;
  document.querySelector("#metric-frames").textContent = analysis.frames;
  document.querySelector("#metric-duration").textContent = `${analysis.duration.toFixed(1)} s`;
  document.querySelector("#metric-tracks").textContent = analysis.trajectories.filter(
    (track) => track.team !== "ball"
  ).length || analysis.max_robots;
  document.querySelector("#metric-ball").textContent = `${analysis.ball_visibility}%`;
  document.querySelector("#red-share").textContent = `${analysis.red_share}%`;
  document.querySelector("#blue-share").textContent = `${analysis.blue_share}%`;
  document.querySelector("#red-bar").style.width = `${analysis.red_share}%`;
  document.querySelector("#blue-bar").style.width = `${analysis.blue_share}%`;

  const video = document.querySelector("#analysis-video");
  video.src = `${analysis.video_url}?v=${Date.now()}`;
  document.querySelector("#heatmap-image").src = `${analysis.heatmap_url}?v=${Date.now()}`;
  document.querySelector("#tactical-eyebrow").textContent = analysis.calibrated
    ? "CAMPO CANÓNICO · HOMOGRAFÍA"
    : "DISTRIBUCIÓN ESPACIAL APROXIMADA";
  document.querySelector("#tactical-title").textContent = analysis.tracking
    ? "Trayectorias por robot"
    : "Mapa de posiciones";
  document.querySelector("#tactical-description").textContent = analysis.calibrated
    ? "Las trayectorias usan el punto inferior de cada robot proyectado al campo canónico."
    : "No existe calibración para este análisis; las posiciones están normalizadas desde la imagen.";

  const narrativeList = document.querySelector("#narrative-list");
  narrativeList.innerHTML = analysis.narrative.map((item) => `<li>${escapeHtml(item)}</li>`).join("");
  drawTacticalMap(analysis);
}

function drawField(ctx, width, height) {
  ctx.fillStyle = "#176238";
  ctx.fillRect(0, 0, width, height);
  ctx.strokeStyle = "rgba(240,247,240,.9)";
  ctx.lineWidth = 3;
  ctx.strokeRect(22, 22, width - 44, height - 44);
  ctx.beginPath();
  ctx.moveTo(width / 2, 22);
  ctx.lineTo(width / 2, height - 22);
  ctx.stroke();
  ctx.beginPath();
  ctx.arc(width / 2, height / 2, 68, 0, Math.PI * 2);
  ctx.stroke();
}

function projectCanvasPoint(ctx, point, sourceSize) {
  const [sourceWidth, sourceHeight] = sourceSize;
  return [
    22 + (point[0] / sourceWidth) * (ctx.canvas.width - 44),
    22 + (point[1] / sourceHeight) * (ctx.canvas.height - 44),
  ];
}

function drawPoints(ctx, points, color, sourceSize, radius) {
  ctx.fillStyle = color;
  for (const point of points) {
    const [px, py] = projectCanvasPoint(ctx, point, sourceSize);
    ctx.globalAlpha = 0.68;
    ctx.beginPath();
    ctx.arc(px, py, radius, 0, Math.PI * 2);
    ctx.fill();
  }
  ctx.globalAlpha = 1;
}

function drawTrajectory(ctx, trajectory, sourceSize) {
  if (!trajectory.points || trajectory.points.length < 2) return;
  const color = trajectory.team === "red_team"
    ? "#ed454f"
    : trajectory.team === "blue_team"
      ? "#3a8dfa"
      : "#f4cf38";
  ctx.strokeStyle = color;
  ctx.lineWidth = trajectory.team === "ball" ? 3 : 4;
  ctx.globalAlpha = 0.78;
  ctx.beginPath();
  trajectory.points.forEach((point, index) => {
    const [px, py] = projectCanvasPoint(ctx, point, sourceSize);
    if (index === 0) ctx.moveTo(px, py);
    else ctx.lineTo(px, py);
  });
  ctx.stroke();
  const last = projectCanvasPoint(ctx, trajectory.points.at(-1), sourceSize);
  ctx.globalAlpha = 1;
  ctx.fillStyle = color;
  ctx.beginPath();
  ctx.arc(last[0], last[1], 7, 0, Math.PI * 2);
  ctx.fill();
  ctx.fillStyle = "#fff";
  ctx.font = "bold 13px sans-serif";
  ctx.fillText(`ID ${trajectory.track_id}`, last[0] + 10, last[1] - 9);
}

function drawTacticalMap(analysis) {
  const canvas = document.querySelector("#tactical-canvas");
  const ctx = canvas.getContext("2d");
  drawField(ctx, canvas.width, canvas.height);
  if (analysis.trajectories?.length) {
    analysis.trajectories.forEach((trajectory) => {
      drawTrajectory(ctx, trajectory, analysis.source_size);
    });
  } else {
    drawPoints(ctx, analysis.positions.red_team || [], "#ed454f", analysis.source_size, 7);
    drawPoints(ctx, analysis.positions.blue_team || [], "#3a8dfa", analysis.source_size, 7);
    drawPoints(ctx, analysis.positions.ball || [], "#f4cf38", analysis.source_size, 5);
  }
}

document.querySelectorAll(".tab").forEach((tab) => {
  tab.addEventListener("click", () => {
    document.querySelectorAll(".tab").forEach((item) => item.classList.remove("active"));
    document.querySelectorAll(".tab-panel").forEach((panel) => panel.classList.remove("active"));
    tab.classList.add("active");
    document.querySelector(`#panel-${tab.dataset.tab}`).classList.add("active");
  });
});

videoInput.addEventListener("change", () => {
  const file = videoInput.files[0];
  fileLabel.textContent = file ? file.name : "Ningún archivo seleccionado";
  if (file && !document.querySelector("#analysis-name").value) {
    document.querySelector("#analysis-name").value = file.name.replace(/\.[^.]+$/, "");
  }
  calibrateButton.disabled = !file;
  state.calibrationPoints = [];
  calibrationInput.value = "";
  calibrationStatus.textContent = "Sin calibración: el mapa será aproximado.";
  calibrationStatus.classList.remove("ready");
  if (state.videoObjectUrl) URL.revokeObjectURL(state.videoObjectUrl);
  state.videoObjectUrl = file ? URL.createObjectURL(file) : null;
});

function drawCalibration() {
  const rect = calibrationCanvas.getBoundingClientRect();
  calibrationCanvas.width = Math.round(rect.width * devicePixelRatio);
  calibrationCanvas.height = Math.round(rect.height * devicePixelRatio);
  const ctx = calibrationCanvas.getContext("2d");
  ctx.scale(devicePixelRatio, devicePixelRatio);
  ctx.clearRect(0, 0, rect.width, rect.height);
  const content = calibrationContentRect(rect);
  if (state.calibrationPoints.length) {
    ctx.strokeStyle = "#49d17d";
    ctx.fillStyle = "#49d17d";
    ctx.lineWidth = 2;
    ctx.beginPath();
    state.calibrationPoints.forEach((point, index) => {
      const x = content.x + point[0] * content.width;
      const y = content.y + point[1] * content.height;
      if (index === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
      ctx.beginPath();
      ctx.arc(x, y, 7, 0, Math.PI * 2);
      ctx.fill();
      ctx.fillStyle = "#07120b";
      ctx.font = "bold 11px sans-serif";
      ctx.fillText(String(index + 1), x - 3, y + 4);
      ctx.fillStyle = "#49d17d";
    });
    if (state.calibrationPoints.length > 1) {
      ctx.beginPath();
      state.calibrationPoints.forEach((point, index) => {
        const x = content.x + point[0] * content.width;
        const y = content.y + point[1] * content.height;
        if (index === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      });
      if (state.calibrationPoints.length === 4) ctx.closePath();
      ctx.stroke();
    }
  }
  calibrationProgress.textContent = `${state.calibrationPoints.length} de 4 puntos`;
  saveCalibrationButton.disabled = state.calibrationPoints.length !== 4;
}

function calibrationContentRect(canvasRect) {
  const videoWidth = calibrationVideo.videoWidth || canvasRect.width;
  const videoHeight = calibrationVideo.videoHeight || canvasRect.height;
  const scale = Math.min(canvasRect.width / videoWidth, canvasRect.height / videoHeight);
  const width = videoWidth * scale;
  const height = videoHeight * scale;
  return {
    x: (canvasRect.width - width) / 2,
    y: (canvasRect.height - height) / 2,
    width,
    height,
  };
}

calibrateButton.addEventListener("click", async () => {
  if (!state.videoObjectUrl) return;
  calibrationVideo.src = state.videoObjectUrl;
  calibrationDialog.showModal();
  await calibrationVideo.play().catch(() => {});
  calibrationVideo.pause();
  calibrationVideo.currentTime = 0;
  requestAnimationFrame(drawCalibration);
});

calibrationVideo.addEventListener("loadedmetadata", () => {
  requestAnimationFrame(drawCalibration);
});

calibrationCanvas.addEventListener("click", (event) => {
  if (state.calibrationPoints.length >= 4) return;
  const rect = calibrationCanvas.getBoundingClientRect();
  const content = calibrationContentRect(rect);
  const x = event.clientX - rect.left - content.x;
  const y = event.clientY - rect.top - content.y;
  if (x < 0 || y < 0 || x > content.width || y > content.height) return;
  state.calibrationPoints.push([
    x / content.width,
    y / content.height,
  ]);
  drawCalibration();
});

document.querySelector("#reset-calibration").addEventListener("click", () => {
  state.calibrationPoints = [];
  drawCalibration();
});

saveCalibrationButton.addEventListener("click", () => {
  calibrationInput.value = JSON.stringify(state.calibrationPoints);
  calibrationStatus.textContent = "Cancha calibrada: se aplicará homografía.";
  calibrationStatus.classList.add("ready");
  calibrationDialog.close();
});

uploadForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  const button = uploadForm.querySelector("button[type=submit]");
  button.disabled = true;
  const formData = new FormData(uploadForm);
  const response = await fetch("/api/analyze", { method: "POST", body: formData });
  const payload = await response.json();
  if (!response.ok) {
    alert(payload.error || "No fue posible iniciar el análisis");
    button.disabled = false;
    return;
  }
  jobStatus.hidden = false;
  jobMessage.textContent = payload.message;
  monitorJob(payload.id, button);
});

function monitorJob(jobId, button) {
  clearInterval(state.jobTimer);
  state.jobTimer = setInterval(async () => {
    const response = await fetch(`/api/jobs/${jobId}`);
    const job = await response.json();
    jobMessage.textContent = job.message;
    if (job.status === "completed") {
      clearInterval(state.jobTimer);
      button.disabled = false;
      jobStatus.hidden = true;
      await loadAnalyses();
      await selectAnalysis(job.analysis_id);
    } else if (job.status === "failed") {
      clearInterval(state.jobTimer);
      button.disabled = false;
      jobStatus.hidden = true;
      alert(job.message || "El análisis falló");
    }
  }, 1800);
}

document.querySelector("#refresh-button").addEventListener("click", () => loadAnalyses());

loadAnalyses(true).catch(() => {
  analysisList.innerHTML = `<p class="analysis-empty">No se pudo conectar con el servidor.</p>`;
});
