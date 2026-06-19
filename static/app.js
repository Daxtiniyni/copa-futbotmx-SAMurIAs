const state = {
  analyses: [],
  current: null,
  jobTimer: null,
};

const analysisList = document.querySelector("#analysis-list");
const emptyState = document.querySelector("#empty-state");
const analysisContent = document.querySelector("#analysis-content");
const uploadForm = document.querySelector("#upload-form");
const videoInput = document.querySelector("#video-input");
const fileLabel = document.querySelector("#file-label");
const jobStatus = document.querySelector("#job-status");
const jobMessage = document.querySelector("#job-message");

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
  document.querySelector("#metric-robots").textContent = analysis.max_robots;
  document.querySelector("#metric-ball").textContent = `${analysis.ball_visibility}%`;
  document.querySelector("#red-share").textContent = `${analysis.red_share}%`;
  document.querySelector("#blue-share").textContent = `${analysis.blue_share}%`;
  document.querySelector("#red-bar").style.width = `${analysis.red_share}%`;
  document.querySelector("#blue-bar").style.width = `${analysis.blue_share}%`;

  const video = document.querySelector("#analysis-video");
  video.src = `${analysis.video_url}?v=${Date.now()}`;
  document.querySelector("#heatmap-image").src = `${analysis.heatmap_url}?v=${Date.now()}`;

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

function drawPoints(ctx, points, color, sourceSize, radius) {
  const [sourceWidth, sourceHeight] = sourceSize;
  ctx.fillStyle = color;
  for (const [x, y] of points) {
    const px = 22 + (x / sourceWidth) * (ctx.canvas.width - 44);
    const py = 22 + (y / sourceHeight) * (ctx.canvas.height - 44);
    ctx.globalAlpha = 0.68;
    ctx.beginPath();
    ctx.arc(px, py, radius, 0, Math.PI * 2);
    ctx.fill();
  }
  ctx.globalAlpha = 1;
}

function drawTacticalMap(analysis) {
  const canvas = document.querySelector("#tactical-canvas");
  const ctx = canvas.getContext("2d");
  drawField(ctx, canvas.width, canvas.height);
  drawPoints(ctx, analysis.positions.red_team || [], "#ed454f", analysis.source_size, 7);
  drawPoints(ctx, analysis.positions.blue_team || [], "#3a8dfa", analysis.source_size, 7);
  drawPoints(ctx, analysis.positions.ball || [], "#f4cf38", analysis.source_size, 5);
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
