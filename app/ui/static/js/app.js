const metricCards = document.getElementById("metricCards");
const datasetSummary = document.getElementById("datasetSummary");
const modelSummary = document.getElementById("modelSummary");
const temporalSummary = document.getElementById("temporalSummary");
const pipelineList = document.getElementById("pipelineList");
const stabilityList = document.getElementById("stabilityList");
const comparisonTableBody = document.querySelector("#comparisonTable tbody");
const healthBadge = document.getElementById("healthBadge");
const activeModelName = document.getElementById("activeModelName");
const databaseMode = document.getElementById("databaseMode");
const recentRecordsTableBody = document.querySelector("#recentRecordsTable tbody");
const recentTasksTableBody = document.querySelector("#recentTasksTable tbody");
const cigaretteAnalysisSummary = document.getElementById("cigaretteAnalysisSummary");
const cigaretteAnalysisExamples = document.getElementById("cigaretteAnalysisExamples");
const detectForm = document.getElementById("detectForm");
const imageInput = document.getElementById("imageInput");
const detectConf = document.getElementById("detectConf");
const detectIou = document.getElementById("detectIou");
const detectModelSelect = document.getElementById("detectModelSelect");
const detectStatus = document.getElementById("detectStatus");
const resultImage = document.getElementById("resultImage");
const detectionList = document.getElementById("detectionList");
const videoForm = document.getElementById("videoForm");
const videoInput = document.getElementById("videoInput");
const videoConf = document.getElementById("videoConf");
const videoIou = document.getElementById("videoIou");
const videoModelSelect = document.getElementById("videoModelSelect");
const videoStatus = document.getElementById("videoStatus");
const videoTaskTableBody = document.querySelector("#videoTaskTable tbody");
const refreshTasksBtn = document.getElementById("refreshTasksBtn");
const recordFilterForm = document.getElementById("recordFilterForm");
const recordClassFilter = document.getElementById("recordClassFilter");
const recordStatusFilter = document.getElementById("recordStatusFilter");
const recordModelFilter = document.getElementById("recordModelFilter");
const recordsTableBody = document.querySelector("#recordsTable tbody");
const recordDetailImage = document.getElementById("recordDetailImage");
const recordDetailText = document.getElementById("recordDetailText");
const modelCards = document.getElementById("modelCards");
const settingsForm = document.getElementById("settingsForm");
const settingsModelSelect = document.getElementById("settingsModelSelect");
const settingsConf = document.getElementById("settingsConf");
const settingsIou = document.getElementById("settingsIou");
const settingsImgsz = document.getElementById("settingsImgsz");
const settingsUploadMb = document.getElementById("settingsUploadMb");
const settingsStatus = document.getElementById("settingsStatus");
const navButtons = document.querySelectorAll(".nav-item");
const viewPanels = document.querySelectorAll(".view-panel");
const switchViewButtons = document.querySelectorAll("[data-switch-view]");

const demoConfig = window.__DEMO_CONFIG__ || {};
const state = { models: [], defaultModelId: null, settings: null, taskPollHandle: null };

function apiFetch(url, options = {}) {
  return fetch(url, options).then(async (response) => {
    const payload = await response.json().catch(() => ({}));
    if (!response.ok) throw new Error(payload.detail || payload.message || "请求失败");
    return payload;
  });
}

function switchView(viewId) {
  viewPanels.forEach((panel) => panel.classList.toggle("is-visible", panel.id === viewId));
  navButtons.forEach((button) => button.classList.toggle("is-active", button.dataset.viewTarget === viewId));
}

function formatDate(value) {
  return value ? new Date(value).toLocaleString("zh-CN", { hour12: false }) : "-";
}

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function renderMetricCards(items) {
  metricCards.innerHTML = "";
  items.forEach((item) => {
    const card = document.createElement("article");
    card.className = "metric-card";
    card.innerHTML = `<h3>${escapeHtml(item.label)}</h3><div class="metric-value">${escapeHtml(item.value)}</div>`;
    metricCards.appendChild(card);
  });
}

function renderSimpleList(container, items) {
  container.innerHTML = "";
  items.forEach((item) => {
    const li = document.createElement("li");
    li.textContent = item;
    container.appendChild(li);
  });
}

function renderTemporalSummary() {
  const runtime = demoConfig.recommended_runtime || {};
  const temporal = runtime.temporal || {};
  const classConf = runtime.class_conf || {};
  const search = demoConfig.temporal_search || {};
  temporalSummary.textContent = [
    `推荐参数组：${search.recommended_param_set || "-"}`,
    `默认 conf=${runtime.default_conf ?? "-"}, IoU=${runtime.default_iou ?? "-"}, imgsz=${runtime.imgsz ?? "-"}`,
    `类别阈值：cigarette=${classConf.cigarette ?? "-"}, smoking_person=${classConf.smoking_person ?? "-"}, smoke=${classConf.smoke ?? "-"}`,
    `时序参数：match_iou=${temporal.match_iou ?? "-"}, stable_hits=${temporal.stable_hits ?? "-"}, bridge_frames=${temporal.bridge_frames ?? "-"}, stale_frames=${temporal.track_stale_frames ?? "-"}`,
    `参数搜索：${search.video_count || 0} 个 HMDB51 smoke 视频，事件命中率 ${search.temporal_event_hit_rate || "-"}，稳定轨迹 ${search.stable_track_count || 0}`,
    `边界说明：${search.boundary || "正样本参数搜索不能单独证明误检率下降。"}`,
  ].join("\n");
}

function renderComparisonRows(experiments = {}) {
  comparisonTableBody.innerHTML = "";
  Object.entries(experiments).forEach(([name, metrics]) => {
    const row = document.createElement("tr");
    row.innerHTML = `
      <td>${escapeHtml(name)}${metrics.note ? `<br><small>${escapeHtml(metrics.note)}</small>` : ""}</td>
      <td>${escapeHtml(metrics.precision ?? "-")}</td>
      <td>${escapeHtml(metrics.recall ?? "-")}</td>
      <td>${escapeHtml(metrics.map50 ?? "-")}</td>
      <td>${escapeHtml(metrics.cigarette_recall ?? "-")}</td>
      <td>${escapeHtml(metrics.cigarette_map50 ?? "-")}</td>
      <td>${escapeHtml(metrics.weights_name ?? "-")}</td>`;
    comparisonTableBody.appendChild(row);
  });
  if (!comparisonTableBody.innerHTML) comparisonTableBody.innerHTML = `<tr><td colspan="7">暂无实验对比数据</td></tr>`;
}

function renderRecentRecords(items = []) {
  recentRecordsTableBody.innerHTML = "";
  if (!items.length) {
    recentRecordsTableBody.innerHTML = `<tr><td colspan="5">暂无记录</td></tr>`;
    return;
  }
  items.forEach((item) => {
    const row = document.createElement("tr");
    row.innerHTML = `<td>${item.id}</td><td>${escapeHtml(item.source_name)}</td><td>${escapeHtml(item.model_name || "-")}</td><td>${item.num_detections}</td><td>${formatDate(item.created_at)}</td>`;
    recentRecordsTableBody.appendChild(row);
  });
}

function taskStatusClass(status) {
  if (status === "completed") return "status-completed";
  if (status === "running") return "status-running";
  if (status === "queued") return "status-queued";
  return "status-failed";
}

function renderRecentTasks(items = []) {
  recentTasksTableBody.innerHTML = "";
  if (!items.length) {
    recentTasksTableBody.innerHTML = `<tr><td colspan="4">暂无视频任务</td></tr>`;
    return;
  }
  items.forEach((item) => {
    const row = document.createElement("tr");
    row.innerHTML = `<td>${item.task_uuid.slice(0, 8)}</td><td>${escapeHtml(item.source_name)}</td><td><span class="status-pill ${taskStatusClass(item.status)}">${escapeHtml(item.status)}</span></td><td>${Math.round((item.progress || 0) * 100)}%</td>`;
    recentTasksTableBody.appendChild(row);
  });
}

function renderCigaretteAnalysis(payload = {}) {
  if (!payload.dataset_summary) {
    cigaretteAnalysisSummary.textContent = "暂无 cigarette 分析报告。";
    cigaretteAnalysisExamples.textContent = "可运行 analyze_cigarette_detection.py 生成小目标漏检样本清单。";
    return;
  }
  const summary = payload.dataset_summary;
  const prediction = payload.prediction_analysis || {};
  cigaretteAnalysisSummary.textContent = [
    `报告文件：${payload.report_path || "-"}`,
    `split：${payload.split || "-"}`,
    `含 cigarette 图像：${summary.images_with_cigarette}/${summary.image_count}`,
    `cigarette 标注框数：${summary.cigarette_box_count}`,
    `面积占比均值：${summary.cigarette_area_ratio_mean}`,
    `面积占比中位数：${summary.cigarette_area_ratio_median}`,
    prediction.missed_cigarette_gt !== undefined ? `漏检 cigarette GT：${prediction.missed_cigarette_gt}` : "",
  ].filter(Boolean).join("\n");
  cigaretteAnalysisExamples.textContent = JSON.stringify({ missed_examples: (prediction.missed_examples || []).slice(0, 8), low_confidence_examples: (prediction.low_confidence_examples || []).slice(0, 8) }, null, 2);
}

function renderModelOptions(selectEl, includeAll = false) {
  if (!selectEl) return;
  const currentValue = selectEl.value;
  selectEl.innerHTML = includeAll ? `<option value="">全部</option>` : "";
  state.models.forEach((model) => {
    const option = document.createElement("option");
    option.value = String(model.id);
    option.textContent = `${model.name}${model.is_default ? "（默认）" : ""}`;
    selectEl.appendChild(option);
  });
  if (currentValue) selectEl.value = currentValue;
  if (!selectEl.value && state.defaultModelId) selectEl.value = String(state.defaultModelId);
}

function renderModels() {
  modelCards.innerHTML = "";
  state.models.forEach((model) => {
    const article = document.createElement("article");
    article.className = "model-card";
    article.innerHTML = `<h4>${escapeHtml(model.name)}</h4><p>${escapeHtml(model.note || "无备注")}</p><div class="model-meta"><span><strong>路径：</strong>${escapeHtml(model.weights_path)}</span><span><strong>设备：</strong>${escapeHtml(model.device)}</span><span><strong>状态：</strong>${model.is_available ? "available" : "missing"}</span></div><button class="primary-btn" data-model-id="${model.id}">${model.is_default ? "当前默认模型" : "设为默认模型"}</button>`;
    modelCards.appendChild(article);
  });
  modelCards.querySelectorAll("[data-model-id]").forEach((button) => {
    button.addEventListener("click", async () => {
      try {
        await apiFetch("/api/models/default", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ model_id: Number(button.dataset.modelId) }) });
        settingsStatus.textContent = "默认模型已更新。";
        await refreshModels();
        await loadDashboard();
      } catch (error) { settingsStatus.textContent = error.message; }
    });
  });
}

function renderSettings(settings) {
  state.settings = settings;
  settingsConf.value = settings.default_conf;
  settingsIou.value = settings.default_iou;
  settingsImgsz.value = settings.default_imgsz;
  settingsUploadMb.value = settings.max_upload_mb;
  renderModelOptions(settingsModelSelect);
  if (settings.default_model_id) settingsModelSelect.value = String(settings.default_model_id);
  detectConf.value = settings.default_conf;
  detectIou.value = settings.default_iou;
  videoConf.value = settings.default_conf;
  videoIou.value = settings.default_iou;
}

function renderHealth(payload) {
  const mode = payload.database_backend || "unknown";
  const healthy = payload.status === "ok";
  healthBadge.className = `health-chip status-pill ${healthy ? "status-ok" : "status-degraded"}`;
  healthBadge.textContent = healthy ? `系统正常 | DB: ${mode} | 模型已就绪` : `系统降级 | DB: ${mode} | ${payload.database_error || "检查配置"}`;
}

function renderDetections(items = []) {
  detectionList.innerHTML = "";
  if (!items.length) {
    detectionList.innerHTML = `<li>未检测到目标</li>`;
    return;
  }
  items.forEach((item) => {
    const li = document.createElement("li");
    li.textContent = `${item.class_name} | conf=${item.confidence.toFixed(3)} | box=[${item.xyxy.map((x) => x.toFixed(1)).join(", ")}]`;
    detectionList.appendChild(li);
  });
}

function formatVideoSummary(summary = {}) {
  if (!summary || summary.message) return summary?.message || "暂无报告";
  const classCounts = Object.entries(summary.per_class_counts || {}).map(([name, count]) => `${name}:${count}`).join(", ") || "none";
  return [`frames=${summary.processed_frames}/${summary.total_frames}`, `raw=${summary.raw_num_detections ?? 0}`, `smoothed=${summary.smoothed_num_detections ?? summary.num_detections ?? 0}`, `stable_tracks=${summary.stable_track_count ?? 0}`, `temporal_hit=${summary.temporal_event_hit ? "yes" : "no"}`, `classes=${classCounts}`].join(" | ");
}

function renderVideoTasks(items = []) {
  videoTaskTableBody.innerHTML = "";
  if (!items.length) {
    videoTaskTableBody.innerHTML = `<tr><td colspan="5">暂无视频任务</td></tr>`;
    return;
  }
  items.forEach((item) => {
    const outputLinks = item.output_video_url ? `<a href="/reports/video/${item.id}" target="_blank" rel="noreferrer">查看中文报告</a> · <a href="${item.output_video_url}" target="_blank" rel="noreferrer">播放视频</a> · <a href="/api/tasks/video/${item.id}" target="_blank" rel="noreferrer">JSON</a>` : "-";
    const reportText = item.status === "failed" ? `error=${item.error_message || "unknown"}` : formatVideoSummary(item.summary || {});
    const row = document.createElement("tr");
    row.innerHTML = `<td>${item.task_uuid.slice(0, 8)}</td><td>${escapeHtml(item.source_name)}</td><td><span class="status-pill ${taskStatusClass(item.status)}">${escapeHtml(item.status)}</span></td><td>${Math.round((item.progress || 0) * 100)}% (${item.processed_frames}/${item.total_frames || 0})</td><td>${outputLinks}<div class="task-report">${escapeHtml(reportText)}</div></td>`;
    videoTaskTableBody.appendChild(row);
  });
}

function renderRecordRows(items = []) {
  recordsTableBody.innerHTML = "";
  if (!items.length) {
    recordsTableBody.innerHTML = `<tr><td colspan="6">暂无检测记录</td></tr>`;
    return;
  }
  items.forEach((item) => {
    const row = document.createElement("tr");
    row.innerHTML = `<td>${item.id}</td><td>${escapeHtml(item.source_name)}</td><td>${escapeHtml(item.model_name || "-")}</td><td>${item.num_detections}</td><td>${formatDate(item.created_at)}</td><td><button class="table-btn" data-record-view="${item.id}">查看</button><button class="table-btn" data-record-delete="${item.id}">删除</button></td>`;
    recordsTableBody.appendChild(row);
  });
  recordsTableBody.querySelectorAll("[data-record-view]").forEach((button) => button.addEventListener("click", () => loadRecordDetail(Number(button.dataset.recordView))));
  recordsTableBody.querySelectorAll("[data-record-delete]").forEach((button) => button.addEventListener("click", () => deleteRecord(Number(button.dataset.recordDelete))));
}

async function loadRecordDetail(recordId) {
  try {
    const payload = await apiFetch(`/api/records/${recordId}`);
    recordDetailImage.src = payload.annotated_image_url || "";
    recordDetailText.textContent = JSON.stringify(payload, null, 2);
  } catch (error) { recordDetailText.textContent = error.message; }
}

async function deleteRecord(recordId) {
  if (!window.confirm(`确认删除本地检测记录 #${recordId}？该操作会从数据库删除记录。`)) return;
  try {
    await apiFetch(`/api/records/${recordId}`, { method: "DELETE" });
    recordDetailText.textContent = `记录 ${recordId} 已删除。`;
    recordDetailImage.src = "";
    await loadRecords();
    await loadDashboard();
  } catch (error) { recordDetailText.textContent = error.message; }
}

async function refreshModels() {
  const payload = await apiFetch("/api/models");
  state.models = payload.items || [];
  state.defaultModelId = payload.default_model_id;
  renderModels();
  renderModelOptions(detectModelSelect);
  renderModelOptions(videoModelSelect);
  renderModelOptions(recordModelFilter, true);
  renderModelOptions(settingsModelSelect);
}

async function loadSettings() { renderSettings(await apiFetch("/api/settings")); }
async function loadHealth() { renderHealth(await apiFetch("/api/health")); }

async function loadDashboard() {
  const payload = await apiFetch("/api/dashboard");
  const defaultModel = payload.models?.find((item) => item.is_default);
  activeModelName.textContent = defaultModel?.name || payload.model.weights_path;
  databaseMode.textContent = payload.storage.database_backend;
  datasetSummary.textContent = `${payload.dataset.name}: train ${payload.dataset.train_images}, val ${payload.dataset.val_images}, test ${payload.dataset.test_images} | classes: ${payload.dataset.classes.join(", ")}`;
  modelSummary.textContent = `默认模型：${defaultModel?.name || payload.model.weights_path} | imgsz=${payload.model.imgsz} | upload=${payload.model.max_upload_mb}MB`;
  renderMetricCards([{ label: "检测记录", value: payload.stats.total_records }, { label: "视频任务", value: payload.stats.total_tasks }, { label: "完成任务", value: payload.stats.completed_tasks }, { label: "可用模型", value: payload.stats.available_models }]);
  renderTemporalSummary();
  renderSimpleList(pipelineList, payload.pipeline || []);
  renderSimpleList(stabilityList, payload.stability || []);
  renderComparisonRows(payload.experiments || {});
  renderRecentRecords(payload.recent_records || []);
  renderRecentTasks(payload.recent_tasks || []);
  renderCigaretteAnalysis(payload.cigarette_analysis || {});
}

async function loadTasks() { renderVideoTasks((await apiFetch("/api/tasks/video?limit=20")).items || []); }

async function loadRecords() {
  const params = new URLSearchParams();
  if (recordClassFilter.value) params.set("class_name", recordClassFilter.value);
  if (recordStatusFilter.value) params.set("status", recordStatusFilter.value);
  if (recordModelFilter.value) params.set("model_id", recordModelFilter.value);
  params.set("limit", "30");
  renderRecordRows((await apiFetch(`/api/records?${params.toString()}`)).items || []);
}

detectForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  if (!imageInput.files.length) { detectStatus.textContent = "请先选择一张图片。"; return; }
  const formData = new FormData();
  formData.append("file", imageInput.files[0]);
  formData.append("conf", detectConf.value);
  formData.append("iou", detectIou.value);
  if (detectModelSelect.value) formData.append("model_id", detectModelSelect.value);
  detectStatus.textContent = "正在执行图片检测...";
  try {
    const payload = await apiFetch("/api/detect/image", { method: "POST", body: formData });
    resultImage.src = payload.annotated_image_url || `data:image/jpeg;base64,${payload.annotated_image_base64}`;
    renderDetections(payload.detections || []);
    detectStatus.textContent = `检测完成，记录 #${payload.record_id}，目标数 ${payload.num_detections}。`;
    await loadRecords();
    await loadDashboard();
  } catch (error) { detectStatus.textContent = error.message; }
});

videoForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  if (!videoInput.files.length) { videoStatus.textContent = "请先选择一个视频。"; return; }
  const formData = new FormData();
  formData.append("file", videoInput.files[0]);
  formData.append("conf", videoConf.value);
  formData.append("iou", videoIou.value);
  if (videoModelSelect.value) formData.append("model_id", videoModelSelect.value);
  videoStatus.textContent = "正在创建视频任务...";
  try {
    const payload = await apiFetch("/api/tasks/video", { method: "POST", body: formData });
    videoStatus.textContent = `视频任务已创建：${payload.task.task_uuid}`;
    await loadTasks();
    await loadDashboard();
  } catch (error) { videoStatus.textContent = error.message; }
});

recordFilterForm.addEventListener("submit", async (event) => { event.preventDefault(); await loadRecords(); });
refreshTasksBtn.addEventListener("click", async () => { await loadTasks(); });

settingsForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  try {
    await apiFetch("/api/settings", { method: "PUT", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ default_model_id: settingsModelSelect.value ? Number(settingsModelSelect.value) : null, default_conf: Number(settingsConf.value), default_iou: Number(settingsIou.value), default_imgsz: Number(settingsImgsz.value), max_upload_mb: Number(settingsUploadMb.value) }) });
    settingsStatus.textContent = "全局参数已更新。";
    await refreshModels();
    await loadSettings();
    await loadDashboard();
  } catch (error) { settingsStatus.textContent = error.message; }
});

navButtons.forEach((button) => button.addEventListener("click", () => switchView(button.dataset.viewTarget)));
switchViewButtons.forEach((button) => button.addEventListener("click", () => switchView(button.dataset.switchView)));

async function initializePage() {
  await loadHealth();
  await refreshModels();
  await loadSettings();
  await loadDashboard();
  await loadTasks();
  await loadRecords();
}

initializePage().catch((error) => { healthBadge.textContent = `初始化失败：${error.message}`; });
state.taskPollHandle = window.setInterval(() => { loadTasks().catch(() => {}); loadHealth().catch(() => {}); }, 6000);
