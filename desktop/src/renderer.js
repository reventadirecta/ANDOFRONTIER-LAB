const MODULES = [
  ["overview", "Case Overview"],
  ["video", "Video & Tracking"],
  ["roi", "Track-Based ROI"],
  ["motion", "Motion / Optical Flow"],
  ["spectral", "Spectral Analysis"],
  ["thermal", "Thermal / IR"],
  ["srv", "SRV / Visual Reconstruction"],
  ["controls", "Controls"],
  ["pca", "PCA"],
  ["autoencoder", "Autoencoder"],
  ["reports", "Reports"],
  ["publicSummary", "Public Summary"]
];

const VIDEO_EXTENSIONS = [".mp4", ".mov", ".avi", ".mkv", ".webm"];

const state = {
  config: null,
  batchId: null,
  cases: [],
  selectedCase: null,
  selectedCaseData: null,
  caseState: null,
  activeTab: "overview",
  fps: 30,
  pendingPreviewLoad: false,
  trackingRunStartedAt: null,
  trackingTimer: null,
  lastProgressPercent: null,
  pcaRunStatus: "idle",
  autoencoderRunStatus: "idle",
  drawing: false,
  start: null,
  current: null,
  box: null
};

const $ = (id) => document.getElementById(id);
const pretty = (value) => JSON.stringify(value ?? {}, null, 2);
const exists = (asset) => Boolean(asset?.exists && asset?.size > 0);

function isSupportedVideoPath(filePath) {
  return VIDEO_EXTENSIONS.some((extension) => filePath.toLowerCase().endsWith(extension));
}

async function refreshBatchSelect(preferredBatchId = "") {
  const batches = await window.forensicDesk.listBatches();
  $("batchSelect").innerHTML = batches.map((id) => `<option value="${id}">${id}</option>`).join("");
  state.batchId = preferredBatchId && batches.includes(preferredBatchId)
    ? preferredBatchId
    : (batches.includes(state.config.default_batch_id) ? state.config.default_batch_id : (batches[0] || ""));
  $("batchSelect").value = state.batchId;
  return batches;
}

function log(text) {
  const target = $("logs");
  if (!target) return;
  target.textContent += `${text}\n`;
  target.scrollTop = target.scrollHeight;
}

function reviewClass(status) {
  if (status === "tracking_human_validated") return "ok";
  if (status === "tracking_unvalidated") return "warn";
  return "bad";
}

function statusPill(status, label = status) {
  const cls = {
    ready: "ok",
    validated: "ok",
    available: "info",
    pending: "warn",
    blocked: "bad",
    failed: "bad",
    missing: "muted",
    placeholder: "muted"
  }[status] || "muted";
  return `<span class="pill ${cls}">${label}</span>`;
}

function fileLine(label, filePath) {
  return `<div class="fileLine"><span>${label}</span><code>${filePath || "not available"}</code></div>`;
}

function imageIf(asset, alt) {
  if (!exists(asset)) return `<div class="emptyState">Not available yet</div>`;
  return `<img class="moduleImage" src="${asset.url}" alt="${alt}" />`;
}

function videoIf(asset, alt) {
  if (!exists(asset)) return `<div class="emptyState">Not available yet</div>`;
  return `<video class="moduleVideo" src="${asset.url}" controls aria-label="${alt}"></video>`;
}

function moduleStatus(moduleId) {
  const cs = state.caseState;
  if (!cs) return { status: "missing", label: "not available", locked: true };
  const gates = cs.gates;
  if (moduleId === "overview" || moduleId === "video") return { status: "ready", label: "ready", locked: false };
  if (!gates.tracking_generated) return { status: "blocked", label: "tracking_required", locked: true };
  if (!gates.tracking_validated) return { status: "pending", label: "tracking_unvalidated", locked: true };
  if (moduleId === "roi") return gates.track_based_analysis_ready ? { status: "available", label: "available output", locked: false } : { status: "validated", label: "track validated", locked: false };
  if (!gates.track_based_analysis_ready) return { status: "pending", label: "requires track-based ROI", locked: true };
  if (moduleId === "motion") return gates.motion_analysis_ready ? { status: "available", label: "motion ready", locked: false } : { status: "validated", label: "ready to run", locked: false };
  if (moduleId === "spectral") return gates.spectral_analysis_ready ? { status: "available", label: "spectral ready", locked: false } : { status: "validated", label: "ready to run", locked: false };
  if (moduleId === "thermal") return gates.thermal_analysis_ready ? { status: "available", label: "thermal ready", locked: false } : { status: "validated", label: "ready to run", locked: false };
  if (moduleId === "srv") return gates.srv_analysis_ready ? { status: "available", label: "srv ready", locked: false } : { status: "validated", label: "ready to run", locked: false };
  if (moduleId === "controls") return gates.controls_analysis_ready ? { status: "available", label: "controls ready", locked: false } : { status: "validated", label: "ready to run", locked: false };
  if (moduleId === "pca") return gates.pca_analysis_ready ? { status: "available", label: "PCA ready", locked: false } : { status: "validated", label: "ready to run", locked: false };
  if (moduleId === "autoencoder") return gates.autoencoder_analysis_ready ? { status: "available", label: "autoencoder ready", locked: false } : { status: "validated", label: "ready to run", locked: false };
  if (["reports"].includes(moduleId)) return { status: "available", label: "available output", locked: false };
  if (moduleId === "publicSummary") return gates.reddit_template_ready ? { status: "available", label: "reddit template ready", locked: false } : { status: gates.unified_report_ready ? "validated" : "pending", label: gates.unified_report_ready ? "ready to generate" : "requires unified report", locked: false };
  return { status: "placeholder", label: "not available yet", locked: false };
}

function moduleHeader(title, moduleId, requirements, outputs) {
  const status = moduleStatus(moduleId);
  return `
    <div class="moduleHeader">
      <div>
        <h2>${title}</h2>
        <div class="moduleMeta">${statusPill(status.status, status.label)}</div>
      </div>
      <div class="moduleReq">
        <div><strong>Requirements:</strong> ${requirements}</div>
        <div><strong>Expected outputs:</strong> ${outputs}</div>
      </div>
    </div>
  `;
}

async function init() {
  state.config = await window.forensicDesk.getConfig();
  if (state.config.config_missing) {
    $("labStatus").textContent = `Missing config: copy ${state.config.config_example_path} to ${state.config.config_required_path}`;
    $("labStatus").classList.add("errorText");
    log("Missing config/lab.local.json. Copy the example and adjust lab_root/python_exe.");
    return;
  }
  $("labStatus").textContent = `Lab: ${state.config.lab_root} | python: ${state.config.python_exists ? "ok" : "missing"}`;
  if (!state.config.lab_exists || !state.config.python_exists) {
    $("labStatus").classList.add("errorText");
    log(`Config problem: lab_exists=${state.config.lab_exists} python_exists=${state.config.python_exists}`);
    return;
  }
  await refreshBatchSelect();
  await loadBatch();
  if (!state.batchId) {
    log("Runtime initialized. No batch manifest found yet.");
  }
  window.forensicDesk.onTrackingLog(({ text }) => log(text.trimEnd()));
  window.forensicDesk.onTrackingProgress((payload) => handleTrackingProgress(payload));
}

async function loadBatch(preferredCaseId = "") {
  state.batchId = $("batchSelect").value || "";
  const batch = await window.forensicDesk.getBatch(state.batchId);
  state.cases = batch.cases || [];
  $("caseCount").textContent = `${state.cases.length} cases`;
  if (state.cases.length === 0) {
    state.selectedCase = null;
    state.caseState = null;
    state.box = null;
    $("casesTable").innerHTML = `
      <tr>
        <td colspan="4" class="muted">No cases found. Create or import a case to begin.</td>
      </tr>
    `;
    $("caseTitle").textContent = "No case selected";
    $("caseSubtitle").textContent = "Track first. Analyze after.";
    renderSidebar();
    renderTabs();
    renderActiveTab();
    return;
  }
  $("casesTable").innerHTML = state.cases.map((item) => `
    <tr data-case="${item.case_id}">
      <td>${item.case_id}</td>
      <td>${item.priority || ""}</td>
      <td>${item.tracking_status}</td>
      <td><span class="pill ${reviewClass(item.review_status)}">${item.review_status}</span></td>
    </tr>
  `).join("");
  const initial = state.cases.find((item) => item.case_id === preferredCaseId)
    || state.cases.find((item) => item.case_id === "dod_111688816")
    || state.cases.find((item) => item.case_id === "dod_111688723")
    || state.cases[0];
  if (initial) await openCase(initial.case_id);
}

async function openCase(caseId) {
  state.selectedCase = caseId;
  state.selectedCaseData = await window.forensicDesk.getCase(state.batchId, caseId);
  state.caseState = await window.forensicDesk.getCaseState(state.batchId, caseId);
  const data = state.selectedCaseData;
  state.fps = Number(data.case.fps || 30);
  state.box = data.request?.object_prompt?.box || null;
  state.start = null;
  state.current = null;
  $("caseTitle").textContent = caseId;
  $("caseSubtitle").textContent = data.case.original_filename || "";
  renderTabs();
  renderActiveTab();
  renderSidebar();
}

function renderTabs() {
  $("moduleTabs").innerHTML = MODULES.map(([id, label]) => {
    const status = moduleStatus(id);
    return `<button class="tab ${state.activeTab === id ? "active" : ""} ${status.locked ? "locked" : ""}" data-tab="${id}">
      <span>${label}</span>${statusPill(status.status, "")}
    </button>`;
  }).join("");
}

function renderActiveTab() {
  if (!state.caseState) {
    $("moduleContent").innerHTML = `
      <section class="firstRunPanel" id="videoDropZone">
        <div>
          <p class="eyebrow">Track first. Analyze after.</p>
          <h2>Import a video to begin</h2>
          <p>Select a local video file. AndoFrontier Lab will create a local case in the portable runtime folder.</p>
        </div>
        <div class="firstRunActions">
          <button id="importVideoEmpty" class="primary">Import Video</button>
          <button id="openRuntimeEmpty">Open Runtime Folder</button>
        </div>
        <p class="muted">You can also drag and drop an MP4, MOV, AVI, MKV, or WEBM file here.</p>
      </section>
    `;
    return;
  }
  const renderers = {
    overview: renderOverview,
    video: renderVideoTracking,
    roi: renderTrackBasedRoi,
    motion: renderMotion,
    spectral: renderSpectral,
    thermal: renderThermal,
    srv: renderSrv,
    pca: renderPca,
    autoencoder: renderAutoencoder,
    controls: renderControls,
    reports: renderReports,
    publicSummary: renderPublicSummary
  };
  $("moduleContent").innerHTML = (renderers[state.activeTab] || renderOverview)();
  afterTabRender();
}

function renderSidebar() {
  const cs = state.caseState;
  $("caseMetadata").textContent = pretty({
    case_id: cs?.case?.case_id,
    original_filename: cs?.case?.original_filename,
    duration_seconds: cs?.case?.duration_seconds ?? "unknown",
    fps: cs?.case?.fps ?? "unknown",
    resolution: cs?.case?.resolution ?? "unknown",
    codec: cs?.case?.codec ?? "unknown",
    priority: cs?.case?.priority ?? "unknown",
    review_status: cs?.review_status ?? "unknown",
    track_based_analysis_ready: cs?.gates?.track_based_analysis_ready ?? false
  });
  $("requestPreview").textContent = pretty(cs?.request || { status: "missing" });
  $("validationPreview").textContent = pretty(cs?.validation || { status: "not created" });
  if (!cs) {
    $("annotationStatus").textContent = "Import a video, then draw a box over the real object.";
  } else {
    $("annotationStatus").textContent = state.box ? `Loaded saved box x=${state.box.x} y=${state.box.y} w=${state.box.w} h=${state.box.h}` : "Draw a box over the real object.";
  }
}

function renderOverview() {
  const cs = state.caseState;
  const c = cs.case;
  return `
    ${moduleHeader("Case Overview", "overview", "Batch metadata and existing case files", "Status summary and key paths")}
    <div class="cards">
      ${card("Identity", `
        ${kv("case_id", c.case_id)}
        ${kv("file", c.original_filename || "unknown")}
        ${kv("priority", c.priority || c.quick_priority || "unknown")}
        ${kv("source quality", c.source_quality || c.source_type || "unknown")}
      `)}
      ${card("Video", `
        ${kv("duration", c.duration_seconds ?? "unknown")}
        ${kv("resolution", c.resolution ? `${c.resolution.width}x${c.resolution.height}` : "unknown")}
        ${kv("fps", c.fps ?? "unknown")}
        ${kv("codec", c.codec || "unknown")}
      `)}
      ${card("Workflow Gates", `
        ${kv("tracking", cs.gates.tracking_generated ? "generated" : "required")}
        ${kv("validation", cs.gates.tracking_validated ? "human validated" : "pending")}
        ${kv("track-based analysis", cs.gates.track_based_analysis_ready ? "ready" : "not available")}
        ${kv("review", cs.review_status)}
      `)}
    </div>
    <div class="moduleGrid">
      ${fileLine("video", cs.paths.video)}
      ${fileLine("track.json", cs.paths.track_json)}
      ${fileLine("track-based report", cs.paths.track_based_report)}
      ${fileLine("case status", cs.paths.case_status)}
    </div>
  `;
}

function renderVideoTracking() {
  const data = state.selectedCaseData;
  const cs = state.caseState;
  const request = data.request || {};
  const quality = cs.metrics.tracking_quality || {};
  const savedStart = request.first_object_frame ?? "not set";
  const savedTime = request.first_object_time_seconds != null ? `${Number(request.first_object_time_seconds).toFixed(3)}s` : "not set";
  const validateDisabled = quality.recommendation === "tracking_failed" ? "disabled" : "";
  return `
  ${moduleHeader("Video & Tracking", "video", "Manual object box and optional tracking command", "track.json, overlay MP4, validation JSON")}
    <p class="helpText">Move to the first frame where the object is visible, draw the box, then save it.</p>
    <div class="videoTools">
      <button id="prevFrame">Prev</button>
      <button id="nextFrame">Next</button>
      <button id="back100">-100</button>
      <button id="back10">-10</button>
      <button id="forward10">+10</button>
      <button id="forward100">+100</button>
      <label class="frameJumpLabel" for="goToFrameInput">Go to frame</label>
      <input id="goToFrameInput" class="frameJumpInput" type="number" min="0" step="1" />
      <button id="goToFrameButton">Go</button>
      <button id="setStartFrame">Set current frame as start</button>
      <button id="saveBox" class="primary">Save box</button>
      <span id="frameReadout">frame --</span>
    </div>
    <p id="startFrameStatus" class="statusLine">Tracking start: frame ${savedStart} | time ${savedTime}</p>
    <div class="videoStage" id="videoStage">
      <video id="caseVideo" controls src="${data.urls.video || ""}"></video>
      <canvas id="annotationCanvas"></canvas>
    </div>
    <div class="assetGrid">
      <figure><figcaption>Temporal contact sheet</figcaption>${imageIf(cs.assets.temporal_contact_sheet.exists ? cs.assets.temporal_contact_sheet : cs.assets.quick_panel, "temporal contact sheet")}</figure>
      <figure><figcaption>Tracking overlay preview</figcaption><video id="overlayVideo" controls src="${cs.assets.overlay.url || ""}"></video></figure>
    </div>
    <div class="buttonRow">
      <button id="runTracking" class="primary">Run tracking</button>
      <button id="reloadPreview">Reload preview</button>
      <button id="openTrackingFolder">Open tracking folder</button>
      <button id="openTrackJson">Open track.json</button>
      <button id="validTrack" ${validateDisabled}>Track correcto</button>
      <button id="invalidTrack">Track incorrecto</button>
      <button id="repromptTrack">Reanotar</button>
    </div>
    <p id="trackingStatus">Idle.</p>
    <div class="cards trackingQualityCards">
      ${card("Tracking Quality", `
        ${kv("recommendation", quality.recommendation || "not available")}
        ${kv("mean confidence", quality.mean_confidence ?? "unknown")}
        ${kv("auto recovered", quality.auto_recovered_frames ?? 0)}
        ${kv("low confidence", quality.low_confidence_frames ?? 0)}
        ${kv("predicted only", quality.predicted_only_frames ?? 0)}
        ${kv("lost", quality.lost_frames ?? 0)}
        ${kv("HUD rejected", quality.hud_rejected_candidates ?? 0)}
        ${kv("reticle rejected", quality.reticle_rejected_candidates ?? 0)}
        ${kv("drift to HUD", quality.drift_to_hud_detected ? "yes" : "no")}
      `)}
    </div>
    <div class="trackingProgress">
      <div id="trackingStage">Idle.</div>
      <progress id="trackingProgressBar" value="0" max="100"></progress>
      <div id="trackingProgressText">Tracking can take several minutes for long videos.</div>
    </div>
  `;
}

function renderTrackBasedRoi() {
  const cs = state.caseState;
  const m = cs.metrics.track_based || {};
  const disabled = cs.gates.tracking_validated ? "" : "disabled";
  return `
    ${moduleHeader("Track-Based ROI", "roi", "Human-validated track", "dynamic_rois.csv/json, crops, contact/trajectory panels")}
    <div class="cards">
      ${card("Tracking", `${kv("track.json", cs.gates.tracking_generated ? "exists" : "missing")}${kv("human validation", cs.gates.tracking_validated ? "validated" : "pending")}${kv("track-based analysis", cs.gates.track_based_analysis_ready ? "exists" : "not available yet")}`)}
      ${card("Metrics", `${kv("frames tracked", m.total_track_frames ?? cs.metrics.track_summary?.total_frames ?? "unknown")}${kv("valid frames", m.valid_analysis_frames ?? "unknown")}${kv("lost frames", m.lost_frames ?? "unknown")}${kv("crops", cs.metrics.crops ?? 0)}${kv("mean velocity", m.trajectory?.mean_velocity_px_frame ?? "unknown")}${kv("max velocity", m.trajectory?.max_velocity_px_frame ?? "unknown")}`)}
    </div>
    <div class="assetGrid">${imageFigure("Contact sheet", cs.assets.track_based_contact_sheet)}${imageFigure("Trajectory", cs.assets.track_based_trajectory_panel)}</div>
    <div class="buttonRow">
      <button id="rebuildFromTrack" class="primary" ${disabled}>Rebuild from validated track</button>
      <button data-open="${cs.paths.track_based_folder}">Open track-based folder</button>
      <button data-open="${cs.paths.dynamic_rois_csv}">Open dynamic_rois.csv</button>
      <button data-open="${cs.paths.track_based_report}">Open report</button>
      <button data-open="${cs.paths.track_based_manifest}">Open manifest</button>
    </div>
  `;
}

function renderMotion() {
  const cs = state.caseState;
  const m = cs.metrics.motion || {};
  const runDisabled = cs.gates.tracking_validated && cs.gates.dynamic_rois_ready ? "" : "disabled";
  return `
    ${moduleHeader("Motion / Optical Flow", "motion", "Human-validated track + dynamic_rois.csv", "metrics JSON, timeseries CSV, trajectory/velocity/flow/stability panels")}
    ${!cs.gates.tracking_validated ? `<div class="emptyState">Blocked: requires human-validated tracking.</div>` : ""}
    ${!cs.gates.dynamic_rois_ready ? `<div class="emptyState">Blocked: requires dynamic_rois.csv from track-based ROI.</div>` : ""}
    <div class="cards">
      ${card("Module State", `${kv("track validated", cs.gates.tracking_validated ? "yes" : "no")}${kv("dynamic ROIs", cs.gates.dynamic_rois_ready ? "yes" : "missing")}${kv("motion analysis", cs.gates.motion_analysis_ready ? "ready" : "not run")}`)}
      ${card("Velocity", `${kv("mean px/frame", fmt(m.mean_velocity_px_frame))}${kv("median px/frame", fmt(m.median_velocity_px_frame))}${kv("max px/frame", fmt(m.max_velocity_px_frame))}${kv("mean acceleration", fmt(m.mean_acceleration_px_frame2))}`)}
      ${card("ROI Motion", `${kv("mean frame diff", fmt(m.mean_frame_difference_inside_roi))}${kv("mean optical flow", fmt(m.mean_optical_flow_magnitude_inside_roi))}${kv("max optical flow", fmt(m.max_optical_flow_magnitude_inside_roi))}`)}
      ${card("Stability", `${kv("valid frames", m.valid_tracked_frames ?? "unknown")}${kv("lost frames", m.lost_frames ?? "unknown")}${kv("continuity", fmt(m.motion_continuity_score))}${kv("jitter", fmt(m.jitter_score))}${kv("stability", fmt(m.track_stability_score))}`)}
    </div>
    <div class="assetGrid">
      ${imageFigure("Trajectory", cs.assets.motion_trajectory_panel)}
      ${imageFigure("Velocity / acceleration", cs.assets.motion_velocity_panel)}
      ${imageFigure("Optical flow inside ROI", cs.assets.motion_optical_flow_panel)}
      ${imageFigure("Stability", cs.assets.motion_stability_panel)}
    </div>
    <div class="buttonRow">
      <button id="runMotionAnalysis" class="primary" ${runDisabled}>Run motion analysis</button>
      <button data-open="${cs.paths.motion_folder}">Open motion folder</button>
      <button data-open="${cs.paths.motion_metrics}">Open motion_metrics.json</button>
      <button data-open="${cs.paths.motion_timeseries}">Open motion_timeseries.csv</button>
      <button data-open="${cs.paths.motion_report}">Open motion report</button>
    </div>
  `;
}

function renderSpectral() {
  const cs = state.caseState;
  const m = cs.metrics.spectral || {};
  const runDisabled = cs.gates.tracking_validated && cs.gates.dynamic_rois_ready ? "" : "disabled";
  return `
    ${moduleHeader("Spectral Analysis", "spectral", "Human-validated track + dynamic_rois.csv", "spectral metrics, timeseries, luminance/color/FFT/spatial panels")}
    ${!cs.gates.tracking_validated ? `<div class="emptyState">Blocked: requires human-validated tracking.</div>` : ""}
    ${!cs.gates.dynamic_rois_ready ? `<div class="emptyState">Blocked: requires dynamic_rois.csv from track-based ROI.</div>` : ""}
    <div class="cards">
      ${card("Module State", `${kv("track validated", cs.gates.tracking_validated ? "yes" : "no")}${kv("dynamic ROIs", cs.gates.dynamic_rois_ready ? "yes" : "missing")}${kv("spectral analysis", cs.gates.spectral_analysis_ready ? "ready" : "not run")}`)}
      ${card("Luminance", `${kv("mean", fmt(m.mean_luminance))}${kv("std", fmt(m.luminance_std))}${kv("min", fmt(m.luminance_min))}${kv("max", fmt(m.luminance_max))}${kv("flicker index", fmt(m.luminance_flicker_index))}`)}
      ${card("Color", `${kv("R mean", fmt(m.mean_red_channel))}${kv("G mean", fmt(m.mean_green_channel))}${kv("B mean", fmt(m.mean_blue_channel))}${kv("Hue mean", fmt(m.mean_hue))}${kv("Sat mean", fmt(m.mean_saturation))}${kv("Value mean", fmt(m.mean_value))}`)}
      ${card("Temporal / Spatial", `${kv("dominant Hz", fmt(m.dominant_temporal_frequency_hz))}${kv("peak strength", fmt(m.temporal_frequency_peak_strength))}${kv("entropy", fmt(m.spectral_entropy_temporal))}${kv("spatial energy", fmt(m.mean_spatial_frequency_energy))}${kv("high freq ratio", fmt(m.high_frequency_energy_ratio))}${kv("noise proxy", fmt(m.compression_noise_proxy))}`)}
    </div>
    <div class="assetGrid">
      ${imageFigure("Luminance", cs.assets.spectral_luminance_panel)}
      ${imageFigure("Color / HSV", cs.assets.spectral_color_panel)}
      ${imageFigure("Temporal FFT", cs.assets.spectral_fft_panel)}
      ${imageFigure("Spatial frequency", cs.assets.spectral_spatial_frequency_panel)}
      ${imageFigure("Tracked crops contact sheet", cs.assets.spectral_contact_sheet)}
    </div>
    <div class="buttonRow">
      <button id="runSpectralAnalysis" class="primary" ${runDisabled}>Run spectral analysis</button>
      <button data-open="${cs.paths.spectral_folder}">Open spectral folder</button>
      <button data-open="${cs.paths.spectral_metrics}">Open spectral_metrics.json</button>
      <button data-open="${cs.paths.spectral_timeseries}">Open spectral_timeseries.csv</button>
      <button data-open="${cs.paths.spectral_report}">Open spectral report</button>
    </div>
  `;
}

function renderThermal() {
  const cs = state.caseState;
  const m = cs.metrics.thermal || {};
  const runDisabled = cs.gates.tracking_validated && cs.gates.dynamic_rois_ready ? "" : "disabled";
  return `
    ${moduleHeader("Thermal / IR", "thermal", "Human-validated track + dynamic_rois.csv", "relative IR intensity metrics, panels, and ROI sequence")}
    <div class="emptyState">FLIR/IR relative intensity analysis - no radiometric temperature units available</div>
    ${!cs.gates.tracking_validated ? `<div class="emptyState">Blocked: requires human-validated tracking.</div>` : ""}
    ${!cs.gates.dynamic_rois_ready ? `<div class="emptyState">Blocked: requires dynamic_rois.csv from track-based ROI.</div>` : ""}
    <div class="cards">
      ${card("IR Mode", `${kv("source_ir_mode", m.source_ir_mode ?? cs.case_status?.source_ir_mode ?? "unknown")}${kv("calibration available", m.calibration_available === false ? "false" : "unknown")}${kv("temperature units", m.temperature_units_available === false ? "false" : "unknown")}${kv("thermal analysis", cs.gates.thermal_analysis_ready ? "ready" : "not run")}`)}
      ${card("Intensity", `${kv("mean ROI", fmt(m.mean_roi_intensity))}${kv("std ROI", fmt(m.std_roi_intensity))}${kv("min ROI", fmt(m.min_roi_intensity))}${kv("max ROI", fmt(m.max_roi_intensity))}${kv("background mean", fmt(m.mean_background_intensity))}`)}
      ${card("Relative Contrast", `${kv("delta mean", fmt(m.roi_background_delta_mean))}${kv("delta std", fmt(m.roi_background_delta_std))}${kv("hot ratio", fmt(m.hot_pixel_ratio))}${kv("cold ratio", fmt(m.cold_pixel_ratio))}${kv("contrast index", fmt(m.thermal_contrast_index))}${kv("stability", fmt(m.intensity_stability_score))}`)}
    </div>
    <div class="assetGrid">
      ${imageFigure("IR intensity", cs.assets.thermal_intensity_panel)}
      ${imageFigure("IR contrast", cs.assets.thermal_contrast_panel)}
      ${imageFigure("ROI examples", cs.assets.thermal_roi_examples_panel)}
      ${imageFigure("ROI-background delta", cs.assets.thermal_delta_panel)}
      ${imageFigure("Thermal contact sheet", cs.assets.thermal_contact_sheet)}
      <figure><figcaption>Thermal ROI sequence</figcaption>${videoIf(cs.assets.thermal_roi_sequence, "Thermal ROI sequence")}</figure>
    </div>
    <div class="buttonRow">
      <button id="runThermalAnalysis" class="primary" ${runDisabled}>Run Thermal / IR analysis</button>
      <button data-open="${cs.paths.thermal_folder}">Open thermal folder</button>
      <button data-open="${cs.paths.thermal_metrics}">Open thermal_metrics.json</button>
      <button data-open="${cs.paths.thermal_timeseries}">Open thermal_timeseries.csv</button>
      <button data-open="${cs.paths.thermal_report}">Open thermal report</button>
    </div>
  `;
}

function renderSrv() {
  const cs = state.caseState;
  const m = cs.metrics.srv || {};
  const core = cs.metrics.srv_core || {};
  const size = Array.isArray(m.normalized_crop_size) ? m.normalized_crop_size.join("x") : "unknown";
  const coreSize = Array.isArray(core.normalized_core_crop_size) ? core.normalized_core_crop_size.join("x") : "unknown";
  const runDisabled = cs.gates.tracking_validated && cs.gates.dynamic_rois_ready ? "" : "disabled";
  return `
    ${moduleHeader("SRV / Visual Reconstruction", "srv", "Human-validated track + dynamic_rois.csv", "stabilized crops, safe enhancement, temporal stacks, quality panels")}
    ${!cs.gates.tracking_validated ? `<div class="emptyState">Blocked: requires human-validated tracking.</div>` : ""}
    ${!cs.gates.dynamic_rois_ready ? `<div class="emptyState">Blocked: requires dynamic_rois.csv from track-based ROI.</div>` : ""}
    <h3>Tracked bbox context</h3>
    <div class="cards">
      ${card("Module State", `${kv("track validated", cs.gates.tracking_validated ? "yes" : "no")}${kv("dynamic ROIs", cs.gates.dynamic_rois_ready ? "yes" : "missing")}${kv("SRV analysis", cs.gates.srv_analysis_ready ? "ready" : "not run")}${kv("generative model", "not used")}`)}
      ${card("Crops", `${kv("valid frames", m.valid_tracked_frames ?? "unknown")}${kv("crop count", m.crop_count ?? "unknown")}${kv("normalized size", size)}${kv("super-resolution used", m.super_resolution_used === false ? "false" : "unknown")}`)}
      ${card("Quality", `${kv("mean sharpness", fmt(m.mean_crop_sharpness))}${kv("median sharpness", fmt(m.median_crop_sharpness))}${kv("max sharpness", fmt(m.max_crop_sharpness))}${kv("mean contrast", fmt(m.mean_contrast))}${kv("mean luminance", fmt(m.mean_luminance))}`)}
    </div>
    <div class="assetGrid">
      ${imageFigure("Raw tracked crops", cs.assets.srv_contact_sheet_raw)}
      ${imageFigure("Enhanced crops", cs.assets.srv_contact_sheet_enhanced)}
      ${imageFigure("Stabilized sequence", cs.assets.srv_stabilized_sequence_panel)}
      ${imageFigure("Comparison", cs.assets.srv_comparison_panel)}
      ${imageFigure("Quality", cs.assets.srv_quality_panel)}
      ${imageFigure("Average stack", cs.assets.srv_stack_average)}
      ${imageFigure("Median stack", cs.assets.srv_stack_median)}
      ${imageFigure("Best sharpness", cs.assets.srv_stack_best_sharpness)}
      <figure><figcaption>Stabilized crop video</figcaption>${videoIf(cs.assets.srv_stabilized_video, "SRV stabilized crop sequence")}</figure>
    </div>
    <div class="buttonRow">
      <button id="runSrvAnalysis" class="primary" ${runDisabled}>Run SRV analysis</button>
      <button data-open="${cs.paths.srv_folder}">Open SRV folder</button>
      <button data-open="${cs.paths.srv_metrics}">Open srv_metrics.json</button>
      <button data-open="${cs.paths.srv_timeseries}">Open srv_timeseries.csv</button>
      <button data-open="${cs.paths.srv_report}">Open SRV report</button>
    </div>
    <h3>Object-core reconstruction</h3>
    <div class="cards">
      ${card("Core State", `${kv("object-core SRV", cs.gates.srv_object_core_ready ? "ready" : "not run")}${kv("valid core frames", core.valid_core_frames ?? "unknown")}${kv("low confidence", core.low_confidence_frames ?? "unknown")}${kv("core not found", core.core_not_found_frames ?? "unknown")}${kv("normalized core", coreSize)}`)}
      ${card("Core Quality", `${kv("mean core size", Array.isArray(core.mean_core_crop_size) ? core.mean_core_crop_size.map((v) => Number(v).toFixed(1)).join("x") : "unknown")}${kv("mean sharpness", fmt(core.mean_sharpness))}${kv("mean contrast", fmt(core.mean_contrast))}${kv("mean luminance", fmt(core.mean_luminance))}${kv("artifact score", fmt(core.artifact_contamination_score))}`)}
    </div>
    <div class="assetGrid">
      ${imageFigure("Core raw crops", cs.assets.srv_core_contact_sheet_raw)}
      ${imageFigure("Core enhanced crops", cs.assets.srv_core_contact_sheet_enhanced)}
      ${imageFigure("Core stabilized sequence", cs.assets.srv_core_stabilized_sequence_panel)}
      ${imageFigure("Core comparison", cs.assets.srv_core_comparison_panel)}
      ${imageFigure("Core quality", cs.assets.srv_core_quality_panel)}
      ${imageFigure("Core average stack", cs.assets.srv_core_stack_average)}
      ${imageFigure("Core median stack", cs.assets.srv_core_stack_median)}
      ${imageFigure("Core best sharpness", cs.assets.srv_core_stack_best_sharpness)}
      <figure><figcaption>Core stabilized crop video</figcaption>${videoIf(cs.assets.srv_core_stabilized_video, "SRV core stabilized crop sequence")}</figure>
    </div>
    <div class="buttonRow">
      <button id="runSrvCoreAnalysis" class="primary" ${runDisabled}>Run object-core SRV</button>
      <button data-open="${cs.paths.srv_core_folder}">Open object-core folder</button>
      <button data-open="${cs.paths.srv_core_rois_csv}">Open object_core_rois.csv</button>
      <button data-open="${cs.paths.srv_core_metrics}">Open srv_core_metrics.json</button>
      <button data-open="${cs.paths.srv_report}">Open SRV report</button>
    </div>
    <div class="emptyState">BBox-level SRV is context. Object-core SRV is recommended for visual reconstruction when HUD/reticle contamination is present. Visual reconstruction is interpretive; no generative model was used.</div>
  `;
}

function renderPca() {
  const cs = state.caseState;
  const p = cs.metrics.pca || {};
  const counts = p.samples_per_class || {};
  const prerequisitesReady = cs.gates.tracking_validated && cs.gates.dynamic_rois_ready && cs.gates.clean_controls_v02_ready;
  const running = state.pcaRunStatus === "running";
  const failed = state.pcaRunStatus === "failed";
  const runDisabled = prerequisitesReady && !running ? "" : "disabled";
  const runLabel = running ? "Running PCA..." : (cs.gates.pca_analysis_ready ? "PCA ready" : (failed ? "PCA failed" : "Run PCA analysis"));
  const pcaOpenDisabled = cs.gates.pca_analysis_ready ? "" : "disabled title=\"Run PCA analysis first.\"";
  const pcaOpenAttr = (targetPath) => cs.gates.pca_analysis_ready ? `data-open="${targetPath}"` : "";
  return `
    ${moduleHeader("PCA", "pca", "Human-validated track + dynamic_rois.csv + Controls v0.2 clean masked", "PCA baseline metrics, samples CSV, scatter, distances and report")}
    ${!cs.gates.tracking_validated ? `<div class="emptyState">Blocked: requires human-validated tracking.</div>` : ""}
    ${!cs.gates.dynamic_rois_ready ? `<div class="emptyState">Blocked: requires dynamic_rois.csv from track-based ROI.</div>` : ""}
    ${!cs.gates.clean_controls_v02_ready ? `<div class="emptyState">Blocked: requires Controls v0.2 clean masked baseline.</div>` : ""}
    <div class="emptyState">PCA is dimensionality reduction; separation from controls is not an origin claim.</div>
    <div class="cards">
      ${card("Module State", `${kv("track validated", cs.gates.tracking_validated ? "yes" : "no")}${kv("dynamic ROIs", cs.gates.dynamic_rois_ready ? "yes" : "missing")}${kv("clean controls v0.2", cs.gates.clean_controls_v02_ready ? "yes" : "missing")}${kv("control validity", fmt(cs.metrics.controls?.control_validity_score))}${kv("PCA analysis", cs.gates.pca_analysis_ready ? "ready" : "not run")}`)}
      ${card("Samples", `${kv("object", counts.object ?? 0)}${kv("near background", counts.near_background ?? 0)}${kv("far background", counts.far_background ?? 0)}${kv("compression noise", counts.compression_noise ?? 0)}${kv("random background", counts.random_background ?? 0)}${kv("HUD artifact", counts.hud_artifact ?? 0)}`)}
      ${card("Explained Variance", `${kv("PC1", fmt(p.pca_pc1_explained_variance))}${kv("PC2", fmt(p.pca_pc2_explained_variance))}${kv("k=5", fmt(p.pca_k5_explained_variance))}${kv("k=10", fmt(p.pca_k10_explained_variance))}`)}
      ${card("Class Separation", `${kv("object vs background", fmt(p.object_vs_background_distance))}${kv("object vs compression", fmt(p.object_vs_compression_distance))}${kv("object vs HUD", fmt(p.object_vs_hud_distance))}${kv("separation score", fmt(p.class_separation_score))}${kv("silhouette", fmt(p.silhouette_score))}${kv("nearest control similarity", fmt(p.nearest_control_similarity_score))}${kv("public-safe score", fmt(p.pca_public_safe_score))}`)}
    </div>
    <div class="assetGrid">
      ${imageFigure("PCA scatter", cs.assets.pca_scatter_panel)}
      ${imageFigure("Explained variance", cs.assets.pca_explained_variance_panel)}
      ${imageFigure("Class distances", cs.assets.pca_class_distance_panel)}
      ${imageFigure("PCA reconstruction", cs.assets.pca_reconstruction_panel)}
      ${imageFigure("PCA contact sheet", cs.assets.pca_contact_sheet_panel)}
    </div>
    <div class="buttonRow">
      <button id="runPcaAnalysis" class="primary" ${runDisabled}>${runLabel}</button>
      <button ${pcaOpenAttr(cs.paths.pca_folder)} ${pcaOpenDisabled}>Open PCA folder</button>
      <button ${pcaOpenAttr(cs.paths.pca_metrics)} ${pcaOpenDisabled}>Open pca_metrics.json</button>
      <button ${pcaOpenAttr(cs.paths.pca_samples)} ${pcaOpenDisabled}>Open pca_samples.csv</button>
      <button ${pcaOpenAttr(cs.paths.pca_report)} ${pcaOpenDisabled}>Open PCA report</button>
    </div>
    ${!cs.gates.pca_analysis_ready ? `<div class="emptyState">Run PCA analysis first.</div>` : ""}
  `;
}

function renderAutoencoder() {
  const cs = state.caseState;
  const m = cs.metrics.autoencoder || {};
  const counts = m.eval_samples_per_class || {};
  const prerequisitesReady = cs.gates.tracking_validated && cs.gates.dynamic_rois_ready && cs.gates.clean_controls_v02_ready;
  const running = state.autoencoderRunStatus === "running";
  const failed = state.autoencoderRunStatus === "failed";
  const runDisabled = prerequisitesReady && !running ? "" : "disabled";
  const runLabel = running ? "Running Autoencoder..." : (cs.gates.autoencoder_analysis_ready ? "Autoencoder ready" : (failed ? "Autoencoder failed" : "Run autoencoder analysis"));
  const quickLabel = running ? "Running..." : "Run quick autoencoder";
  const aeOpenDisabled = cs.gates.autoencoder_analysis_ready ? "" : "disabled title=\"Run Autoencoder analysis first.\"";
  const aeOpenAttr = (targetPath) => cs.gates.autoencoder_analysis_ready ? `data-open="${targetPath}"` : "";
  return `
    ${moduleHeader("Autoencoder Track-Based + Clean Controls Baseline", "autoencoder", "Human-validated track + dynamic_rois.csv + Controls v0.2 clean masked + PCA baseline", "autoencoder metrics, timeseries CSV, training curve, panels and report")}
    ${!cs.gates.tracking_validated ? `<div class="emptyState">Blocked: requires human-validated tracking.</div>` : ""}
    ${!cs.gates.dynamic_rois_ready ? `<div class="emptyState">Blocked: requires dynamic_rois.csv from track-based ROI.</div>` : ""}
    ${!cs.gates.clean_controls_v02_ready ? `<div class="emptyState">Blocked: requires Controls v0.2 clean masked baseline.</div>` : ""}
    ${!cs.gates.pca_analysis_ready ? `<div class="emptyState">Warning: PCA baseline is expected before autoencoder interpretation.</div>` : ""}
    <div class="emptyState">Autoencoder reconstruction error indicates statistical/visual mismatch under this model; it does not determine origin. Extreme z-scores stay exploratory.</div>
    <div class="cards">
      ${card("Module State", `${kv("track validated", cs.gates.tracking_validated ? "yes" : "no")}${kv("dynamic ROIs", cs.gates.dynamic_rois_ready ? "yes" : "missing")}${kv("clean controls v0.2", cs.gates.clean_controls_v02_ready ? "yes" : "missing")}${kv("PCA baseline", cs.gates.pca_analysis_ready ? "yes" : "missing")}${kv("control validity", fmt(cs.metrics.controls?.control_validity_score))}${kv("autoencoder", cs.gates.autoencoder_analysis_ready ? "ready" : "not run")}`)}
      ${card("Training", `${kv("mode", m.mode)}${kv("strategy", m.training_strategy)}${kv("model", m.model_type)}${kv("latent dim", m.latent_dim)}${kv("epochs", m.epochs)}${kv("train samples", m.train_samples)}${kv("train loss", fmt(m.train_loss_final))}${kv("val loss", fmt(m.val_loss_final))}`)}
      ${card("Eval Samples", `${kv("object", counts.object ?? 0)}${kv("near background", counts.near_background ?? 0)}${kv("far background", counts.far_background ?? 0)}${kv("compression noise", counts.compression_noise ?? 0)}${kv("random background", counts.random_background ?? 0)}${kv("HUD artifact", counts.hud_artifact ?? 0)}`)}
      ${card("Public-safe Metrics", `${kv("mean object z-score", fmt(m.public_safe_zscore))}${kv("object percentile", fmt(m.object_error_percentile_vs_controls))}${kv("public-safe anomaly score", fmt(m.anomaly_score_public_safe))}${kv("object/background ratio", fmt(m.object_vs_background_error_ratio))}${kv("object/compression ratio", fmt(m.object_vs_compression_error_ratio))}${kv("PCA public-safe score", fmt(m.pca_public_safe_score))}`)}
      ${card("Exploratory Metrics", `${kv("max object z-score", fmt(m.exploratory_max_zscore))}${kv("exploratory anomaly score", fmt(m.anomaly_score_exploratory))}${kv("object/HUD ratio", fmt(m.object_vs_hud_error_ratio))}<div class="emptyState">Do not use peak z-score alone as public evidence.</div>`)}
      ${card("Mean Reconstruction Error", `${kv("object", fmt(m.reconstruction_error_mean_object))}${kv("near background", fmt(m.reconstruction_error_mean_near_background))}${kv("far background", fmt(m.reconstruction_error_mean_far_background))}${kv("compression", fmt(m.reconstruction_error_mean_compression_noise))}${kv("random", fmt(m.reconstruction_error_mean_random_background))}${kv("HUD artifact", fmt(m.reconstruction_error_mean_hud_artifact))}`)}
    </div>
    <div class="assetGrid">
      ${imageFigure("Error distribution", cs.assets.autoencoder_error_distribution_panel)}
      ${imageFigure("Error timeseries", cs.assets.autoencoder_timeseries_panel)}
      ${imageFigure("Reconstruction examples", cs.assets.autoencoder_reconstruction_examples_panel)}
      ${imageFigure("Latent PCA", cs.assets.autoencoder_latent_panel)}
      ${imageFigure("Summary", cs.assets.autoencoder_summary_panel)}
    </div>
    <div class="buttonRow">
      <button id="runAutoencoderAnalysis" class="primary" ${runDisabled}>${runLabel}</button>
      <button id="runQuickAutoencoder" ${runDisabled}>${quickLabel}</button>
      <button ${aeOpenAttr(cs.paths.autoencoder_folder)} ${aeOpenDisabled}>Open autoencoder folder</button>
      <button ${aeOpenAttr(cs.paths.autoencoder_metrics)} ${aeOpenDisabled}>Open autoencoder_metrics.json</button>
      <button ${aeOpenAttr(cs.paths.autoencoder_timeseries)} ${aeOpenDisabled}>Open autoencoder_timeseries.csv</button>
      <button ${aeOpenAttr(cs.paths.autoencoder_report)} ${aeOpenDisabled}>Open autoencoder report</button>
      <button ${aeOpenAttr(cs.paths.autoencoder_config)} ${aeOpenDisabled}>Open autoencoder config</button>
    </div>
    ${!cs.gates.autoencoder_analysis_ready ? `<div class="emptyState">Run Autoencoder analysis first.</div>` : ""}
  `;
}

function renderControls() {
  const cs = state.caseState;
  const m = cs.metrics.controls || {};
  const counts = m.controls_generated_per_type || {};
  const missing = m.missing_control_types || [];
  const runDisabled = cs.gates.tracking_validated && cs.gates.dynamic_rois_ready ? "" : "disabled";
  return `
    ${moduleHeader("Controls v0.2 clean masked", "controls", "Human-validated track + dynamic_rois.csv + artifact mask", "clean object-vs-background/HUD/compression controls, rejection CSV, panels and report")}
    ${!cs.gates.tracking_validated ? `<div class="emptyState">Blocked: requires human-validated tracking.</div>` : ""}
    ${!cs.gates.dynamic_rois_ready ? `<div class="emptyState">Blocked: requires dynamic_rois.csv from track-based ROI.</div>` : ""}
    <div class="emptyState">HUD/cyan/black overlays are isolated as artifact controls and excluded from normal background controls.</div>
    <div class="cards">
      ${card("Module State", `${kv("version", m.controls_version || "Controls v0.2 clean masked")}${kv("track validated", cs.gates.tracking_validated ? "yes" : "no")}${kv("dynamic ROIs", cs.gates.dynamic_rois_ready ? "yes" : "missing")}${kv("controls analysis", cs.gates.controls_analysis_ready ? "ready" : "not run")}${kv("validity score", fmt(m.control_validity_score))}`)}
      ${card("Controls Generated", `${kv("near background", counts.near_background ?? 0)}${kv("far background", counts.far_background ?? 0)}${kv("HUD/artifact", counts.hud_artifact ?? 0)}${kv("dark region", counts.dark_region ?? 0)}${kv("bright region", counts.bright_region ?? 0)}${kv("compression/noise", counts.compression_noise ?? 0)}${kv("random background", counts.random_background ?? 0)}`)}
      ${card("Clean Mask Quality", `${kv("artifact contamination", fmt(m.artifact_contamination_rate))}${kv("HUD leakage", fmt(m.hud_leakage_rate))}${kv("fallback rate", fmt(m.fallback_control_rate))}${kv("clean near ratio", fmt(m.clean_near_background_ratio))}${kv("clean far ratio", fmt(m.clean_far_background_ratio))}${kv("clean compression", fmt(m.clean_compression_control_ratio))}`)}
      ${card("Candidate Validation", `${kv("accepted candidates", m.accepted_control_candidates ?? "unknown")}${kv("rejected candidates", m.rejected_control_candidates ?? "unknown")}`)}
      ${card("Object vs Background", `${kv("object luminance", fmt(m.object_luminance_mean))}${kv("near bg luminance", fmt(m.near_background_luminance_mean))}${kv("far bg luminance", fmt(m.far_background_luminance_mean))}${kv("object-near delta", fmt(m.object_vs_near_background_delta_luminance))}${kv("object-far delta", fmt(m.object_vs_far_background_delta_luminance))}`)}
      ${card("Motion / Spectral / Similarity", `${kv("object flow", fmt(m.object_motion_flow_mean))}${kv("background flow", fmt(m.background_motion_flow_mean))}${kv("object HF ratio", fmt(m.object_high_frequency_ratio))}${kv("background HF ratio", fmt(m.background_high_frequency_ratio))}${kv("HUD similarity", fmt(m.hud_similarity_score))}${kv("compression similarity", fmt(m.compression_similarity_score))}${kv("background similarity", fmt(m.background_similarity_score))}`)}
      ${card("Missing Controls", missing.length ? missing.map((item) => `<div class="kv"><span>${item}</span><strong>not available</strong></div>`).join("") : "none")}
    </div>
    <div class="assetGrid">
      ${imageFigure("Controls summary", cs.assets.controls_summary_panel)}
      ${imageFigure("Luminance controls", cs.assets.controls_luminance_panel)}
      ${imageFigure("Thermal / IR controls", cs.assets.controls_thermal_panel)}
      ${imageFigure("Spectral controls", cs.assets.controls_spectral_panel)}
      ${imageFigure("Motion controls", cs.assets.controls_motion_panel)}
      ${imageFigure("Controls contact sheet", cs.assets.controls_contact_sheet)}
      ${imageFigure("Artifact mask debug", cs.assets.controls_artifact_mask_panel)}
      ${imageFigure("Controls quality", cs.assets.controls_quality_panel)}
    </div>
    <div class="buttonRow">
      <button id="runControlsAnalysis" class="primary" ${runDisabled}>Run controls analysis</button>
      <button data-open="${cs.paths.controls_folder}">Open controls folder</button>
      <button data-open="${cs.paths.controls_metrics}">Open controls_metrics.json</button>
      <button data-open="${cs.paths.controls_timeseries}">Open controls_timeseries.csv</button>
      <button data-open="${cs.paths.controls_rejection_report}">Open rejection report</button>
      <button data-open="${cs.paths.controls_report}">Open controls report</button>
    </div>
  `;
}

function renderReports() {
  const cs = state.caseState;
  const unified = cs.metrics.unified || {};
  const moduleStatus = unified.module_status || {};
  const gates = [
    ["Tracking", cs.gates.tracking_validated ? "ready" : "missing"],
    ["Dynamic ROI", cs.gates.dynamic_rois_ready ? "ready" : "missing"],
    ["Motion", cs.gates.motion_analysis_ready ? "ready" : "missing"],
    ["Spectral", cs.gates.spectral_analysis_ready ? "ready" : "missing"],
    ["Thermal / IR", cs.gates.thermal_analysis_ready ? "ready" : "missing"],
    ["SRV", cs.gates.srv_analysis_ready ? "ready" : "missing"],
    ["Controls", cs.gates.clean_controls_v02_ready ? "ready" : "missing"],
    ["PCA", cs.gates.pca_analysis_ready ? "ready" : "missing"],
    ["Autoencoder", cs.gates.autoencoder_analysis_ready ? "ready" : "missing"],
    ["Unified Report", cs.gates.unified_report_ready ? "ready" : "missing"]
  ];
  return `
    ${moduleHeader("Reports", "reports", "Existing report files and unified case report generator", "unified report, summary JSON, metrics card, scorecard and manifest")}
    <div class="cards">
      ${card("Unified Report State", `${kv("unified report", cs.gates.unified_report_ready ? "ready" : "not generated")}${kv("overall assessment", unified.overall_public_safe_assessment || cs.case_status?.overall_public_safe_assessment || "not available")}`)}
      ${card("Workflow Gates", gates.map(([label, value]) => kv(label, value)).join(""))}
      ${card("Module Status", Object.keys(moduleStatus).length ? Object.entries(moduleStatus).map(([label, value]) => kv(label, value)).join("") : "Generate unified report to populate module status.")}
    </div>
    <div class="assetGrid">
      ${imageFigure("Unified case scorecard", cs.assets.unified_scorecard)}
    </div>
    <div class="buttonRow">
      <button id="generateUnifiedReport" class="primary">Generate unified report</button>
      <button data-open="${cs.paths.unified_report}">Open unified report</button>
      <button data-open="${cs.paths.unified_summary}">Open unified summary JSON</button>
      <button data-open="${cs.paths.unified_metrics_card}">Open metrics card JSON</button>
      <button data-open="${cs.paths.unified_report_folder}">Open unified report folder</button>
      <button data-open="${cs.paths.unified_manifest}">Open report manifest</button>
    </div>
    <div class="moduleGrid">
      ${reportLink("Tracking report", cs.assets.tracking_report, cs.paths.tracking_report)}
      ${reportLink("Track-based report", cs.assets.track_based_report, cs.paths.track_based_report)}
      ${reportLink("Motion report", cs.assets.motion_report, cs.paths.motion_report)}
      ${reportLink("Spectral report", cs.assets.spectral_report, cs.paths.spectral_report)}
      ${reportLink("Thermal report", cs.assets.thermal_report, cs.paths.thermal_report)}
      ${reportLink("Controls report", cs.assets.controls_report, cs.paths.controls_report)}
      ${reportLink("PCA report", cs.assets.pca_report, cs.paths.pca_report)}
      ${reportLink("Autoencoder report", cs.assets.autoencoder_report, cs.paths.autoencoder_report)}
      ${reportLink("Unified report", cs.assets.unified_report, cs.paths.unified_report)}
      ${reportLink("Manifest", cs.assets.track_based_manifest, cs.paths.track_based_manifest)}
      ${reportLink("Case status", { exists: true, size: 1 }, cs.paths.case_status)}
    </div>
  `;
}

function renderPublicSummary() {
  const cs = state.caseState;
  const rt = cs.metrics.reddit_template || {};
  const unified = cs.metrics.unified || {};
  const unifiedCard = cs.metrics.unified_card || {};
  const runDisabled = cs.gates.unified_report_ready ? "" : "disabled";
  return `
    ${moduleHeader("Public Summary", "publicSummary", "Unified report + public-safe metrics only", "Reddit technical post template and structured data")}
    ${!cs.gates.unified_report_ready ? `<div class="emptyState">Generate unified report first.</div>` : ""}
    <div class="emptyState">Public release cuts at Unified Case Report. This panel generates only a Reddit technical post template. No origin claim.</div>
    <div class="cards">
      ${card("Reddit Template State", `${kv("reddit template", cs.gates.reddit_template_ready ? "ready" : "not generated")}${kv("unified report", cs.gates.unified_report_ready ? "ready" : "missing")}${kv("no origin claim", rt.no_origin_claim === false ? "check" : "true")}`)}
      ${card("Public-safe Metrics", `${kv("overall assessment", unified.overall_public_safe_assessment || "not available")}${kv("tracking", unifiedCard.tracking_status || "unknown")}${kv("controls", unifiedCard.controls_status || "unknown")}${kv("PCA score", fmt(unifiedCard.pca_public_safe_score))}${kv("AE score", fmt(unifiedCard.autoencoder_public_safe_score))}`)}
      ${card("Template Data", `${kv("case info", rt.case_info ? "loaded" : "not loaded")}${kv("module metrics", rt.module_metrics ? "loaded" : "not loaded")}${kv("attachments", (rt.recommended_attachments || []).length || 0)}`)}
    </div>
    <div class="assetGrid">
      ${imageFigure("Unified technical scorecard", cs.assets.unified_scorecard)}
    </div>
    <div class="buttonRow">
      <button id="generateRedditTemplate" class="primary" ${runDisabled}>Generate Reddit Technical Template</button>
      <button data-open="${cs.paths.reddit_template_en}">Open long Reddit template</button>
      <button data-open="${cs.paths.reddit_template_short_en}">Open short Reddit post</button>
      <button data-open="${cs.paths.reddit_template_titles_en}">Open title options</button>
      <button data-open="${cs.paths.reddit_template_data}">Open data JSON</button>
      <button data-open="${cs.paths.reddit_template_folder}">Open reddit template folder</button>
    </div>
  `;
}

function shellModule(title, id, req, outputs, button, warning = "Not available yet") {
  return `${moduleHeader(title, id, req, outputs)}<div class="emptyState">${warning}</div><button disabled>${button} - not implemented yet</button>`;
}

function card(title, body) {
  return `<section class="moduleCard"><h3>${title}</h3><div>${body}</div></section>`;
}

function kv(label, value) {
  return `<div class="kv"><span>${label}</span><strong>${value ?? "unknown"}</strong></div>`;
}

function fmt(value) {
  return Number.isFinite(Number(value)) ? Number(value).toFixed(4) : "unknown";
}

function imageFigure(label, asset) {
  return `<figure><figcaption>${label}</figcaption>${imageIf(asset, label)}</figure>`;
}

function reportLink(label, asset, filePath) {
  const ok = exists(asset) || (filePath && !filePath.includes("undefined"));
  return `<div class="reportLine"><span>${label}</span><code>${filePath || "not available"}</code><button data-open="${filePath}" ${ok ? "" : "disabled"}>Open</button></div>`;
}

function afterTabRender() {
  if (state.activeTab === "video") {
    syncCanvas();
    updateFrameReadout();
  }
}

function currentFrame() {
  const video = $("caseVideo");
  return video ? Math.max(0, Math.round((video.currentTime || 0) * state.fps)) : 0;
}

function currentTimeSeconds() {
  const video = $("caseVideo");
  return video ? Number(video.currentTime || 0) : 0;
}

function maxVideoFrame() {
  const video = $("caseVideo");
  if (!video || !Number.isFinite(video.duration) || video.duration <= 0) return null;
  return Math.max(0, Math.round(video.duration * state.fps));
}

function seekToFrame(frame) {
  const video = $("caseVideo");
  if (!video) return;
  const maxFrame = maxVideoFrame();
  const target = Math.max(0, Math.round(Number(frame) || 0));
  const clamped = maxFrame == null ? target : Math.min(target, maxFrame);
  video.currentTime = clamped / state.fps;
  updateFrameReadout();
}

function seekFrames(delta) {
  const video = $("caseVideo");
  if (!video) return;
  seekToFrame(currentFrame() + delta);
}

function syncCanvas() {
  const video = $("caseVideo");
  const canvas = $("annotationCanvas");
  if (!video || !canvas) return;
  const rect = video.getBoundingClientRect();
  canvas.width = Math.max(1, Math.round(rect.width));
  canvas.height = Math.max(1, Math.round(rect.height));
  canvas.style.width = `${rect.width}px`;
  canvas.style.height = `${rect.height}px`;
  drawCanvas();
}

function canvasToVideo(point) {
  const video = $("caseVideo");
  const canvas = $("annotationCanvas");
  const sx = (video.videoWidth || canvas.width) / canvas.width;
  const sy = (video.videoHeight || canvas.height) / canvas.height;
  return { x: Math.round(point.x * sx), y: Math.round(point.y * sy) };
}

function videoToCanvasBox(box) {
  const video = $("caseVideo");
  const canvas = $("annotationCanvas");
  const sx = canvas.width / (video.videoWidth || canvas.width);
  const sy = canvas.height / (video.videoHeight || canvas.height);
  return { x: box.x * sx, y: box.y * sy, w: box.w * sx, h: box.h * sy };
}

function drawCanvas() {
  const canvas = $("annotationCanvas");
  if (!canvas) return;
  const ctx = canvas.getContext("2d");
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  const active = state.current && state.start ? {
    x: Math.min(state.start.x, state.current.x),
    y: Math.min(state.start.y, state.current.y),
    w: Math.abs(state.current.x - state.start.x),
    h: Math.abs(state.current.y - state.start.y)
  } : (state.box ? videoToCanvasBox(state.box) : null);
  if (!active) return;
  ctx.strokeStyle = state.start && state.current ? "#ffd43b" : "#ff3333";
  ctx.lineWidth = 3;
  ctx.strokeRect(active.x, active.y, active.w, active.h);
}

function pointerPosition(event) {
  const rect = $("annotationCanvas").getBoundingClientRect();
  return { x: event.clientX - rect.left, y: event.clientY - rect.top };
}

function updateFrameReadout() {
  const readout = $("frameReadout");
  const video = $("caseVideo");
  if (readout && video) readout.textContent = `frame ${currentFrame()} | time ${video.currentTime.toFixed(3)}s`;
}

async function setCurrentFrameAsStart() {
  if (!state.selectedCase) return;
  const frame = currentFrame();
  const time = currentTimeSeconds();
  const result = await window.forensicDesk.setStartFrame({ caseId: state.selectedCase, frame, time });
  const message = `Tracking start set to frame ${frame} | time ${time.toFixed(3)}s`;
  if ($("startFrameStatus")) $("startFrameStatus").textContent = message;
  $("annotationStatus").textContent = message;
  $("requestPreview").textContent = pretty(result.request);
  state.selectedCaseData.request = result.request;
  state.caseState = await window.forensicDesk.getCaseState(state.batchId, state.selectedCase);
  log(`Set tracking start: frame ${frame}`);
}

async function saveBox() {
  if (!state.selectedCase) return;
  let box = state.box;
  if (state.start && state.current) {
    const p0 = canvasToVideo(state.start);
    const p1 = canvasToVideo(state.current);
    const x = Math.min(p0.x, p1.x);
    const y = Math.min(p0.y, p1.y);
    box = { x, y, w: Math.max(1, Math.abs(p1.x - p0.x)), h: Math.max(1, Math.abs(p1.y - p0.y)) };
  }
  if (!box) {
    $("annotationStatus").textContent = "Draw a box before saving.";
    return;
  }
  state.box = box;
  const frame = currentFrame();
  const time = currentTimeSeconds();
  const result = await window.forensicDesk.saveAnnotation({ caseId: state.selectedCase, frame, time, box });
  const message = `Saved frame ${frame} at ${time.toFixed(3)}s | box x=${box.x} y=${box.y} w=${box.w} h=${box.h}`;
  $("annotationStatus").textContent = message;
  if ($("startFrameStatus")) $("startFrameStatus").textContent = `Tracking start: frame ${frame} | time ${time.toFixed(3)}s`;
  $("requestPreview").textContent = pretty(result.request);
  log(`Saved annotation: frame ${frame}, time ${time.toFixed(3)}s`);
  state.caseState = await window.forensicDesk.getCaseState(state.batchId, state.selectedCase);
  state.selectedCaseData.request = result.request;
}

async function runTracking() {
  if (!state.selectedCase) return;
  clearTrackingTimer();
  state.trackingRunStartedAt = Date.now();
  state.lastProgressPercent = null;
  setTrackingProgress({ stage: "starting", message: "Tracking starting..." });
  state.trackingTimer = window.setInterval(() => {
    const elapsed = Math.round((Date.now() - state.trackingRunStartedAt) / 1000);
    if (state.lastProgressPercent == null) {
      setTrackingProgress({
        stage: "starting",
        message: `Tracking running... elapsed ${elapsed}s. Tracking can take several minutes for long videos.`,
      });
    }
  }, 1000);
  log(`Running tracking for ${state.selectedCase}`);
  const result = await window.forensicDesk.runTracking({ caseId: state.selectedCase, backend: "opencv" });
  clearTrackingTimer();
  log(`Command: ${result.command}`);
  log(`Exit code: ${result.code}`);
  log(`Tracking log: ${result.log_path}`);
  log(`track.json: ${result.track_json?.path || ""} exists=${result.track_json?.exists} size=${result.track_json?.size}`);
  log(`tracking_quality.json: ${result.tracking_quality?.path || ""} exists=${result.tracking_quality?.exists} size=${result.tracking_quality?.size}`);
  log(`raw mp4: ${result.overlay?.path || ""} exists=${result.overlay?.exists} size=${result.overlay?.size}`);
  log(`web mp4: ${result.web_overlay?.path || ""} exists=${result.web_overlay?.exists} size=${result.web_overlay?.size}`);
  log(`preview url: ${result.preview_url || "none"}`);
  if (result.code !== 0) {
    setTrackingProgress({ stage: "failed", message: "Tracking failed" });
    return;
  }
  if (!result.track_json?.exists || result.track_json?.size <= 0 || !result.preview?.exists || result.preview?.size <= 0 || !result.preview_url) {
    setTrackingProgress({ stage: "preview_missing", message: "Tracking generated but preview missing" });
    return;
  }
  state.caseState = await window.forensicDesk.getCaseState(state.batchId, state.selectedCase);
  renderTabs();
  renderActiveTab();
  loadPreview(result.preview_url, true);
  setTrackingProgress({ stage: "complete", percent: 100, message: "Tracking complete - preview loading..." });
  const quality = state.caseState.metrics?.tracking_quality;
  if (quality) {
    log(`Tracking recommendation: ${quality.recommendation}`);
    log(`Tracking quality: mean=${quality.mean_confidence} recovered=${quality.auto_recovered_frames} low=${quality.low_confidence_frames} lost=${quality.lost_frames} hud_rejected=${quality.hud_rejected_candidates} reticle_rejected=${quality.reticle_rejected_candidates}`);
  }
}

function clearTrackingTimer() {
  if (state.trackingTimer) {
    window.clearInterval(state.trackingTimer);
    state.trackingTimer = null;
  }
}

function progressMessage(payload) {
  if (payload.stage === "tracking" && payload.total != null && payload.frame != null) {
    const percent = payload.percent != null ? Number(payload.percent).toFixed(1) : "0.0";
    return `Tracking frame ${payload.frame} / ${payload.total} (${percent}%)`;
  }
  if (payload.message) return payload.message;
  return String(payload.stage || "Tracking");
}

function setTrackingProgress(payload) {
  const message = progressMessage(payload);
  const percent = payload.percent != null ? Math.max(0, Math.min(100, Number(payload.percent))) : state.lastProgressPercent;
  if (payload.percent != null) state.lastProgressPercent = percent;
  if ($("trackingStatus")) $("trackingStatus").textContent = message;
  if ($("trackingStage")) $("trackingStage").textContent = payload.stage ? `Stage: ${payload.stage}` : "Stage: tracking";
  if ($("trackingProgressText")) $("trackingProgressText").textContent = message;
  if ($("trackingProgressBar") && percent != null && Number.isFinite(percent)) $("trackingProgressBar").value = percent;
}

function handleTrackingProgress(payload) {
  if (!payload || payload.caseId !== state.selectedCase) return;
  setTrackingProgress(payload);
  const message = progressMessage(payload);
  if (payload.stage !== "tracking" || payload.frame === 0 || payload.percent === 100 || Math.round(Number(payload.percent || 0)) % 10 === 0) {
    log(message);
  }
}

function loadPreview(url, expectLoad) {
  const video = $("overlayVideo");
  if (!video) return;
  state.pendingPreviewLoad = Boolean(expectLoad);
  if (!url) {
    video.removeAttribute("src");
    video.load();
    if (expectLoad) $("trackingStatus").textContent = "Tracking generated but preview missing";
    return;
  }
  video.src = url;
  video.load();
  if (expectLoad) $("trackingStatus").textContent = "Tracking generated - loading preview...";
}

async function reloadPreview() {
  if (!state.selectedCase) return;
  const data = await window.forensicDesk.getCaseState(state.batchId, state.selectedCase);
  state.caseState = data;
  if (data.assets.overlay.url) {
    $("trackingStatus").textContent = "Reloading preview...";
    loadPreview(data.assets.overlay.url, true);
  } else {
    $("trackingStatus").textContent = "Tracking generated but preview missing";
  }
}

async function saveValidation(decision) {
  if (!state.selectedCase) return;
  const result = await window.forensicDesk.saveValidation({ caseId: state.selectedCase, decision, notes: $("validationNotes").value });
  $("validationPreview").textContent = pretty(result.validation);
  log(`Saved validation: ${result.validation_path}`);
  state.caseState = await window.forensicDesk.getCaseState(state.batchId, state.selectedCase);
  await loadBatch();
}

async function rebuildFromTrack() {
  if (!state.selectedCase || !state.caseState?.gates?.tracking_validated) return;
  log(`Rebuilding track-based analysis for ${state.selectedCase}`);
  const result = await window.forensicDesk.rebuildFromTrack({ caseId: state.selectedCase });
  log(`Command: ${result.command}`);
  log(`Exit code: ${result.code}`);
  log(`Log: ${result.log_path}`);
  state.caseState = await window.forensicDesk.getCaseState(state.batchId, state.selectedCase);
  renderTabs();
  renderActiveTab();
}

async function runMotionAnalysis() {
  if (!state.selectedCase || !state.caseState?.gates?.tracking_validated || !state.caseState?.gates?.dynamic_rois_ready) return;
  log(`Running track-based motion analysis for ${state.selectedCase}`);
  const result = await window.forensicDesk.runMotionAnalysis({ caseId: state.selectedCase });
  log(`Command: ${result.command}`);
  log(`CWD: ${result.cwd}`);
  log(`Exit code: ${result.code}`);
  log(`Log: ${result.log_path}`);
  log(`Motion outputs: ${pretty(result.outputs)}`);
  if (result.stderr) log(`stderr: ${result.stderr}`);
  state.caseState = await window.forensicDesk.getCaseState(state.batchId, state.selectedCase);
  renderTabs();
  renderActiveTab();
}

async function runSpectralAnalysis() {
  if (!state.selectedCase || !state.caseState?.gates?.tracking_validated || !state.caseState?.gates?.dynamic_rois_ready) return;
  log(`Running track-based spectral analysis for ${state.selectedCase}`);
  const result = await window.forensicDesk.runSpectralAnalysis({ caseId: state.selectedCase });
  log(`Command: ${result.command}`);
  log(`CWD: ${result.cwd}`);
  log(`Exit code: ${result.code}`);
  log(`Log: ${result.log_path}`);
  log(`Spectral outputs: ${pretty(result.outputs)}`);
  if (result.stderr) log(`stderr: ${result.stderr}`);
  state.caseState = await window.forensicDesk.getCaseState(state.batchId, state.selectedCase);
  renderTabs();
  renderActiveTab();
}

async function runThermalAnalysis() {
  if (!state.selectedCase || !state.caseState?.gates?.tracking_validated || !state.caseState?.gates?.dynamic_rois_ready) return;
  log(`Running track-based Thermal / IR analysis for ${state.selectedCase}`);
  const result = await window.forensicDesk.runThermalAnalysis({ caseId: state.selectedCase });
  log(`Command: ${result.command}`);
  log(`CWD: ${result.cwd}`);
  log(`Exit code: ${result.code}`);
  log(`Log: ${result.log_path}`);
  log(`Thermal outputs: ${pretty(result.outputs)}`);
  if (result.stderr) log(`stderr: ${result.stderr}`);
  state.caseState = await window.forensicDesk.getCaseState(state.batchId, state.selectedCase);
  renderTabs();
  renderActiveTab();
}

async function runSrvAnalysis() {
  if (!state.selectedCase || !state.caseState?.gates?.tracking_validated || !state.caseState?.gates?.dynamic_rois_ready) return;
  log(`Running track-based SRV / visual reconstruction for ${state.selectedCase}`);
  const result = await window.forensicDesk.runSrvAnalysis({ caseId: state.selectedCase });
  log(`Command: ${result.command}`);
  log(`CWD: ${result.cwd}`);
  log(`Exit code: ${result.code}`);
  log(`Log: ${result.log_path}`);
  log(`SRV outputs: ${pretty(result.outputs)}`);
  if (result.stderr) log(`stderr: ${result.stderr}`);
  state.caseState = await window.forensicDesk.getCaseState(state.batchId, state.selectedCase);
  renderTabs();
  renderActiveTab();
}

async function runSrvCoreAnalysis() {
  if (!state.selectedCase || !state.caseState?.gates?.tracking_validated || !state.caseState?.gates?.dynamic_rois_ready) return;
  log(`Running object-core SRV / visual reconstruction for ${state.selectedCase}`);
  const result = await window.forensicDesk.runSrvCoreAnalysis({ caseId: state.selectedCase });
  log(`Command: ${result.command}`);
  log(`CWD: ${result.cwd}`);
  log(`Exit code: ${result.code}`);
  log(`Log: ${result.log_path}`);
  log(`Object-core SRV outputs: ${pretty(result.outputs)}`);
  if (result.stderr) log(`stderr: ${result.stderr}`);
  state.caseState = await window.forensicDesk.getCaseState(state.batchId, state.selectedCase);
  renderTabs();
  renderActiveTab();
}

async function runControlsAnalysis() {
  if (!state.selectedCase || !state.caseState?.gates?.tracking_validated || !state.caseState?.gates?.dynamic_rois_ready) return;
  log(`Running track-based controls analysis for ${state.selectedCase}`);
  const result = await window.forensicDesk.runControlsAnalysis({ caseId: state.selectedCase });
  log(`Command: ${result.command}`);
  log(`CWD: ${result.cwd}`);
  log(`Exit code: ${result.code}`);
  log(`Log: ${result.log_path}`);
  log(`Controls outputs: ${pretty(result.outputs)}`);
  if (result.stderr) log(`stderr: ${result.stderr}`);
  state.caseState = await window.forensicDesk.getCaseState(state.batchId, state.selectedCase);
  renderTabs();
  renderActiveTab();
}

async function runPcaAnalysis() {
  if (!state.selectedCase || !state.caseState?.gates?.tracking_validated || !state.caseState?.gates?.dynamic_rois_ready || !state.caseState?.gates?.clean_controls_v02_ready) return;
  state.pcaRunStatus = "running";
  renderActiveTab();
  log(`Starting PCA analysis for case ${state.selectedCase}`);
  try {
    const result = await window.forensicDesk.runPcaAnalysis({ caseId: state.selectedCase });
    log(`Command: ${result.command}`);
    log(`CWD: ${result.cwd}`);
    log(`Exit code: ${result.code}`);
    log(`Log: ${result.log_path}`);
    log(`PCA outputs: ${pretty(result.outputs)}`);
    if (result.stderr) log(`stderr: ${result.stderr}`);
    if (!result.ok) {
      state.pcaRunStatus = "failed";
      log(`PCA failed: ${result.stderr || result.stdout || "required outputs were not generated"}`);
    } else {
      state.pcaRunStatus = "ready";
      log("PCA analysis complete");
    }
  } catch (error) {
    state.pcaRunStatus = "failed";
    log(`PCA failed: ${error.message}`);
  }
  state.caseState = await window.forensicDesk.getCaseState(state.batchId, state.selectedCase);
  if (state.caseState?.gates?.pca_analysis_ready) state.pcaRunStatus = "ready";
  renderTabs();
  renderActiveTab();
}

async function runAutoencoderAnalysis(quick = false) {
  if (!state.selectedCase || !state.caseState?.gates?.tracking_validated || !state.caseState?.gates?.dynamic_rois_ready || !state.caseState?.gates?.clean_controls_v02_ready) return;
  const ok = window.confirm("Autoencoder can take longer than other modules. Continue?");
  if (!ok) {
    log("Autoencoder run cancelled by user.");
    return;
  }
  state.autoencoderRunStatus = "running";
  renderActiveTab();
  log(`Starting ${quick ? "quick " : ""}Autoencoder analysis for case ${state.selectedCase}`);
  try {
    const result = await window.forensicDesk.runAutoencoderAnalysis({ caseId: state.selectedCase, quick });
    log(`Command: ${result.command}`);
    log(`CWD: ${result.cwd}`);
    log(`Exit code: ${result.code}`);
    log(`Log: ${result.log_path}`);
    log(`Autoencoder outputs: ${pretty(result.outputs)}`);
    if (result.stderr) log(`stderr: ${result.stderr}`);
    if (!result.ok) {
      state.autoencoderRunStatus = "failed";
      log(`Autoencoder failed: ${result.stderr || result.stdout || "required outputs were not generated"}`);
    } else {
      state.autoencoderRunStatus = "ready";
      log("Autoencoder analysis complete");
    }
  } catch (error) {
    state.autoencoderRunStatus = "failed";
    log(`Autoencoder failed: ${error.message}`);
  }
  state.caseState = await window.forensicDesk.getCaseState(state.batchId, state.selectedCase);
  if (state.caseState?.gates?.autoencoder_analysis_ready) state.autoencoderRunStatus = "ready";
  renderTabs();
  renderActiveTab();
}

async function generateUnifiedReport() {
  if (!state.selectedCase) return;
  log(`Generating unified case report for ${state.selectedCase}`);
  const result = await window.forensicDesk.generateUnifiedReport({ caseId: state.selectedCase });
  log(`Command: ${result.command}`);
  log(`CWD: ${result.cwd}`);
  log(`Exit code: ${result.code}`);
  log(`Log: ${result.log_path}`);
  log(`Unified report outputs: ${pretty(result.outputs)}`);
  if (result.stderr) log(`stderr: ${result.stderr}`);
  state.caseState = await window.forensicDesk.getCaseState(state.batchId, state.selectedCase);
  renderTabs();
  renderActiveTab();
}

async function generateRedditTemplate() {
  if (!state.selectedCase) return;
  if (!state.caseState?.gates?.unified_report_ready) {
    log("Generate unified report first.");
    return;
  }
  log(`Generating Reddit technical post template for ${state.selectedCase}`);
  const result = await window.forensicDesk.generateRedditTemplate({ caseId: state.selectedCase });
  log(`Command: ${result.command}`);
  log(`CWD: ${result.cwd}`);
  log(`Exit code: ${result.code}`);
  log(`Log: ${result.log_path}`);
  log(`Reddit template outputs: ${pretty(result.outputs)}`);
  if (result.stderr) log(`stderr: ${result.stderr}`);
  state.caseState = await window.forensicDesk.getCaseState(state.batchId, state.selectedCase);
  renderTabs();
  renderActiveTab();
}

async function openTarget(targetPath) {
  const result = await window.forensicDesk.openPath(targetPath);
  if (!result.ok) log(`Open failed: ${result.error}`);
}

async function openRuntimeFolder() {
  const target = state.config?.lab_root || state.config?.writable_runtime_root;
  if (!target) return;
  await openTarget(target);
}

async function importVideo(videoPath = null) {
  if (videoPath && !isSupportedVideoPath(videoPath)) {
    log("Unsupported file type.");
    return;
  }
  log("Import Video: selecting/copying local video into portable runtime.");
  const result = await window.forensicDesk.importVideo(videoPath ? { videoPath } : {});
  if (result.canceled) {
    log(result.message || "Import cancelled.");
    return;
  }
  if (!result.ok) {
    log(`Import failed: ${result.error || "unknown error"}`);
    if (result.stderr) log(`stderr: ${result.stderr}`);
    return;
  }
  log(`Imported video as ${result.case_id}`);
  log(`Copied source: ${result.source_video}`);
  await refreshBatchSelect(result.batch_id);
  await loadBatch(result.case_id);
}

function bindEvents() {
  $("reloadBatch").addEventListener("click", loadBatch);
  $("importVideoTop").addEventListener("click", () => importVideo());
  $("openRuntimeTop").addEventListener("click", openRuntimeFolder);
  $("batchSelect").addEventListener("change", loadBatch);
  $("casesTable").addEventListener("click", (event) => {
    const row = event.target.closest("tr[data-case]");
    if (row) openCase(row.dataset.case);
  });
  $("moduleTabs").addEventListener("click", (event) => {
    const button = event.target.closest("button[data-tab]");
    if (!button) return;
    state.activeTab = button.dataset.tab;
    renderTabs();
    renderActiveTab();
  });
  $("moduleContent").addEventListener("click", async (event) => {
    const target = event.target;
    if (target.id === "prevFrame") seekFrames(-1);
    if (target.id === "nextFrame") seekFrames(1);
    if (target.id === "back100") seekFrames(-100);
    if (target.id === "back10") seekFrames(-10);
    if (target.id === "forward10") seekFrames(10);
    if (target.id === "forward100") seekFrames(100);
    if (target.id === "goToFrameButton") seekToFrame($("goToFrameInput")?.value);
    if (target.id === "setStartFrame") await setCurrentFrameAsStart();
    if (target.id === "importVideoEmpty") await importVideo();
    if (target.id === "openRuntimeEmpty") await openRuntimeFolder();
    if (target.id === "saveBox") await saveBox();
    if (target.id === "runTracking") await runTracking();
    if (target.id === "reloadPreview") await reloadPreview();
    if (target.id === "openTrackingFolder") await openTarget(state.caseState.paths.tracking_folder);
    if (target.id === "openTrackJson") await openTarget(state.caseState.paths.track_json);
    if (target.id === "validTrack") await saveValidation("correct");
    if (target.id === "invalidTrack") await saveValidation("incorrect");
    if (target.id === "repromptTrack") await saveValidation("reprompt");
    if (target.id === "rebuildFromTrack") await rebuildFromTrack();
    if (target.id === "runMotionAnalysis") await runMotionAnalysis();
    if (target.id === "runSpectralAnalysis") await runSpectralAnalysis();
    if (target.id === "runThermalAnalysis") await runThermalAnalysis();
    if (target.id === "runSrvAnalysis") await runSrvAnalysis();
    if (target.id === "runSrvCoreAnalysis") await runSrvCoreAnalysis();
    if (target.id === "runControlsAnalysis") await runControlsAnalysis();
    if (target.id === "runPcaAnalysis") await runPcaAnalysis();
    if (target.id === "runAutoencoderAnalysis") await runAutoencoderAnalysis(false);
    if (target.id === "runQuickAutoencoder") await runAutoencoderAnalysis(true);
    if (target.id === "generateUnifiedReport") await generateUnifiedReport();
    if (target.id === "generateRedditTemplate") await generateRedditTemplate();
    const openButton = target.closest("[data-open]");
    if (openButton) await openTarget(openButton.dataset.open);
  });
  $("moduleContent").addEventListener("dragover", (event) => {
    event.preventDefault();
    const zone = event.target.closest("#videoDropZone");
    if (zone) zone.classList.add("dragOver");
  });
  $("moduleContent").addEventListener("dragleave", (event) => {
    const zone = event.target.closest("#videoDropZone");
    if (zone) zone.classList.remove("dragOver");
  });
  $("moduleContent").addEventListener("drop", async (event) => {
    event.preventDefault();
    const zone = event.target.closest("#videoDropZone");
    if (zone) zone.classList.remove("dragOver");
    const file = event.dataTransfer?.files?.[0];
    const filePath = file?.path || "";
    if (!filePath) {
      log("Drag and drop did not expose a local file path. Use Import Video.");
      return;
    }
    await importVideo(filePath);
  });
  $("moduleContent").addEventListener("pointerdown", (event) => {
    if (event.target.id !== "annotationCanvas") return;
    state.drawing = true;
    state.start = pointerPosition(event);
    state.current = state.start;
    drawCanvas();
  });
  $("moduleContent").addEventListener("pointermove", (event) => {
    if (!state.drawing || event.target.id !== "annotationCanvas") return;
    state.current = pointerPosition(event);
    drawCanvas();
  });
  $("moduleContent").addEventListener("pointerup", (event) => {
    if (event.target.id !== "annotationCanvas") return;
    state.drawing = false;
    state.current = pointerPosition(event);
    drawCanvas();
  });
  $("moduleContent").addEventListener("loadedmetadata", (event) => {
    if (event.target.id === "caseVideo") syncCanvas();
  }, true);
  $("moduleContent").addEventListener("timeupdate", (event) => {
    if (event.target.id === "caseVideo") updateFrameReadout();
  }, true);
  $("moduleContent").addEventListener("keydown", (event) => {
    if (event.target.id === "goToFrameInput" && event.key === "Enter") {
      seekToFrame(event.target.value);
    }
  });
  $("moduleContent").addEventListener("loadeddata", (event) => {
    if (event.target.id === "overlayVideo" && state.pendingPreviewLoad) {
      $("trackingStatus").textContent = "Tracking complete - preview loaded";
      state.pendingPreviewLoad = false;
    }
  }, true);
  $("moduleContent").addEventListener("error", (event) => {
    if (event.target.id === "overlayVideo" && state.pendingPreviewLoad) {
      $("trackingStatus").textContent = "Tracking video not playable";
      state.pendingPreviewLoad = false;
    }
  }, true);
  window.addEventListener("resize", syncCanvas);
}

bindEvents();
init().catch((error) => {
  $("labStatus").textContent = `Error: ${error.message}`;
  log(error.stack || error.message);
});
