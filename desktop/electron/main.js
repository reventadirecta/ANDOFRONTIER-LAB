const { app, BrowserWindow, ipcMain, shell, dialog } = require("electron");
const path = require("node:path");
const fs = require("node:fs");
const fsp = require("node:fs/promises");
const { spawn } = require("node:child_process");
const { pathToFileURL } = require("node:url");

const APP_ROOT = path.resolve(__dirname, "..");
const PACKAGED_BACKEND_EXE = path.join(process.resourcesPath || APP_ROOT, "backend", "andofrontier-lab-cli.exe");
const DEV_RUNTIME_ROOT = APP_ROOT;
const APP_CONFIG_EXAMPLE_PATH = path.join(APP_ROOT, "config", "lab.local.example.json");

function canWrite(dirPath) {
  try {
    fs.mkdirSync(dirPath, { recursive: true });
    const probe = path.join(dirPath, ".write-test");
    fs.writeFileSync(probe, "ok");
    fs.unlinkSync(probe);
    return true;
  } catch {
    return false;
  }
}

function getWritableRuntimeRoot() {
  if (!app.isPackaged) return DEV_RUNTIME_ROOT;
  const candidates = [
    process.env.PORTABLE_EXECUTABLE_DIR,
    process.env.PORTABLE_EXECUTABLE_FILE ? path.dirname(process.env.PORTABLE_EXECUTABLE_FILE) : null,
    path.dirname(process.execPath),
  ].filter(Boolean);
  for (const candidate of candidates) {
    if (candidate && !candidate.toLowerCase().includes(`${path.sep}temp${path.sep}`) && canWrite(candidate)) {
      return candidate;
    }
  }
  const fallback = path.join(app.getPath("userData"), "portable-runtime");
  fs.mkdirSync(fallback, { recursive: true });
  return fallback;
}

function runtimePath(...parts) {
  return path.join(getWritableRuntimeRoot(), ...parts);
}

function logDir() {
  return runtimePath("runtime", "logs");
}

function initRuntime(config) {
  const dirs = [
    runtimePath("runtime"),
    config.lab_root,
    path.join(config.lab_root, "data"),
    path.join(config.lab_root, "data", "batches"),
    path.join(config.lab_root, "data", "cases"),
    path.join(config.lab_root, "data", "outputs"),
    path.join(config.lab_root, "data", "reports"),
    path.join(config.lab_root, "config"),
    logDir(),
  ];
  for (const dir of dirs) fs.mkdirSync(dir, { recursive: true });
}

function readJson(filePath) {
  return JSON.parse(fs.readFileSync(filePath, "utf8"));
}

async function writeJson(filePath, payload) {
  await fsp.mkdir(path.dirname(filePath), { recursive: true });
  await fsp.writeFile(filePath, JSON.stringify(payload, null, 2) + "\n", "utf8");
}

function loadConfig() {
  if (app.isPackaged) {
    const writableRoot = getWritableRuntimeRoot();
    const config = {
      writable_runtime_root: writableRoot,
      lab_root: path.join(writableRoot, "runtime", "lab"),
      python_exe: PACKAGED_BACKEND_EXE,
      default_batch_id: "local_batch",
      config_path: path.join(writableRoot, "config", "lab.local.json"),
      config_missing: false,
      config_required_path: path.join(writableRoot, "config", "lab.local.json"),
      config_example_path: path.join(writableRoot, "config", "lab.local.example.json"),
      packaged_backend: true
    };
    initRuntime(config);
    if (fs.existsSync(APP_CONFIG_EXAMPLE_PATH) && !fs.existsSync(config.config_example_path)) {
      fs.mkdirSync(path.dirname(config.config_example_path), { recursive: true });
      fs.copyFileSync(APP_CONFIG_EXAMPLE_PATH, config.config_example_path);
    }
    process.env.ANDOFRONTIER_LAB_ROOT = config.lab_root;
    config.lab_exists = fs.existsSync(config.lab_root);
    config.python_exists = fs.existsSync(config.python_exe);
    config.runtime_initialized = true;
    return config;
  }
  const CONFIG_PATH = runtimePath("config", "lab.local.json");
  const RUNTIME_CONFIG_EXAMPLE_PATH = runtimePath("config", "lab.local.example.json");
  const selected = fs.existsSync(CONFIG_PATH)
    ? CONFIG_PATH
    : (fs.existsSync(RUNTIME_CONFIG_EXAMPLE_PATH) ? RUNTIME_CONFIG_EXAMPLE_PATH : APP_CONFIG_EXAMPLE_PATH);
  const config = readJson(selected);
  config.config_path = selected;
  config.config_missing = !fs.existsSync(CONFIG_PATH);
  config.config_required_path = CONFIG_PATH;
  config.config_example_path = fs.existsSync(RUNTIME_CONFIG_EXAMPLE_PATH) ? RUNTIME_CONFIG_EXAMPLE_PATH : APP_CONFIG_EXAMPLE_PATH;
  process.env.ANDOFRONTIER_LAB_ROOT = config.lab_root;
  config.lab_exists = fs.existsSync(config.lab_root);
  config.python_exists = fs.existsSync(config.python_exe);
  return config;
}

function labPath(...parts) {
  return path.join(loadConfig().lab_root, ...parts);
}

function fileUrl(filePath, cacheBust = false) {
  if (!filePath || !fs.existsSync(filePath)) return null;
  const url = pathToFileURL(filePath).toString();
  return cacheBust ? `${url}?t=${Date.now()}` : url;
}

function backendCommand(config, command, args = []) {
  if (config.packaged_backend) {
    return {
      command: config.python_exe,
      exe: config.python_exe,
      args: [command, ...args],
      cwd: config.lab_root,
      text: `"${config.python_exe}" ${[command, ...args].join(" ")}`
    };
  }
  const cliScript = path.join(config.lab_root, "scripts", "andofrontier_lab_cli.py");
  return {
    command: config.python_exe,
    exe: config.python_exe,
    args: [cliScript, command, ...args],
    cwd: config.lab_root,
    text: `"${config.python_exe}" "${cliScript}" ${[command, ...args].join(" ")}`
  };
}

function runBackendJson(config, command, args = []) {
  const cmd = backendCommand(config, command, args);
  return new Promise((resolve) => {
    const child = spawn(cmd.exe, cmd.args, { cwd: cmd.cwd, windowsHide: true });
    let stdout = "";
    let stderr = "";
    child.stdout.on("data", (data) => { stdout += data.toString(); });
    child.stderr.on("data", (data) => { stderr += data.toString(); });
    child.on("close", (code) => {
      try {
        const match = stdout.match(/\{[\s\S]*\}\s*$/);
        const parsed = match ? JSON.parse(match[0]) : JSON.parse(stdout);
        resolve({ ...parsed, code, command: cmd.text, stderr });
      } catch {
        resolve({
          ok: false,
          code,
          command: cmd.text,
          error: "Backend did not return valid JSON.",
          stdout,
          stderr
        });
      }
    });
    child.on("error", (error) => {
      resolve({ ok: false, command: cmd.text, error: error.message, stdout, stderr });
    });
  });
}

function runCommandCapture(exe, args, cwd, logPath, onData = null) {
  return new Promise((resolve) => {
    const child = spawn(exe, args, { cwd, windowsHide: true });
    const stream = fs.createWriteStream(logPath, { flags: "a" });
    let stdout = "";
    let stderr = "";
    child.stdout.on("data", (data) => {
      const text = data.toString();
      stdout += text;
      stream.write(text);
      if (onData) onData(text);
    });
    child.stderr.on("data", (data) => {
      const text = data.toString();
      stderr += text;
      stream.write(text);
      if (onData) onData(text);
    });
    child.on("close", (code) => {
      stream.end();
      resolve({ code, stdout, stderr });
    });
    child.on("error", (error) => {
      stream.end();
      resolve({ code: 1, stdout, stderr, error: error.message });
    });
  });
}

function fileInfo(filePath) {
  if (!filePath || !fs.existsSync(filePath)) return { exists: false, size: 0, path: filePath };
  const stat = fs.statSync(filePath);
  return { exists: true, size: stat.size, path: filePath };
}

function emptyInteractiveRequest(caseId) {
  return {
    case_id: caseId,
    tracking_mode: "manual_first_appearance",
    backend_preference: ["sam2", "cotracker", "cutie", "xmem", "opencv"],
    object_prompt: { type: "box_or_points", box: null, positive_points: [], negative_points: [] }
  };
}

function forwardTrackingStdout(event, caseId, text, state) {
  state.stdout += text;
  state.stream.write(text);
  state.buffer += text;
  const lines = state.buffer.split(/\r?\n/);
  state.buffer = lines.pop() || "";
  for (const line of lines) {
    const trimmed = line.trim();
    if (!trimmed) continue;
    try {
      const payload = JSON.parse(trimmed);
      if (payload && payload.stage) {
        event.sender.send("tracking:progress", { caseId, ...payload });
        continue;
      }
    } catch {
      // Non-JSON output is a normal human log line.
    }
    event.sender.send("tracking:log", { caseId, text: `${line}\n` });
  }
}

function batchManifestPath(batchId) {
  return labPath("data", "batches", batchId, "batch_manifest.json");
}

function caseById(batchId, caseId) {
  if (!batchId) return null;
  const manifestPath = batchManifestPath(batchId);
  if (!fs.existsSync(manifestPath)) return null;
  const manifest = readJson(manifestPath);
  return manifest.cases.find((item) => item.case_id === caseId);
}

function interactiveRequestPath(caseId) {
  return labPath("data", "cases", caseId, "interactive_track_request.json");
}

function validationPath(caseId) {
  return labPath("data", "cases", caseId, "track_validation.json");
}

function interactiveOutput(caseId, name) {
  return labPath("data", "outputs", caseId, "interactive_tracking", name);
}

function outputPath(caseId, ...parts) {
  return labPath("data", "outputs", caseId, ...parts);
}

function reportPath(caseId, ...parts) {
  return labPath("data", "reports", caseId, ...parts);
}

function caseDataPath(caseId, ...parts) {
  return labPath("data", "cases", caseId, ...parts);
}

function safeReadJson(filePath, fallback = null) {
  try {
    return fs.existsSync(filePath) ? readJson(filePath) : fallback;
  } catch (error) {
    return { error: error.message, path: filePath };
  }
}

function asset(filePath, playable = false) {
  const info = fileInfo(filePath);
  return {
    ...info,
    url: info.exists && info.size > 0 ? fileUrl(filePath, playable) : null
  };
}

function countFiles(dirPath, ext = null) {
  if (!fs.existsSync(dirPath)) return 0;
  return fs.readdirSync(dirPath).filter((name) => !ext || name.toLowerCase().endsWith(ext)).length;
}

function loadCaseState(batchId, caseId) {
  const item = caseById(batchId, caseId);
  if (!item) throw new Error(`Case not found: ${caseId}`);
  const caseStatusPath = caseDataPath(caseId, "case_status.json");
  const validationFile = validationPath(caseId);
  const requestPath = interactiveRequestPath(caseId);
  const trackFile = interactiveOutput(caseId, "track.json");
  const trackingQualityPath = interactiveOutput(caseId, "tracking_quality.json");
  const trackFolder = path.dirname(trackFile);
  const trackBasedFolder = outputPath(caseId, "track_based_analysis");
  const trackBasedMetricsPath = path.join(trackBasedFolder, "track_based_metrics.json");
  const trackBasedReport = reportPath(caseId, "track_based_analysis_report.md");
  const trackBasedManifest = path.join(trackBasedFolder, "track_based_manifest.md");
  const motionFolder = outputPath(caseId, "motion_analysis");
  const motionMetricsPath = path.join(motionFolder, "motion_metrics.json");
  const motionTimeseriesPath = path.join(motionFolder, "motion_timeseries.csv");
  const motionReportPath = path.join(motionFolder, "motion_analysis_report.md");
  const motionManifestPath = path.join(motionFolder, "motion_manifest.md");
  const spectralFolder = outputPath(caseId, "spectral_analysis");
  const spectralMetricsPath = path.join(spectralFolder, "spectral_metrics.json");
  const spectralTimeseriesPath = path.join(spectralFolder, "spectral_timeseries.csv");
  const spectralReportPath = path.join(spectralFolder, "spectral_analysis_report.md");
  const spectralManifestPath = path.join(spectralFolder, "spectral_manifest.md");
  const thermalFolder = outputPath(caseId, "thermal_analysis");
  const thermalMetricsPath = path.join(thermalFolder, "thermal_metrics.json");
  const thermalTimeseriesPath = path.join(thermalFolder, "thermal_timeseries.csv");
  const thermalReportPath = reportPath(caseId, "thermal_analysis_report.md");
  const thermalManifestPath = path.join(thermalFolder, "thermal_manifest.md");
  const controlsFolder = outputPath(caseId, "controls_analysis");
  const controlsMetricsPath = path.join(controlsFolder, "controls_metrics.json");
  const controlsTimeseriesPath = path.join(controlsFolder, "controls_timeseries.csv");
  const controlsRejectionPath = path.join(controlsFolder, "controls_rejection_report.csv");
  const controlsReportPath = reportPath(caseId, "controls_report.md");
  const controlsManifestPath = path.join(controlsFolder, "clean_controls_manifest.md");
  const pcaFolder = outputPath(caseId, "pca_analysis");
  const pcaMetricsPath = path.join(pcaFolder, "pca_metrics.json");
  const pcaSamplesPath = path.join(pcaFolder, "pca_samples.csv");
  const pcaExplainedPath = path.join(pcaFolder, "pca_explained_variance.csv");
  const pcaReportPath = reportPath(caseId, "pca_analysis_report.md");
  const pcaManifestPath = path.join(pcaFolder, "pca_manifest.md");
  const autoencoderFolder = outputPath(caseId, "autoencoder_analysis");
  const autoencoderMetricsPath = path.join(autoencoderFolder, "autoencoder_metrics.json");
  const autoencoderTimeseriesPath = path.join(autoencoderFolder, "autoencoder_timeseries.csv");
  const autoencoderCurvePath = path.join(autoencoderFolder, "autoencoder_training_curve.csv");
  const autoencoderConfigPath = path.join(autoencoderFolder, "autoencoder_config.json");
  const autoencoderReportPath = reportPath(caseId, "autoencoder_analysis_report.md");
  const autoencoderManifestPath = path.join(autoencoderFolder, "autoencoder_manifest.md");
  const unifiedFolder = reportPath(caseId, "unified_report");
  const unifiedReportPath = path.join(unifiedFolder, "unified_case_report.md");
  const unifiedSummaryPath = path.join(unifiedFolder, "unified_case_summary.json");
  const unifiedMetricsCardPath = path.join(unifiedFolder, "unified_metrics_card.json");
  const unifiedScorecardPath = path.join(unifiedFolder, "unified_case_scorecard.png");
  const unifiedManifestPath = path.join(unifiedFolder, "unified_report_manifest.md");
  const redditTemplateFolder = reportPath(caseId, "reddit_template");
  const redditTemplateEnPath = path.join(redditTemplateFolder, "reddit_post_template_en.md");
  const redditTemplateEsPath = path.join(redditTemplateFolder, "reddit_post_template_es.md");
  const redditTemplateShortEnPath = path.join(redditTemplateFolder, "reddit_post_short_en.md");
  const redditTemplateTitlesEnPath = path.join(redditTemplateFolder, "reddit_title_options_en.txt");
  const redditTemplateDataPath = path.join(redditTemplateFolder, "reddit_post_data.json");
  const redditTemplateManifestPath = path.join(redditTemplateFolder, "reddit_template_manifest.md");
  const videoExportFolder = reportPath(caseId, "video_exports");
  const videoExportManifestPath = path.join(videoExportFolder, "video_export_manifest.json");
  const videoExport16x9Path = path.join(videoExportFolder, `${caseId}_16x9.mp4`);
  const videoExport9x16Path = path.join(videoExportFolder, `${caseId}_9x16.mp4`);
  const videoExport1x1Path = path.join(videoExportFolder, `${caseId}_1x1.mp4`);
  const srvFolder = outputPath(caseId, "srv_analysis");
  const srvMetricsPath = path.join(srvFolder, "srv_metrics.json");
  const srvTimeseriesPath = path.join(srvFolder, "srv_timeseries.csv");
  const srvReportPath = reportPath(caseId, "srv_analysis_report.md");
  const srvManifestPath = path.join(srvFolder, "srv_manifest.md");
  const srvCoreFolder = path.join(srvFolder, "object_core");
  const srvCoreMetricsPath = path.join(srvCoreFolder, "srv_core_metrics.json");
  const srvCoreRoisPath = path.join(srvCoreFolder, "object_core_rois.csv");
  const srvCoreRoisJsonPath = path.join(srvCoreFolder, "object_core_rois.json");
  const srvCoreManifestPath = path.join(srvCoreFolder, "srv_core_manifest.md");
  const trackingReport = interactiveOutput(caseId, "tracking_report.md");
  const validation = safeReadJson(validationFile);
  const track = safeReadJson(trackFile);
  const trackingQuality = safeReadJson(trackingQualityPath);
  const trackBasedMetrics = safeReadJson(trackBasedMetricsPath);
  const motionMetrics = safeReadJson(motionMetricsPath);
  const spectralMetrics = safeReadJson(spectralMetricsPath);
  const thermalMetrics = safeReadJson(thermalMetricsPath);
  const controlsMetrics = safeReadJson(controlsMetricsPath);
  const pcaMetrics = safeReadJson(pcaMetricsPath);
  const autoencoderMetrics = safeReadJson(autoencoderMetricsPath);
  const unifiedSummary = safeReadJson(unifiedSummaryPath);
  const unifiedMetricsCard = safeReadJson(unifiedMetricsCardPath);
  const redditTemplateData = safeReadJson(redditTemplateDataPath);
  const videoExportManifest = safeReadJson(videoExportManifestPath);
  const srvMetrics = safeReadJson(srvMetricsPath);
  const srvCoreMetrics = safeReadJson(srvCoreMetricsPath);
  const caseStatus = safeReadJson(caseStatusPath);
  const trackGenerated = fs.existsSync(trackFile);
  const trackingFailed = trackingQuality?.recommendation === "tracking_failed";
  const trackValidated = Boolean(!trackingFailed && validation?.track_validated && validation?.track_is_correct && validation?.object_is_real_target);
  const trackBasedReady = fs.existsSync(trackBasedMetricsPath) && fs.existsSync(path.join(trackBasedFolder, "dynamic_rois.json"));
  const dynamicRoisReady = fs.existsSync(path.join(trackBasedFolder, "dynamic_rois.csv"));
  const motionReady = fs.existsSync(motionMetricsPath) && fs.existsSync(motionTimeseriesPath);
  const spectralReady = fs.existsSync(spectralMetricsPath) && fs.existsSync(spectralTimeseriesPath);
  const thermalReady = fs.existsSync(thermalMetricsPath) && fs.existsSync(thermalTimeseriesPath);
  const controlsReady = fs.existsSync(controlsMetricsPath) && fs.existsSync(controlsTimeseriesPath);
  const cleanControlsReady = controlsReady && controlsMetrics?.controls_version === "Controls v0.2 clean masked";
  const pcaReady = fs.existsSync(pcaMetricsPath) && fs.existsSync(pcaSamplesPath);
  const autoencoderReady = fs.existsSync(autoencoderMetricsPath) && fs.existsSync(autoencoderTimeseriesPath);
  const unifiedReady = fs.existsSync(unifiedReportPath) && fs.existsSync(unifiedSummaryPath);
  const redditTemplateReady = fs.existsSync(redditTemplateEnPath) && fs.existsSync(redditTemplateDataPath);
  const videoExportReady = fs.existsSync(videoExportManifestPath) && (
    fs.existsSync(videoExport16x9Path) || fs.existsSync(videoExport9x16Path) || fs.existsSync(videoExport1x1Path)
  );
  const srvReady = fs.existsSync(srvMetricsPath) && fs.existsSync(srvTimeseriesPath);
  const srvCoreReady = fs.existsSync(srvCoreMetricsPath) && fs.existsSync(srvCoreRoisPath);
  let reviewStatus = "tracking_required";
  if (trackGenerated) reviewStatus = "tracking_unvalidated";
  if (trackValidated) reviewStatus = "tracking_human_validated";
  return {
    case: item,
    case_status: caseStatus,
    request: safeReadJson(requestPath),
    validation,
    review_status: reviewStatus,
    gates: {
      tracking_generated: trackGenerated,
      tracking_validated: trackValidated,
      track_based_analysis_ready: trackBasedReady,
      dynamic_rois_ready: dynamicRoisReady,
      motion_analysis_ready: motionReady,
      spectral_analysis_ready: spectralReady,
      thermal_analysis_ready: thermalReady,
      controls_analysis_ready: controlsReady,
      clean_controls_v02_ready: cleanControlsReady,
      pca_analysis_ready: pcaReady,
      autoencoder_analysis_ready: autoencoderReady,
      unified_report_ready: unifiedReady,
      reddit_template_ready: redditTemplateReady,
      evidence_video_export_ready: videoExportReady,
      srv_analysis_ready: srvReady,
      srv_object_core_ready: srvCoreReady
    },
    paths: {
      video: item.video_path,
      request: requestPath,
      validation: validationFile,
      case_status: caseStatusPath,
      tracking_folder: trackFolder,
      track_json: trackFile,
      tracking_quality: trackingQualityPath,
      tracking_report: trackingReport,
      overlay: interactiveOutput(caseId, "track_overlay_preview.mp4"),
      web_overlay: interactiveOutput(caseId, "track_overlay_preview_web.mp4"),
      temporal_contact_sheet: interactiveOutput(caseId, "temporal_contact_sheet.png"),
      track_based_folder: trackBasedFolder,
      dynamic_rois_csv: path.join(trackBasedFolder, "dynamic_rois.csv"),
      dynamic_rois_json: path.join(trackBasedFolder, "dynamic_rois.json"),
      track_based_report: trackBasedReport,
      track_based_manifest: trackBasedManifest,
      motion_folder: motionFolder,
      motion_metrics: motionMetricsPath,
      motion_timeseries: motionTimeseriesPath,
      motion_report: motionReportPath,
      motion_manifest: motionManifestPath,
      spectral_folder: spectralFolder,
      spectral_metrics: spectralMetricsPath,
      spectral_timeseries: spectralTimeseriesPath,
      spectral_report: spectralReportPath,
      spectral_manifest: spectralManifestPath,
      thermal_folder: thermalFolder,
      thermal_metrics: thermalMetricsPath,
      thermal_timeseries: thermalTimeseriesPath,
      thermal_report: thermalReportPath,
      thermal_manifest: thermalManifestPath,
      controls_folder: controlsFolder,
      controls_metrics: controlsMetricsPath,
      controls_timeseries: controlsTimeseriesPath,
      controls_rejection_report: controlsRejectionPath,
      controls_report: controlsReportPath,
      controls_manifest: controlsManifestPath,
      pca_folder: pcaFolder,
      pca_metrics: pcaMetricsPath,
      pca_samples: pcaSamplesPath,
      pca_explained_variance: pcaExplainedPath,
      pca_report: pcaReportPath,
      pca_manifest: pcaManifestPath,
      autoencoder_folder: autoencoderFolder,
      autoencoder_metrics: autoencoderMetricsPath,
      autoencoder_timeseries: autoencoderTimeseriesPath,
      autoencoder_training_curve: autoencoderCurvePath,
      autoencoder_config: autoencoderConfigPath,
      autoencoder_report: autoencoderReportPath,
      autoencoder_manifest: autoencoderManifestPath,
      unified_report_folder: unifiedFolder,
      unified_report: unifiedReportPath,
      unified_summary: unifiedSummaryPath,
      unified_metrics_card: unifiedMetricsCardPath,
      unified_scorecard: unifiedScorecardPath,
      unified_manifest: unifiedManifestPath,
      reddit_template_folder: redditTemplateFolder,
      reddit_template_en: redditTemplateEnPath,
      reddit_template_es: redditTemplateEsPath,
      reddit_template_short_en: redditTemplateShortEnPath,
      reddit_template_titles_en: redditTemplateTitlesEnPath,
      reddit_template_data: redditTemplateDataPath,
      reddit_template_manifest: redditTemplateManifestPath,
      video_export_folder: videoExportFolder,
      video_export_manifest: videoExportManifestPath,
      video_export_16x9: videoExport16x9Path,
      video_export_9x16: videoExport9x16Path,
      video_export_1x1: videoExport1x1Path,
      srv_folder: srvFolder,
      srv_metrics: srvMetricsPath,
      srv_timeseries: srvTimeseriesPath,
      srv_report: srvReportPath,
      srv_manifest: srvManifestPath,
      srv_core_folder: srvCoreFolder,
      srv_core_metrics: srvCoreMetricsPath,
      srv_core_rois_csv: srvCoreRoisPath,
      srv_core_rois_json: srvCoreRoisJsonPath,
      srv_core_manifest: srvCoreManifestPath,
      content_folder: outputPath(caseId, "entity_visuals"),
      renders_folder: labPath("data", "renders", caseId)
    },
    assets: {
      video: asset(item.video_path, true),
      temporal_contact_sheet: asset(interactiveOutput(caseId, "temporal_contact_sheet.png")),
      overlay: asset(fs.existsSync(interactiveOutput(caseId, "track_overlay_preview_web.mp4")) ? interactiveOutput(caseId, "track_overlay_preview_web.mp4") : interactiveOutput(caseId, "track_overlay_preview.mp4"), true),
      quick_panel: asset(outputPath(caseId, "batch_quick", "quick_panel.png")),
      track_based_contact_sheet: asset(path.join(trackBasedFolder, "track_based_contact_sheet.png")),
      track_based_motion_panel: asset(path.join(trackBasedFolder, "track_based_motion_panel.png")),
      track_based_trajectory_panel: asset(path.join(trackBasedFolder, "track_based_trajectory_panel.png")),
      track_based_pca_panel: asset(path.join(trackBasedFolder, "track_based_pca_panel.png")),
      track_based_autoencoder_panel: asset(path.join(trackBasedFolder, "track_based_autoencoder_panel.png")),
      motion_trajectory_panel: asset(path.join(motionFolder, "motion_trajectory_panel.png")),
      motion_velocity_panel: asset(path.join(motionFolder, "motion_velocity_panel.png")),
      motion_optical_flow_panel: asset(path.join(motionFolder, "motion_optical_flow_panel.png")),
      motion_stability_panel: asset(path.join(motionFolder, "motion_stability_panel.png")),
      spectral_luminance_panel: asset(path.join(spectralFolder, "spectral_luminance_panel.png")),
      spectral_color_panel: asset(path.join(spectralFolder, "spectral_color_panel.png")),
      spectral_fft_panel: asset(path.join(spectralFolder, "spectral_fft_panel.png")),
      spectral_spatial_frequency_panel: asset(path.join(spectralFolder, "spectral_spatial_frequency_panel.png")),
      spectral_contact_sheet: asset(path.join(spectralFolder, "spectral_contact_sheet.png")),
      thermal_intensity_panel: asset(path.join(thermalFolder, "thermal_intensity_panel.png")),
      thermal_contrast_panel: asset(path.join(thermalFolder, "thermal_contrast_panel.png")),
      thermal_roi_examples_panel: asset(path.join(thermalFolder, "thermal_roi_examples_panel.png")),
      thermal_delta_panel: asset(path.join(thermalFolder, "thermal_delta_panel.png")),
      thermal_contact_sheet: asset(path.join(thermalFolder, "thermal_contact_sheet.png")),
      thermal_roi_sequence: asset(path.join(thermalFolder, "thermal_roi_sequence.mp4"), true),
      controls_summary_panel: asset(path.join(controlsFolder, "controls_summary_panel.png")),
      controls_luminance_panel: asset(path.join(controlsFolder, "controls_luminance_panel.png")),
      controls_thermal_panel: asset(path.join(controlsFolder, "controls_thermal_panel.png")),
      controls_spectral_panel: asset(path.join(controlsFolder, "controls_spectral_panel.png")),
      controls_motion_panel: asset(path.join(controlsFolder, "controls_motion_panel.png")),
      controls_contact_sheet: asset(path.join(controlsFolder, "controls_contact_sheet.png")),
      controls_artifact_mask_panel: asset(path.join(controlsFolder, "artifact_mask_debug_panel.png")),
      controls_quality_panel: asset(path.join(controlsFolder, "controls_quality_panel.png")),
      pca_scatter_panel: asset(path.join(pcaFolder, "pca_scatter_panel.png")),
      pca_explained_variance_panel: asset(path.join(pcaFolder, "pca_explained_variance_panel.png")),
      pca_class_distance_panel: asset(path.join(pcaFolder, "pca_class_distance_panel.png")),
      pca_reconstruction_panel: asset(path.join(pcaFolder, "pca_reconstruction_panel.png")),
      pca_contact_sheet_panel: asset(path.join(pcaFolder, "pca_contact_sheet_panel.png")),
      autoencoder_error_distribution_panel: asset(path.join(autoencoderFolder, "autoencoder_error_distribution_panel.png")),
      autoencoder_timeseries_panel: asset(path.join(autoencoderFolder, "autoencoder_timeseries_panel.png")),
      autoencoder_reconstruction_examples_panel: asset(path.join(autoencoderFolder, "autoencoder_reconstruction_examples_panel.png")),
      autoencoder_latent_panel: asset(path.join(autoencoderFolder, "autoencoder_latent_panel.png")),
      autoencoder_summary_panel: asset(path.join(autoencoderFolder, "autoencoder_summary_panel.png")),
      unified_scorecard: asset(unifiedScorecardPath),
      srv_contact_sheet_raw: asset(path.join(srvFolder, "srv_contact_sheet_raw.png")),
      srv_contact_sheet_enhanced: asset(path.join(srvFolder, "srv_contact_sheet_enhanced.png")),
      srv_stabilized_sequence_panel: asset(path.join(srvFolder, "srv_stabilized_sequence_panel.png")),
      srv_comparison_panel: asset(path.join(srvFolder, "srv_comparison_panel.png")),
      srv_quality_panel: asset(path.join(srvFolder, "srv_quality_panel.png")),
      srv_stack_average: asset(path.join(srvFolder, "srv_stack_average.png")),
      srv_stack_median: asset(path.join(srvFolder, "srv_stack_median.png")),
      srv_stack_best_sharpness: asset(path.join(srvFolder, "srv_stack_best_sharpness.png")),
      srv_stabilized_video: asset(path.join(srvFolder, "srv_stabilized_crop_sequence.mp4"), true),
      srv_core_contact_sheet_raw: asset(path.join(srvCoreFolder, "srv_core_contact_sheet_raw.png")),
      srv_core_contact_sheet_enhanced: asset(path.join(srvCoreFolder, "srv_core_contact_sheet_enhanced.png")),
      srv_core_stabilized_sequence_panel: asset(path.join(srvCoreFolder, "srv_core_stabilized_sequence_panel.png")),
      srv_core_comparison_panel: asset(path.join(srvCoreFolder, "srv_core_comparison_panel.png")),
      srv_core_quality_panel: asset(path.join(srvCoreFolder, "srv_core_quality_panel.png")),
      srv_core_stack_average: asset(path.join(srvCoreFolder, "srv_core_stack_average.png")),
      srv_core_stack_median: asset(path.join(srvCoreFolder, "srv_core_stack_median.png")),
      srv_core_stack_best_sharpness: asset(path.join(srvCoreFolder, "srv_core_stack_best_sharpness.png")),
      srv_core_stabilized_video: asset(path.join(srvCoreFolder, "srv_core_stabilized_crop_sequence.mp4"), true),
      tracking_report: asset(trackingReport),
      track_based_report: asset(trackBasedReport),
      track_based_manifest: asset(trackBasedManifest),
      motion_report: asset(motionReportPath),
      motion_manifest: asset(motionManifestPath),
      spectral_report: asset(spectralReportPath),
      spectral_manifest: asset(spectralManifestPath),
      thermal_report: asset(thermalReportPath),
      thermal_manifest: asset(thermalManifestPath),
      controls_report: asset(controlsReportPath),
      controls_manifest: asset(controlsManifestPath),
      pca_report: asset(pcaReportPath),
      pca_manifest: asset(pcaManifestPath),
      autoencoder_report: asset(autoencoderReportPath),
      autoencoder_manifest: asset(autoencoderManifestPath),
      unified_report: asset(unifiedReportPath),
      unified_manifest: asset(unifiedManifestPath),
      reddit_template_en: asset(redditTemplateEnPath),
      reddit_template_manifest: asset(redditTemplateManifestPath),
      video_export_manifest: asset(videoExportManifestPath),
      video_export_16x9: asset(videoExport16x9Path, true),
      video_export_9x16: asset(videoExport9x16Path, true),
      video_export_1x1: asset(videoExport1x1Path, true),
      srv_report: asset(srvReportPath),
      srv_manifest: asset(srvManifestPath),
      srv_core_manifest: asset(srvCoreManifestPath)
    },
    metrics: {
      track_summary: track?.summary || null,
      tracking_quality: trackingQuality || track?.tracking_quality || track?.summary || null,
      track_based: trackBasedMetrics || null,
      motion: motionMetrics || null,
      spectral: spectralMetrics || null,
      thermal: thermalMetrics || null,
      controls: controlsMetrics || null,
      pca: pcaMetrics || null,
      autoencoder: autoencoderMetrics || null,
      unified: unifiedSummary || null,
      unified_card: unifiedMetricsCard || null,
      reddit_template: redditTemplateData || null,
      video_export: videoExportManifest || null,
      srv: srvMetrics || null,
      srv_core: srvCoreMetrics || null,
      crops: countFiles(path.join(trackBasedFolder, "crops"), ".png"),
      normalized_crops: countFiles(path.join(trackBasedFolder, "crops_normalized_64"), ".png")
    }
  };
}

function createWindow() {
  const win = new BrowserWindow({
    width: 1500,
    height: 980,
    minWidth: 1180,
    minHeight: 760,
    backgroundColor: "#101416",
    webPreferences: {
      preload: path.join(__dirname, "preload.js"),
      contextIsolation: true,
      nodeIntegration: false
    }
  });
  win.loadFile(path.join(APP_ROOT, "src", "index.html"));
}

app.whenReady().then(() => {
  createWindow();
  app.on("activate", () => {
    if (BrowserWindow.getAllWindows().length === 0) createWindow();
  });
});

app.on("window-all-closed", () => {
  if (process.platform !== "darwin") app.quit();
});

ipcMain.handle("config:get", async () => loadConfig());

ipcMain.handle("batches:list", async () => {
  const config = loadConfig();
  const root = path.join(config.lab_root, "data", "batches");
  if (!fs.existsSync(root)) return [];
  return fs.readdirSync(root, { withFileTypes: true })
    .filter((entry) => entry.isDirectory() && fs.existsSync(path.join(root, entry.name, "batch_manifest.json")))
    .map((entry) => entry.name);
});

ipcMain.handle("batch:get", async (_event, batchId) => {
  if (!batchId) {
    return {
      batch_id: null,
      manifest_path: null,
      missing: true,
      message: "No batch loaded. Create or select a batch.",
      cases: []
    };
  }
  const manifestPath = batchManifestPath(batchId);
  if (!fs.existsSync(manifestPath)) {
    return {
      batch_id: batchId,
      manifest_path: manifestPath,
      missing: true,
      message: "No batch loaded. Create or select a batch.",
      cases: []
    };
  }
  const manifest = readJson(manifestPath);
  const cases = manifest.cases.map((item) => {
    const request = interactiveRequestPath(item.case_id);
    const track = interactiveOutput(item.case_id, "track.json");
    const validation = validationPath(item.case_id);
    let reviewStatus = "tracking_required";
    if (fs.existsSync(track)) reviewStatus = "tracking_unvalidated";
    if (fs.existsSync(validation)) {
      const data = readJson(validation);
      if (data.track_validated && data.track_is_correct && data.object_is_real_target) {
        reviewStatus = "tracking_human_validated";
      }
    }
    return {
      case_id: item.case_id,
      original_filename: item.original_filename,
      priority: item.priority || item.quick_priority || "",
      tracking_status: item.tracking_status || "tracking_not_run",
      review_status: reviewStatus,
      video_path: item.video_path,
      request_path: request,
      request_exists: fs.existsSync(request),
      track_path: track,
      track_exists: fs.existsSync(track),
      validation_path: validation,
      validation_exists: fs.existsSync(validation)
    };
  });
  return { batch_id: batchId, manifest_path: manifestPath, cases };
});

ipcMain.handle("case:importVideo", async (_event, payload = {}) => {
  const config = loadConfig();
  const allowedExtensions = [".mp4", ".mov", ".avi", ".mkv", ".webm"];
  let videoPaths = Array.isArray(payload.videoPaths)
    ? payload.videoPaths
    : (payload.videoPath ? [payload.videoPath] : []);
  if (!videoPaths.length) {
    const result = await dialog.showOpenDialog({
      title: "Import Video",
      properties: ["openFile", "multiSelections"],
      filters: [
        { name: "Video files", extensions: ["mp4", "mov", "avi", "mkv", "webm"] }
      ]
    });
    if (result.canceled || !result.filePaths.length) {
      return { ok: false, canceled: true, message: "Import cancelled." };
    }
    videoPaths = result.filePaths;
  }
  videoPaths = [...new Set(videoPaths.map((item) => path.resolve(item)))];
  const unsupported = videoPaths.filter((videoPath) => !allowedExtensions.includes(path.extname(videoPath).toLowerCase()));
  if (unsupported.length) {
    return { ok: false, error: `Unsupported file type: ${unsupported[0]}` };
  }
  const runtimeRoot = config.packaged_backend ? config.writable_runtime_root : config.lab_root;
  const args = ["--runtime-root", runtimeRoot];
  for (const videoPath of videoPaths) {
    args.push("--video", videoPath);
  }
  const imported = await runBackendJson(config, "import-video", args);
  if (!imported.ok) {
    return { ...imported, error: imported.error || "Import failed." };
  }
  return imported;
});

ipcMain.handle("case:get", async (_event, { batchId, caseId }) => {
  const item = caseById(batchId, caseId);
  if (!item) throw new Error(`Case not found: ${caseId}`);
  const requestPath = interactiveRequestPath(caseId);
  const validationFile = validationPath(caseId);
  const contactSheet = interactiveOutput(caseId, "temporal_contact_sheet.png");
  const overlay = interactiveOutput(caseId, "track_overlay_preview.mp4");
  const webOverlay = interactiveOutput(caseId, "track_overlay_preview_web.mp4");
  const quickPanel = labPath("data", "outputs", caseId, "batch_quick", "quick_panel.png");
  const request = fs.existsSync(requestPath) ? readJson(requestPath) : null;
  const validation = fs.existsSync(validationFile) ? readJson(validationFile) : null;
  return {
    case: item,
    paths: {
      video: item.video_path,
      request: requestPath,
      validation: validationFile,
      contact_sheet: contactSheet,
      quick_panel: quickPanel,
      overlay,
      web_overlay: webOverlay,
      track_json: interactiveOutput(caseId, "track.json"),
      tracking_folder: path.dirname(overlay)
    },
    urls: {
      video: fileUrl(item.video_path),
      contact_sheet: fileUrl(contactSheet),
      quick_panel: fileUrl(quickPanel),
      overlay: fileUrl(fs.existsSync(webOverlay) ? webOverlay : overlay, true)
    },
    request,
    validation
  };
});

ipcMain.handle("case:state", async (_event, { batchId, caseId }) => loadCaseState(batchId, caseId));

ipcMain.handle("annotation:save", async (_event, payload) => {
  const { caseId, frame, time, box } = payload;
  const requestPath = interactiveRequestPath(caseId);
  let request;
  if (fs.existsSync(requestPath)) {
    request = readJson(requestPath);
  } else {
    request = emptyInteractiveRequest(caseId);
  }
  const cx = box.x + box.w / 2;
  const cy = box.y + box.h / 2;
  request.first_object_frame = Number(frame);
  request.first_object_time_seconds = Number(time || 0);
  request.do_not_track_before_first_frame = true;
  request.object_prompt = {
    type: "box_or_points",
    box: { x: Number(box.x), y: Number(box.y), w: Number(box.w), h: Number(box.h) },
    positive_points: [{ x: Number(cx), y: Number(cy), frame: Number(frame), time_seconds: Number(time || 0) }],
    negative_points: []
  };
  request.updated_at = new Date().toISOString();
  await writeJson(requestPath, request);
  return { request_path: requestPath, request };
});

ipcMain.handle("annotation:setStartFrame", async (_event, payload) => {
  const { caseId, frame, time } = payload;
  const requestPath = interactiveRequestPath(caseId);
  const request = fs.existsSync(requestPath) ? readJson(requestPath) : emptyInteractiveRequest(caseId);
  request.first_object_frame = Number(frame);
  request.first_object_time_seconds = Number(time || 0);
  request.do_not_track_before_first_frame = true;
  request.updated_at = new Date().toISOString();
  await writeJson(requestPath, request);
  return { request_path: requestPath, request };
});

ipcMain.handle("tracking:run", async (event, { caseId, backend }) => {
  const config = loadConfig();
  await fsp.mkdir(logDir(), { recursive: true });
  const logPath = path.join(logDir(), `${caseId}_tracking_${Date.now()}.log`);
  const args = ["-m", "scripts.run_interactive_track", "--case-id", caseId, "--backend", backend || "opencv", "--progress-jsonl"];
  const commandText = `"${config.python_exe}" ${args.join(" ")}`;
  event.sender.send("tracking:log", { caseId, text: `COMMAND ${commandText}\n` });
  const child = spawn(config.python_exe, args, { cwd: config.lab_root, windowsHide: true });
  const stream = fs.createWriteStream(logPath, { flags: "a" });
  const progressState = { stdout: "", stream, buffer: "" };
  let stderr = "";
  child.stdout.on("data", (data) => {
    const text = data.toString();
    forwardTrackingStdout(event, caseId, text, progressState);
  });
  child.stderr.on("data", (data) => {
    const text = data.toString();
    stderr += text;
    stream.write(text);
    event.sender.send("tracking:log", { caseId, text });
  });
  const result = await new Promise((resolve) => {
    child.on("close", (code) => resolve({ code }));
  });
  if (progressState.buffer.trim()) {
    forwardTrackingStdout(event, caseId, "\n", progressState);
  }
  stream.end();
  const trackJson = interactiveOutput(caseId, "track.json");
  const trackingQuality = interactiveOutput(caseId, "tracking_quality.json");
  const overlayWeb = interactiveOutput(caseId, "track_overlay_preview_web.mp4");
  const overlayRaw = interactiveOutput(caseId, "track_overlay_preview.mp4");
  const preferred = fs.existsSync(overlayWeb) ? overlayWeb : overlayRaw;
  const previewInfo = fileInfo(preferred);
  const trackInfo = fileInfo(trackJson);
  const previewUrl = previewInfo.exists && previewInfo.size > 0 ? fileUrl(preferred, true) : null;
  return {
    ...result,
    command: commandText,
    stdout: progressState.stdout,
    stderr,
    log_path: logPath,
    track_json: trackInfo,
    tracking_quality: fileInfo(trackingQuality),
    overlay: fileInfo(overlayRaw),
    web_overlay: fileInfo(overlayWeb),
    preview: previewInfo,
    preview_url: previewUrl
  };
});

ipcMain.handle("trackBased:rebuild", async (event, { caseId }) => {
  const config = loadConfig();
  await fsp.mkdir(logDir(), { recursive: true });
  const logPath = path.join(logDir(), `${caseId}_rebuild_from_track_${Date.now()}.log`);
  const args = ["-m", "scripts.rebuild_from_track", "--case-id", caseId];
  const commandText = `"${config.python_exe}" ${args.join(" ")}`;
  event.sender.send("tracking:log", { caseId, text: `COMMAND ${commandText}\n` });
  const child = spawn(config.python_exe, args, { cwd: config.lab_root, windowsHide: true });
  const stream = fs.createWriteStream(logPath, { flags: "a" });
  let stdout = "";
  let stderr = "";
  child.stdout.on("data", (data) => {
    const text = data.toString();
    stdout += text;
    stream.write(text);
    event.sender.send("tracking:log", { caseId, text });
  });
  child.stderr.on("data", (data) => {
    const text = data.toString();
    stderr += text;
    stream.write(text);
    event.sender.send("tracking:log", { caseId, text });
  });
  const result = await new Promise((resolve) => child.on("close", (code) => resolve({ code })));
  stream.end();
  return { ...result, command: commandText, stdout, stderr, log_path: logPath };
});

ipcMain.handle("motion:run", async (event, { caseId }) => {
  const config = loadConfig();
  await fsp.mkdir(logDir(), { recursive: true });
  const logPath = path.join(logDir(), `${caseId}_motion_analysis_${Date.now()}.log`);
  const args = ["-m", "scripts.run_track_motion_analysis", "--case-id", caseId];
  const commandText = `"${config.python_exe}" ${args.join(" ")}`;
  event.sender.send("tracking:log", { caseId, text: `COMMAND ${commandText}\nCWD ${config.lab_root}\n` });
  const child = spawn(config.python_exe, args, { cwd: config.lab_root, windowsHide: true });
  const stream = fs.createWriteStream(logPath, { flags: "a" });
  let stdout = "";
  let stderr = "";
  child.stdout.on("data", (data) => {
    const text = data.toString();
    stdout += text;
    stream.write(text);
    event.sender.send("tracking:log", { caseId, text });
  });
  child.stderr.on("data", (data) => {
    const text = data.toString();
    stderr += text;
    stream.write(text);
    event.sender.send("tracking:log", { caseId, text });
  });
  const result = await new Promise((resolve) => child.on("close", (code) => resolve({ code })));
  stream.end();
  const folder = outputPath(caseId, "motion_analysis");
  const outputs = {
    metrics: fileInfo(path.join(folder, "motion_metrics.json")),
    csv: fileInfo(path.join(folder, "motion_timeseries.csv")),
    report: fileInfo(path.join(folder, "motion_analysis_report.md")),
    trajectory: fileInfo(path.join(folder, "motion_trajectory_panel.png")),
    velocity: fileInfo(path.join(folder, "motion_velocity_panel.png")),
    optical_flow: fileInfo(path.join(folder, "motion_optical_flow_panel.png")),
    stability: fileInfo(path.join(folder, "motion_stability_panel.png"))
  };
  return { ...result, command: commandText, cwd: config.lab_root, stdout, stderr, log_path: logPath, outputs };
});

ipcMain.handle("spectral:run", async (event, { caseId }) => {
  const config = loadConfig();
  await fsp.mkdir(logDir(), { recursive: true });
  const logPath = path.join(logDir(), `${caseId}_spectral_analysis_${Date.now()}.log`);
  const args = ["-m", "scripts.run_track_spectral_analysis", "--case-id", caseId];
  const commandText = `"${config.python_exe}" ${args.join(" ")}`;
  event.sender.send("tracking:log", { caseId, text: `COMMAND ${commandText}\nCWD ${config.lab_root}\n` });
  const child = spawn(config.python_exe, args, { cwd: config.lab_root, windowsHide: true });
  const stream = fs.createWriteStream(logPath, { flags: "a" });
  let stdout = "";
  let stderr = "";
  child.stdout.on("data", (data) => {
    const text = data.toString();
    stdout += text;
    stream.write(text);
    event.sender.send("tracking:log", { caseId, text });
  });
  child.stderr.on("data", (data) => {
    const text = data.toString();
    stderr += text;
    stream.write(text);
    event.sender.send("tracking:log", { caseId, text });
  });
  const result = await new Promise((resolve) => child.on("close", (code) => resolve({ code })));
  stream.end();
  const folder = outputPath(caseId, "spectral_analysis");
  const outputs = {
    metrics: fileInfo(path.join(folder, "spectral_metrics.json")),
    csv: fileInfo(path.join(folder, "spectral_timeseries.csv")),
    report: fileInfo(path.join(folder, "spectral_analysis_report.md")),
    luminance: fileInfo(path.join(folder, "spectral_luminance_panel.png")),
    color: fileInfo(path.join(folder, "spectral_color_panel.png")),
    fft: fileInfo(path.join(folder, "spectral_fft_panel.png")),
    spatial_frequency: fileInfo(path.join(folder, "spectral_spatial_frequency_panel.png")),
    contact_sheet: fileInfo(path.join(folder, "spectral_contact_sheet.png"))
  };
  return { ...result, command: commandText, cwd: config.lab_root, stdout, stderr, log_path: logPath, outputs };
});

ipcMain.handle("thermal:run", async (event, { caseId }) => {
  const config = loadConfig();
  await fsp.mkdir(logDir(), { recursive: true });
  const logPath = path.join(logDir(), `${caseId}_thermal_analysis_${Date.now()}.log`);
  const args = ["-m", "scripts.run_track_thermal_analysis", "--case-id", caseId];
  const commandText = `"${config.python_exe}" ${args.join(" ")}`;
  event.sender.send("tracking:log", { caseId, text: `COMMAND ${commandText}\nCWD ${config.lab_root}\n` });
  const child = spawn(config.python_exe, args, { cwd: config.lab_root, windowsHide: true });
  const stream = fs.createWriteStream(logPath, { flags: "a" });
  let stdout = "";
  let stderr = "";
  child.stdout.on("data", (data) => {
    const text = data.toString();
    stdout += text;
    stream.write(text);
    event.sender.send("tracking:log", { caseId, text });
  });
  child.stderr.on("data", (data) => {
    const text = data.toString();
    stderr += text;
    stream.write(text);
    event.sender.send("tracking:log", { caseId, text });
  });
  const result = await new Promise((resolve) => child.on("close", (code) => resolve({ code })));
  stream.end();
  const folder = outputPath(caseId, "thermal_analysis");
  const outputs = {
    metrics: fileInfo(path.join(folder, "thermal_metrics.json")),
    csv: fileInfo(path.join(folder, "thermal_timeseries.csv")),
    report: fileInfo(reportPath(caseId, "thermal_analysis_report.md")),
    intensity: fileInfo(path.join(folder, "thermal_intensity_panel.png")),
    contrast: fileInfo(path.join(folder, "thermal_contrast_panel.png")),
    examples: fileInfo(path.join(folder, "thermal_roi_examples_panel.png")),
    delta: fileInfo(path.join(folder, "thermal_delta_panel.png")),
    contact_sheet: fileInfo(path.join(folder, "thermal_contact_sheet.png")),
    video: fileInfo(path.join(folder, "thermal_roi_sequence.mp4"))
  };
  return { ...result, command: commandText, cwd: config.lab_root, stdout, stderr, log_path: logPath, outputs };
});

ipcMain.handle("controls:run", async (event, { caseId }) => {
  const config = loadConfig();
  await fsp.mkdir(logDir(), { recursive: true });
  const logPath = path.join(logDir(), `${caseId}_controls_analysis_${Date.now()}.log`);
  const args = ["-m", "scripts.run_track_controls_analysis", "--case-id", caseId];
  const commandText = `"${config.python_exe}" ${args.join(" ")}`;
  event.sender.send("tracking:log", { caseId, text: `COMMAND ${commandText}\nCWD ${config.lab_root}\n` });
  const child = spawn(config.python_exe, args, { cwd: config.lab_root, windowsHide: true });
  const stream = fs.createWriteStream(logPath, { flags: "a" });
  let stdout = "";
  let stderr = "";
  child.stdout.on("data", (data) => {
    const text = data.toString();
    stdout += text;
    stream.write(text);
    event.sender.send("tracking:log", { caseId, text });
  });
  child.stderr.on("data", (data) => {
    const text = data.toString();
    stderr += text;
    stream.write(text);
    event.sender.send("tracking:log", { caseId, text });
  });
  const result = await new Promise((resolve) => child.on("close", (code) => resolve({ code })));
  stream.end();
  const folder = outputPath(caseId, "controls_analysis");
  const outputs = {
    metrics: fileInfo(path.join(folder, "controls_metrics.json")),
    csv: fileInfo(path.join(folder, "controls_timeseries.csv")),
    report: fileInfo(reportPath(caseId, "controls_report.md")),
    manifest: fileInfo(path.join(folder, "clean_controls_manifest.md")),
    rejection_report: fileInfo(path.join(folder, "controls_rejection_report.csv")),
    summary: fileInfo(path.join(folder, "controls_summary_panel.png")),
    luminance: fileInfo(path.join(folder, "controls_luminance_panel.png")),
    thermal: fileInfo(path.join(folder, "controls_thermal_panel.png")),
    spectral: fileInfo(path.join(folder, "controls_spectral_panel.png")),
    motion: fileInfo(path.join(folder, "controls_motion_panel.png")),
    contact_sheet: fileInfo(path.join(folder, "controls_contact_sheet.png")),
    artifact_mask: fileInfo(path.join(folder, "artifact_mask_debug_panel.png")),
    quality: fileInfo(path.join(folder, "controls_quality_panel.png"))
  };
  return { ...result, command: commandText, cwd: config.lab_root, stdout, stderr, log_path: logPath, outputs };
});

ipcMain.handle("pca:run", async (event, { caseId }) => {
  const config = loadConfig();
  await fsp.mkdir(logDir(), { recursive: true });
  const logPath = path.join(logDir(), `${caseId}_pca_analysis_${Date.now()}.log`);
  const cmd = backendCommand(config, "run-pca", ["--case-id", caseId]);
  event.sender.send("tracking:log", { caseId, text: `Starting PCA analysis for case ${caseId}\nCOMMAND ${cmd.text}\nCWD ${cmd.cwd}\n` });
  const stream = fs.createWriteStream(logPath, { flags: "a" });
  let stdout = "";
  let stderr = "";
  const inputs = [
    ["dynamic_rois.csv", outputPath(caseId, "track_based_analysis", "dynamic_rois.csv")],
    ["object crops", outputPath(caseId, "track_based_analysis", "crops")],
    ["controls_metrics.json", outputPath(caseId, "controls_analysis", "controls_metrics.json")],
    ["controls_timeseries.csv", outputPath(caseId, "controls_analysis", "controls_timeseries.csv")]
  ];
  for (const [label, inputPath] of inputs) {
    const line = `PCA input ${label}: ${fs.existsSync(inputPath) ? "found" : "missing"}\n`;
    stream.write(line);
    event.sender.send("tracking:log", { caseId, text: line });
  }
  event.sender.send("tracking:log", { caseId, text: "Running PCA backend...\n" });
  const child = spawn(cmd.exe, cmd.args, { cwd: cmd.cwd, windowsHide: true });
  child.stdout.on("data", (data) => {
    const text = data.toString();
    stdout += text;
    stream.write(text);
    event.sender.send("tracking:log", { caseId, text });
  });
  child.stderr.on("data", (data) => {
    const text = data.toString();
    stderr += text;
    stream.write(text);
    event.sender.send("tracking:log", { caseId, text });
  });
  const result = await new Promise((resolve) => {
    child.on("error", (error) => resolve({ code: -1, error: error.message }));
    child.on("close", (code) => resolve({ code }));
  });
  stream.end();
  const folder = outputPath(caseId, "pca_analysis");
  const outputs = {
    metrics: fileInfo(path.join(folder, "pca_metrics.json")),
    samples: fileInfo(path.join(folder, "pca_samples.csv")),
    explained_variance: fileInfo(path.join(folder, "pca_explained_variance.csv")),
    report: fileInfo(reportPath(caseId, "pca_analysis_report.md")),
    manifest: fileInfo(path.join(folder, "pca_manifest.md")),
    scatter: fileInfo(path.join(folder, "pca_scatter_panel.png")),
    explained_panel: fileInfo(path.join(folder, "pca_explained_variance_panel.png")),
    class_distance: fileInfo(path.join(folder, "pca_class_distance_panel.png")),
    reconstruction: fileInfo(path.join(folder, "pca_reconstruction_panel.png")),
    contact_sheet: fileInfo(path.join(folder, "pca_contact_sheet_panel.png"))
  };
  const ok = result.code === 0 && outputs.metrics.exists && outputs.samples.exists;
  if (!ok) {
    const reason = result.error || stderr || stdout || "PCA backend finished without required outputs.";
    event.sender.send("tracking:log", { caseId, text: `PCA failed: ${reason}\n` });
  } else {
    event.sender.send("tracking:log", { caseId, text: "PCA analysis complete\n" });
  }
  return { ...result, ok, command: cmd.text, cwd: cmd.cwd, stdout, stderr, log_path: logPath, outputs };
});

ipcMain.handle("autoencoder:run", async (event, { caseId, quick = false }) => {
  const config = loadConfig();
  await fsp.mkdir(logDir(), { recursive: true });
  const logPath = path.join(logDir(), `${caseId}_autoencoder_analysis_${quick ? "quick_" : ""}${Date.now()}.log`);
  const args = ["--case-id", caseId];
  if (quick) args.push("--quick");
  const cmd = backendCommand(config, "run-autoencoder", args);
  event.sender.send("tracking:log", { caseId, text: `Starting Autoencoder analysis for case ${caseId}\nCOMMAND ${cmd.text}\nCWD ${cmd.cwd}\n` });
  const stream = fs.createWriteStream(logPath, { flags: "a" });
  let stdout = "";
  let stderr = "";
  const inputs = [
    ["dynamic_rois.csv", outputPath(caseId, "track_based_analysis", "dynamic_rois.csv")],
    ["object crops", outputPath(caseId, "track_based_analysis", "crops")],
    ["controls_metrics.json", outputPath(caseId, "controls_analysis", "controls_metrics.json")],
    ["controls_timeseries.csv", outputPath(caseId, "controls_analysis", "controls_timeseries.csv")],
    ["pca_metrics.json", outputPath(caseId, "pca_analysis", "pca_metrics.json")]
  ];
  for (const [label, inputPath] of inputs) {
    const line = `Autoencoder input ${label}: ${fs.existsSync(inputPath) ? "found" : "missing"}\n`;
    stream.write(line);
    event.sender.send("tracking:log", { caseId, text: line });
  }
  event.sender.send("tracking:log", { caseId, text: "Running Autoencoder backend...\n" });
  const child = spawn(cmd.exe, cmd.args, { cwd: cmd.cwd, windowsHide: true });
  child.stdout.on("data", (data) => {
    const text = data.toString();
    stdout += text;
    stream.write(text);
    event.sender.send("tracking:log", { caseId, text });
  });
  child.stderr.on("data", (data) => {
    const text = data.toString();
    stderr += text;
    stream.write(text);
    event.sender.send("tracking:log", { caseId, text });
  });
  const result = await new Promise((resolve) => {
    child.on("error", (error) => resolve({ code: -1, error: error.message }));
    child.on("close", (code) => resolve({ code }));
  });
  stream.end();
  const folder = outputPath(caseId, "autoencoder_analysis");
  const outputs = {
    metrics: fileInfo(path.join(folder, "autoencoder_metrics.json")),
    timeseries: fileInfo(path.join(folder, "autoencoder_timeseries.csv")),
    training_curve: fileInfo(path.join(folder, "autoencoder_training_curve.csv")),
    config: fileInfo(path.join(folder, "autoencoder_config.json")),
    report: fileInfo(reportPath(caseId, "autoencoder_analysis_report.md")),
    manifest: fileInfo(path.join(folder, "autoencoder_manifest.md")),
    model: fileInfo(path.join(folder, "autoencoder_model.pt")),
    error_distribution: fileInfo(path.join(folder, "autoencoder_error_distribution_panel.png")),
    timeseries_panel: fileInfo(path.join(folder, "autoencoder_timeseries_panel.png")),
    reconstruction_examples: fileInfo(path.join(folder, "autoencoder_reconstruction_examples_panel.png")),
    latent: fileInfo(path.join(folder, "autoencoder_latent_panel.png")),
    summary: fileInfo(path.join(folder, "autoencoder_summary_panel.png"))
  };
  const ok = result.code === 0 && outputs.metrics.exists && outputs.timeseries.exists;
  if (!ok) {
    const reason = result.error || stderr || stdout || "Autoencoder backend finished without required outputs.";
    event.sender.send("tracking:log", { caseId, text: `Autoencoder failed: ${reason}\n` });
  } else {
    event.sender.send("tracking:log", { caseId, text: "Autoencoder analysis complete\n" });
  }
  return { ...result, ok, command: cmd.text, cwd: cmd.cwd, stdout, stderr, log_path: logPath, outputs };
});

ipcMain.handle("unifiedReport:generate", async (event, { caseId }) => {
  const config = loadConfig();
  await fsp.mkdir(logDir(), { recursive: true });
  const logPath = path.join(logDir(), `${caseId}_unified_report_${Date.now()}.log`);
  const cmd = backendCommand(config, "generate-unified-report", ["--case-id", caseId]);
  event.sender.send("tracking:log", { caseId, text: `COMMAND ${cmd.text}\nCWD ${cmd.cwd}\n` });
  const result = await runCommandCapture(cmd.command, cmd.args, cmd.cwd, logPath, (text) => {
    event.sender.send("tracking:log", { caseId, text });
  });
  const stdout = result.stdout || "";
  const stderr = result.stderr || "";
  const folder = reportPath(caseId, "unified_report");
  const outputs = {
    report: fileInfo(path.join(folder, "unified_case_report.md")),
    summary: fileInfo(path.join(folder, "unified_case_summary.json")),
    metrics_card: fileInfo(path.join(folder, "unified_metrics_card.json")),
    scorecard: fileInfo(path.join(folder, "unified_case_scorecard.png")),
    manifest: fileInfo(path.join(folder, "unified_report_manifest.md"))
  };
  return { ...result, command: cmd.text, cwd: cmd.cwd, stdout, stderr, log_path: logPath, outputs };
});

ipcMain.handle("redditTemplate:generate", async (event, { caseId }) => {
  const config = loadConfig();
  await fsp.mkdir(logDir(), { recursive: true });
  const logPath = path.join(logDir(), `${caseId}_reddit_template_${Date.now()}.log`);
  const cmd = backendCommand(config, "generate-reddit-template", ["--case-id", caseId]);
  event.sender.send("tracking:log", { caseId, text: `COMMAND ${cmd.text}\nCWD ${cmd.cwd}\n` });
  const result = await runCommandCapture(cmd.command, cmd.args, cmd.cwd, logPath, (text) => {
    event.sender.send("tracking:log", { caseId, text });
  });
  const stdout = result.stdout || "";
  const stderr = result.stderr || "";
  const folder = reportPath(caseId, "reddit_template");
  const outputs = {
    template_en: fileInfo(path.join(folder, "reddit_post_template_en.md")),
    template_es: fileInfo(path.join(folder, "reddit_post_template_es.md")),
    short_en: fileInfo(path.join(folder, "reddit_post_short_en.md")),
    titles_en: fileInfo(path.join(folder, "reddit_title_options_en.txt")),
    data: fileInfo(path.join(folder, "reddit_post_data.json")),
    manifest: fileInfo(path.join(folder, "reddit_template_manifest.md"))
  };
  return { ...result, command: cmd.text, cwd: cmd.cwd, stdout, stderr, log_path: logPath, outputs };
});

ipcMain.handle("evidenceVideo:export", async (event, { caseId, format }) => {
  const config = loadConfig();
  await fsp.mkdir(logDir(), { recursive: true });
  const selectedFormat = format || "all";
  const logPath = path.join(logDir(), `${caseId}_evidence_video_${selectedFormat}_${Date.now()}.log`);
  const cmd = backendCommand(config, "export-evidence-video", ["--case-id", caseId, "--format", selectedFormat]);
  event.sender.send("tracking:log", { caseId, text: `COMMAND ${cmd.text}\nCWD ${cmd.cwd}\n` });
  const result = await runCommandCapture(cmd.command, cmd.args, cmd.cwd, logPath, (text) => {
    event.sender.send("tracking:log", { caseId, text });
  });
  const folder = reportPath(caseId, "video_exports");
  const outputs = {
    folder: fileInfo(folder),
    manifest: fileInfo(path.join(folder, "video_export_manifest.json")),
    video_16x9: fileInfo(path.join(folder, `${caseId}_16x9.mp4`)),
    video_9x16: fileInfo(path.join(folder, `${caseId}_9x16.mp4`)),
    video_1x1: fileInfo(path.join(folder, `${caseId}_1x1.mp4`))
  };
  const ok = result.code === 0 && outputs.manifest.exists && (
    outputs.video_16x9.exists || outputs.video_9x16.exists || outputs.video_1x1.exists
  );
  if (!ok) {
    event.sender.send("tracking:log", { caseId, text: `Evidence video export failed: ${result.stderr || result.stdout || result.error || "required outputs were not generated"}\n` });
  } else {
    event.sender.send("tracking:log", { caseId, text: "Evidence video export complete\n" });
  }
  return { ...result, ok, command: cmd.text, cwd: cmd.cwd, log_path: logPath, outputs };
});

ipcMain.handle("srv:run", async (event, { caseId }) => {
  const config = loadConfig();
  await fsp.mkdir(logDir(), { recursive: true });
  const logPath = path.join(logDir(), `${caseId}_srv_analysis_${Date.now()}.log`);
  const args = ["-m", "scripts.run_track_srv_analysis", "--case-id", caseId];
  const commandText = `"${config.python_exe}" ${args.join(" ")}`;
  event.sender.send("tracking:log", { caseId, text: `COMMAND ${commandText}\nCWD ${config.lab_root}\n` });
  const child = spawn(config.python_exe, args, { cwd: config.lab_root, windowsHide: true });
  const stream = fs.createWriteStream(logPath, { flags: "a" });
  let stdout = "";
  let stderr = "";
  child.stdout.on("data", (data) => {
    const text = data.toString();
    stdout += text;
    stream.write(text);
    event.sender.send("tracking:log", { caseId, text });
  });
  child.stderr.on("data", (data) => {
    const text = data.toString();
    stderr += text;
    stream.write(text);
    event.sender.send("tracking:log", { caseId, text });
  });
  const result = await new Promise((resolve) => child.on("close", (code) => resolve({ code })));
  stream.end();
  const folder = outputPath(caseId, "srv_analysis");
  const outputs = {
    metrics: fileInfo(path.join(folder, "srv_metrics.json")),
    csv: fileInfo(path.join(folder, "srv_timeseries.csv")),
    report: fileInfo(reportPath(caseId, "srv_analysis_report.md")),
    raw: fileInfo(path.join(folder, "srv_contact_sheet_raw.png")),
    enhanced: fileInfo(path.join(folder, "srv_contact_sheet_enhanced.png")),
    sequence: fileInfo(path.join(folder, "srv_stabilized_sequence_panel.png")),
    comparison: fileInfo(path.join(folder, "srv_comparison_panel.png")),
    quality: fileInfo(path.join(folder, "srv_quality_panel.png")),
    average_stack: fileInfo(path.join(folder, "srv_stack_average.png")),
    median_stack: fileInfo(path.join(folder, "srv_stack_median.png")),
    best_sharpness: fileInfo(path.join(folder, "srv_stack_best_sharpness.png")),
    video: fileInfo(path.join(folder, "srv_stabilized_crop_sequence.mp4"))
  };
  return { ...result, command: commandText, cwd: config.lab_root, stdout, stderr, log_path: logPath, outputs };
});

ipcMain.handle("srvCore:run", async (event, { caseId }) => {
  const config = loadConfig();
  await fsp.mkdir(logDir(), { recursive: true });
  const logPath = path.join(logDir(), `${caseId}_srv_object_core_${Date.now()}.log`);
  const args = ["-m", "scripts.run_track_srv_analysis", "--case-id", caseId, "--object-core"];
  const commandText = `"${config.python_exe}" ${args.join(" ")}`;
  event.sender.send("tracking:log", { caseId, text: `COMMAND ${commandText}\nCWD ${config.lab_root}\n` });
  const child = spawn(config.python_exe, args, { cwd: config.lab_root, windowsHide: true });
  const stream = fs.createWriteStream(logPath, { flags: "a" });
  let stdout = "";
  let stderr = "";
  child.stdout.on("data", (data) => {
    const text = data.toString();
    stdout += text;
    stream.write(text);
    event.sender.send("tracking:log", { caseId, text });
  });
  child.stderr.on("data", (data) => {
    const text = data.toString();
    stderr += text;
    stream.write(text);
    event.sender.send("tracking:log", { caseId, text });
  });
  const result = await new Promise((resolve) => child.on("close", (code) => resolve({ code })));
  stream.end();
  const folder = outputPath(caseId, "srv_analysis", "object_core");
  const outputs = {
    metrics: fileInfo(path.join(folder, "srv_core_metrics.json")),
    csv: fileInfo(path.join(folder, "object_core_rois.csv")),
    json: fileInfo(path.join(folder, "object_core_rois.json")),
    raw: fileInfo(path.join(folder, "srv_core_contact_sheet_raw.png")),
    enhanced: fileInfo(path.join(folder, "srv_core_contact_sheet_enhanced.png")),
    sequence: fileInfo(path.join(folder, "srv_core_stabilized_sequence_panel.png")),
    comparison: fileInfo(path.join(folder, "srv_core_comparison_panel.png")),
    quality: fileInfo(path.join(folder, "srv_core_quality_panel.png")),
    average_stack: fileInfo(path.join(folder, "srv_core_stack_average.png")),
    median_stack: fileInfo(path.join(folder, "srv_core_stack_median.png")),
    best_sharpness: fileInfo(path.join(folder, "srv_core_stack_best_sharpness.png")),
    video: fileInfo(path.join(folder, "srv_core_stabilized_crop_sequence.mp4")),
    report: fileInfo(reportPath(caseId, "srv_analysis_report.md"))
  };
  return { ...result, command: commandText, cwd: config.lab_root, stdout, stderr, log_path: logPath, outputs };
});

ipcMain.handle("path:open", async (_event, targetPath) => {
  if (!targetPath) return { ok: false, error: "No path provided" };
  const result = await shell.openPath(targetPath);
  return { ok: result === "", error: result };
});

ipcMain.handle("validation:save", async (_event, payload) => {
  const { caseId, decision, notes } = payload;
  const file = validationPath(caseId);
  let validation = fs.existsSync(file) ? readJson(file) : {
    case_id: caseId,
    selected_backend: "opencv",
    bad_segments: []
  };
  if (decision === "correct") {
    validation.track_validated = true;
    validation.track_is_correct = true;
    validation.object_is_real_target = true;
    validation.needs_reprompt = false;
  } else if (decision === "incorrect") {
    validation.track_validated = false;
    validation.track_is_correct = false;
    validation.object_is_real_target = false;
    validation.needs_reprompt = true;
  } else if (decision === "reprompt") {
    validation.track_validated = false;
    validation.track_is_correct = null;
    validation.object_is_real_target = null;
    validation.needs_reprompt = true;
  }
  validation.notes = notes || validation.notes || "";
  await writeJson(file, validation);
  return { validation_path: file, validation };
});
