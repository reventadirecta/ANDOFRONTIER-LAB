const { contextBridge, ipcRenderer } = require("electron");

contextBridge.exposeInMainWorld("forensicDesk", {
  getConfig: () => ipcRenderer.invoke("config:get"),
  listBatches: () => ipcRenderer.invoke("batches:list"),
  getBatch: (batchId) => ipcRenderer.invoke("batch:get", batchId),
  getCase: (batchId, caseId) => ipcRenderer.invoke("case:get", { batchId, caseId }),
  getCaseState: (batchId, caseId) => ipcRenderer.invoke("case:state", { batchId, caseId }),
  importVideo: (payload) => ipcRenderer.invoke("case:importVideo", payload),
  saveAnnotation: (payload) => ipcRenderer.invoke("annotation:save", payload),
  setStartFrame: (payload) => ipcRenderer.invoke("annotation:setStartFrame", payload),
  runTracking: (payload) => ipcRenderer.invoke("tracking:run", payload),
  rebuildFromTrack: (payload) => ipcRenderer.invoke("trackBased:rebuild", payload),
  runMotionAnalysis: (payload) => ipcRenderer.invoke("motion:run", payload),
  runSpectralAnalysis: (payload) => ipcRenderer.invoke("spectral:run", payload),
  runThermalAnalysis: (payload) => ipcRenderer.invoke("thermal:run", payload),
  runSrvAnalysis: (payload) => ipcRenderer.invoke("srv:run", payload),
  runSrvCoreAnalysis: (payload) => ipcRenderer.invoke("srvCore:run", payload),
  runControlsAnalysis: (payload) => ipcRenderer.invoke("controls:run", payload),
  runPcaAnalysis: (payload) => ipcRenderer.invoke("pca:run", payload),
  runAutoencoderAnalysis: (payload) => ipcRenderer.invoke("autoencoder:run", payload),
  generateUnifiedReport: (payload) => ipcRenderer.invoke("unifiedReport:generate", payload),
  generateRedditTemplate: (payload) => ipcRenderer.invoke("redditTemplate:generate", payload),
  exportEvidenceVideo: (payload) => ipcRenderer.invoke("evidenceVideo:export", payload),
  saveValidation: (payload) => ipcRenderer.invoke("validation:save", payload),
  openPath: (targetPath) => ipcRenderer.invoke("path:open", targetPath),
  onTrackingLog: (handler) => {
    const listener = (_event, payload) => handler(payload);
    ipcRenderer.on("tracking:log", listener);
    return () => ipcRenderer.removeListener("tracking:log", listener);
  },
  onTrackingProgress: (handler) => {
    const listener = (_event, payload) => handler(payload);
    ipcRenderer.on("tracking:progress", listener);
    return () => ipcRenderer.removeListener("tracking:progress", listener);
  }
});
