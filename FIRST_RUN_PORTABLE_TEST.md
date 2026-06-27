# First-Run Portable Test

This checklist verifies that a freshly unzipped Windows portable build starts cleanly without private data, workspace paths, or a pre-existing batch.

## Scope

- Build artifact: `release_build/AndoFrontier-Lab-v0.3-Windows.zip`
- App executable after unzip: `AndoFrontier Lab.exe`
- Expected writable runtime root: the folder containing `AndoFrontier Lab.exe`
- Expected data root: `<folder_del_exe>/runtime/lab/data`

## Procedure

1. Copy `release_build/AndoFrontier-Lab-v0.3-Windows.zip` to a clean folder outside the workspace.
2. Extract the ZIP into an empty folder.
3. Run `AndoFrontier Lab.exe`.
4. Confirm that the app opens without an `ENOENT` error for `batch_manifest.json`.
5. Confirm that these folders are created beside the executable:
   - `runtime/`
   - `runtime/lab/`
   - `runtime/lab/data/`
   - `runtime/lab/data/batches/`
   - `runtime/lab/data/cases/`
   - `runtime/lab/data/outputs/`
   - `runtime/lab/data/reports/`
   - `runtime/logs/`
6. Confirm that no persistent data root is created under Electron's temporary extraction folder.
7. Confirm that the UI starts in an empty state:
   - Header: `No case selected`
   - Cases panel: `No cases found. Create or import a case to begin.`
   - Log: `Runtime initialized. No batch manifest found yet.`
8. Confirm that no Content Pack or Publisher Pack module is present.
9. Confirm that Public Summary remains available in the UI, but generation is blocked until a Unified Report exists.

## Expected Result

The portable app starts cleanly with no batch loaded. Missing `local_batch/batch_manifest.json` is treated as a normal first-run condition, not a critical error.
