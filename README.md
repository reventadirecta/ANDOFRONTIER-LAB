# AndoFrontier Lab

Source-available UAP forensic analysis lab.

**Track first. Analyze after.**

## Download

Normal users should download the Windows release from GitHub Releases, run `AndoFrontier Lab.exe`, and start with `Import Video`.

Expected release asset:

```text
AndoFrontier-Lab-v0.3-Windows.zip
```

The Windows release is designed so users do not need to install Node, run `npm install`, create a Python virtual environment, or manually configure Python.

## What The App Does

AndoFrontier Lab is a local Windows app for traceable UAP video review. A human marks the target object first; analysis follows the validated track.

Public modules:

- human-validated tracking
- dynamic ROI reconstruction
- motion / optical flow
- spectral analysis
- FLIR/IR relative intensity
- SRV / visual reconstruction
- clean controls
- PCA baseline
- autoencoder baseline
- unified case report
- public Reddit technical template

## Public Scope

This public release cuts at:

1. Unified Case Report
2. Public Summary / Reddit Technical Template

No original videos, heavy outputs, private reports, generated publication clips, local machine paths, secrets, local configs, or private publication automation are included.

## What This Is Not

- not an origin-claim system
- not a proof engine
- not a dataset release
- not a video archive
- not a publisher automation suite

The public workflow is intended for transparent technical review. Any public write-up should preserve uncertainty, source limitations, and method dependence.

## User Instructions

See `README_RUN_WINDOWS.md` in the Windows release ZIP.

## Developer Setup

Developer setup is secondary. The normal user flow is the packaged Windows app.

Repository layout:

```text
andofrontier-lab-public/
  lab/        Python analysis engine
  desktop/    Electron Windows app
  scripts/    Windows release build and safety scripts
```

Lab development:

```powershell
cd lab
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
copy config\lab.example.json config\lab.local.json
```

Desktop development:

```powershell
cd desktop
npm install
copy config\lab.local.example.json config\lab.local.json
npm start
```

Build Windows release:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\build_windows_app_release.ps1
```

## Public Validation

From the repository root:

```powershell
python -m compileall lab/scripts lab/uap_forensics
python lab/scripts/check_public_release_safety.py
cd desktop
npm run smoke
node --check src\renderer.js
node --check electron\main.js
node --check electron\preload.js
```

Release package safety:

```powershell
python scripts/check_release_package_safety.py
```

## Private Modules

Private content/publisher automation is documented in `PRIVATE_MODULES.md` and is not part of this release.

## License

This project is released under the AndoFrontier Lab Source-Available Non-Commercial License v1.0. See `LICENSE.md`.
