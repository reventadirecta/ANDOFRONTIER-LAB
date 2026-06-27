# Public Release Smoke Test

Run from the repository root.

## Lab

```powershell
python -m compileall lab/scripts lab/uap_forensics
python lab/scripts/check_public_release_safety.py
```

The safety check must not report critical findings.

## Desktop

```powershell
cd desktop
npm run smoke
node --check src\renderer.js
node --check electron\main.js
node --check electron\preload.js
```

The public smoke test does not require private videos, private batch data, or generated outputs.

## Optional Local App Run

After creating `desktop/config/lab.local.json` with your private local paths:

```powershell
npm start
```

The app should show `Public Summary`, not a publication automation tab.
