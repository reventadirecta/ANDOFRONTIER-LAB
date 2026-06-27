# Public Release Checklist

Use this before any push or public release.

- [ ] No original videos are tracked.
- [ ] No heavy generated outputs are tracked.
- [ ] No local paths are present in public files.
- [ ] No `.env`, token, secret, local config, or personal data is tracked.
- [ ] No private publication automation is exposed in the app.
- [ ] Public workflow ends at Unified Case Report and Reddit Technical Template.
- [ ] README states no origin claim and no proof engine.
- [ ] `PRIVATE_MODULES.md` documents private-local-only modules.
- [ ] Safety check passes:

```powershell
python lab/scripts/check_public_release_safety.py
```

- [ ] Compile check passes:

```powershell
python -m compileall lab/scripts lab/uap_forensics
```

- [ ] Desktop static checks pass:

```powershell
cd desktop
npm run smoke
node --check src\renderer.js
node --check electron\main.js
node --check electron\preload.js
```
