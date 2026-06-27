$ErrorActionPreference = "Stop"

$RepoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
$ReleaseRoot = Join-Path $RepoRoot "release_build"
$BackendDir = Join-Path $ReleaseRoot "backend"
$PortableDir = Join-Path $ReleaseRoot "AndoFrontier-Lab-v0.3.1-Windows"
$ZipPath = Join-Path $ReleaseRoot "AndoFrontier-Lab-v0.3.1-Windows.zip"
$DesktopDir = Join-Path $RepoRoot "desktop"

Remove-Item -Recurse -Force $ReleaseRoot -ErrorAction SilentlyContinue
New-Item -ItemType Directory -Force $ReleaseRoot | Out-Null

Push-Location $RepoRoot
try {
  python -m compileall lab/scripts lab/uap_forensics
  python lab/scripts/check_public_release_safety.py
  powershell -ExecutionPolicy Bypass -File scripts\build_backend_windows.ps1
} finally {
  Pop-Location
}

if (-not (Test-Path (Join-Path $BackendDir "andofrontier-lab-cli.exe"))) {
  throw "Backend exe missing after build."
}

Push-Location $DesktopDir
try {
  if (-not (Test-Path "node_modules")) {
    npm install
  }
  npm run smoke
  node --check src\renderer.js
  node --check electron\main.js
  node --check electron\preload.js
  npm run dist:win
} finally {
  Pop-Location
}

$BuiltExe = Join-Path $DesktopDir "dist\AndoFrontier Lab.exe"
if (-not (Test-Path $BuiltExe)) {
  throw "Electron portable exe missing: $BuiltExe"
}

New-Item -ItemType Directory -Force $PortableDir | Out-Null
Copy-Item $BuiltExe (Join-Path $PortableDir "AndoFrontier Lab.exe") -Force
Copy-Item (Join-Path $RepoRoot "start-andofrontier-lab.bat") $PortableDir -Force
Copy-Item (Join-Path $RepoRoot "README_RUN_WINDOWS.md") $PortableDir -Force
Copy-Item (Join-Path $RepoRoot "LICENSE.md") $PortableDir -Force
Copy-Item (Join-Path $RepoRoot "DISCLAIMER.md") $PortableDir -Force

New-Item -ItemType Directory -Force (Join-Path $PortableDir "runtime\lab\data") | Out-Null
New-Item -ItemType Directory -Force (Join-Path $PortableDir "runtime\lab\config") | Out-Null

Remove-Item -Force $ZipPath -ErrorAction SilentlyContinue
Compress-Archive -Path (Join-Path $PortableDir "*") -DestinationPath $ZipPath -Force

Write-Host "Release folder: $PortableDir"
Write-Host "Release zip: $ZipPath"
