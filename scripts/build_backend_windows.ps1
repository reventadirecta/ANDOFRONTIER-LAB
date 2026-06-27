param(
  [switch]$SkipDependencyInstall
)

$ErrorActionPreference = "Stop"
$RepoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
$LabRoot = Join-Path $RepoRoot "lab"
$BuildRoot = Join-Path $RepoRoot "release_build"
$BackendOut = Join-Path $BuildRoot "backend"
$Entry = Join-Path $LabRoot "scripts\andofrontier_lab_cli.py"

New-Item -ItemType Directory -Force $BuildRoot | Out-Null
New-Item -ItemType Directory -Force $BackendOut | Out-Null

if (-not $SkipDependencyInstall) {
  python -m pip install --upgrade pyinstaller | Write-Host
  python -m pip install -r (Join-Path $LabRoot "requirements.txt") | Write-Host
}

$pyinstaller = (Get-Command pyinstaller -ErrorAction SilentlyContinue)
if (-not $pyinstaller) {
  $pyinstaller = (Get-Command python -ErrorAction Stop)
  $prefix = @("-m", "PyInstaller")
} else {
  $prefix = @()
}

$distPath = Join-Path $BuildRoot "pyinstaller_dist"
$workPath = Join-Path $BuildRoot "pyinstaller_work"
$specPath = Join-Path $BuildRoot "pyinstaller_spec"
Remove-Item -Recurse -Force $distPath, $workPath, $specPath -ErrorAction SilentlyContinue
New-Item -ItemType Directory -Force $specPath | Out-Null

$args = @(
  "--onefile",
  "--clean",
  "--noconfirm",
  "--name", "andofrontier-lab-cli",
  "--distpath", $distPath,
  "--workpath", $workPath,
  "--specpath", $specPath,
  "--paths", $LabRoot,
  "--add-data", "$($LabRoot)\scripts;scripts",
  "--add-data", "$($LabRoot)\uap_forensics;uap_forensics",
  "--hidden-import", "numpy",
  "--hidden-import", "cv2",
  "--hidden-import", "matplotlib",
  "--hidden-import", "pandas",
  "--hidden-import", "sklearn",
  "--hidden-import", "torch",
  $Entry
)

if ($prefix.Count -gt 0) {
  & $pyinstaller.Source @prefix @args
} else {
  & $pyinstaller.Source @args
}

$exe = Join-Path $distPath "andofrontier-lab-cli.exe"
if (-not (Test-Path $exe)) {
  throw "PyInstaller did not produce $exe"
}

Copy-Item $exe (Join-Path $BackendOut "andofrontier-lab-cli.exe") -Force
Write-Host "Backend built: $BackendOut\andofrontier-lab-cli.exe"
