<#
.SYNOPSIS
  Builds the Rust IngeTrazo Windows installer (M9.1).

.DESCRIPTION
  1. cargo build --release -p ingetrazo-app
  2. stages the package in dist\rust\ingetrazo (exe, runtime resources, the
     LibreDWG DWG bridge, license, icons)
  3. runs packaging\audit_no_python.ps1 on the staged folder (M9.4)
  4. builds dist\rust\ingetrazo-setup-v<version>-rust.exe with Inno Setup
  5. optionally signs the exe and the installer (see -Sign)

  The Python app's packaging (ingetrazo.spec, installer\ingetrazo.iss) is
  untouched and keeps shipping until M10.

.PARAMETER Version
  Installer version. Default: the Rust workspace version.
.PARAMETER SkipBuild
  Reuse an existing rust\target\release\ingetrazo.exe.
.PARAMETER SkipDwg
  Do not fetch the LibreDWG win64 bridge (the package then has no DWG import).
.PARAMETER SkipInstaller
  Stage and audit only (no Inno Setup needed).
.PARAMETER Sign
  Sign the exe and the installer when a certificate is configured through
  the environment (INGETRAZO_SIGN_PFX = path to a .pfx, INGETRAZO_SIGN_PASSWORD).
  Never required: without them the build goes out unsigned.
#>
[CmdletBinding()]
param(
    [string]$Version,
    [switch]$SkipBuild,
    [switch]$SkipDwg,
    [switch]$SkipInstaller,
    [switch]$Sign
)
$ErrorActionPreference = 'Stop'

$Repo  = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
$Out   = Join-Path $Repo 'dist\rust'
$Stage = Join-Path $Out 'ingetrazo'

# ── version ──────────────────────────────────────────────────────────────
if (-not $Version) {
    $toml = Get-Content (Join-Path $Repo 'rust\Cargo.toml') -Raw
    if ($toml -match '(?ms)\[workspace\.package\].*?^version\s*=\s*"([^"]+)"') {
        $Version = $Matches[1]
    } else { throw 'cannot read the workspace version' }
}
Write-Host "IngeTrazo (Rust) installer, version $Version"

# ── 1. build ─────────────────────────────────────────────────────────────
$Exe = Join-Path $Repo 'rust\target\release\ingetrazo.exe'
if (-not $SkipBuild) {
    $sw = [Diagnostics.Stopwatch]::StartNew()
    Push-Location (Join-Path $Repo 'rust')
    try {
        cargo build --release --locked -p ingetrazo-app
        if ($LASTEXITCODE -ne 0) { throw 'cargo build failed' }
    } finally { Pop-Location }
    Write-Host ("cargo build --release: {0:n0} s" -f $sw.Elapsed.TotalSeconds)
}
if (-not (Test-Path $Exe)) { throw "missing $Exe" }

# ── 2. stage ─────────────────────────────────────────────────────────────
if (Test-Path $Stage) { Remove-Item $Stage -Recurse -Force }
New-Item -ItemType Directory -Force $Stage | Out-Null
Copy-Item $Exe $Stage
Copy-Item (Join-Path $Repo 'LICENSE') $Stage

# Document icons referenced by the file-type registry entries.
Copy-Item (Join-Path $Repo 'resources\icons\ingetrazo.ico') $Stage
foreach ($n in 'igz', 'dae', 'skp') {
    Copy-Item (Join-Path $Repo "resources\icons\mimetypes\ingetrazo-$n.ico") $Stage
}

# Runtime data next to the exe. The i18n catalogs and the shaders are
# compiled into the binary (include_str!) and are not shipped loose.
# resources\blender holds a .py render script: deliberately left out.
foreach ($d in 'components', 'textures', 'colors', 'styles', 'mime', 'ui') {
    $src = Join-Path $Repo "resources\$d"
    if (-not (Test-Path $src)) { continue }
    $dst = Join-Path $Stage "resources\$d"
    New-Item -ItemType Directory -Force $dst | Out-Null
    Copy-Item "$src\*" $dst -Recurse -Force -Exclude '.gitkeep', '*.md', '*.py'
}
$presets = Join-Path $Repo 'materials\presets'
if ((Test-Path $presets) -and (Get-ChildItem $presets -File -ErrorAction SilentlyContinue)) {
    New-Item -ItemType Directory -Force (Join-Path $Stage 'materials') | Out-Null
    Copy-Item $presets (Join-Path $Stage 'materials\presets') -Recurse -Exclude '.gitkeep'
}

# DWG bridge: the official LibreDWG win64 build, the same one the Python
# installer ships (build-windows.yml). The app looks in <app>\vendor\libredwg\bin.
if (-not $SkipDwg) {
    $bin = Join-Path $Stage 'vendor\libredwg\bin'
    New-Item -ItemType Directory -Force $bin | Out-Null
    $local = Join-Path $Repo 'vendor\libredwg\bin'
    if (Test-Path (Join-Path $local 'dwg2dxf.exe')) {
        Copy-Item (Join-Path $local '*.exe'), (Join-Path $local '*.dll') $bin
    } else {
        $V   = '0.14.8597'
        $Sha = '7fee5c67c4f451f0542c0d6a3b12e91e99b15faf363fbf8b98ad84bac17ddfa2'
        $tmp = Join-Path ([IO.Path]::GetTempPath()) "libredwg-$V"
        New-Item -ItemType Directory -Force $tmp | Out-Null
        $zip = Join-Path $tmp 'ldwg.zip'
        Invoke-WebRequest "https://github.com/LibreDWG/libredwg/releases/download/$V/libredwg-$V-win64.zip" -OutFile $zip
        $got = (Get-FileHash $zip -Algorithm SHA256).Hash.ToLower()
        if ($got -ne $Sha.ToLower()) { throw "LibreDWG zip hash mismatch: $got" }
        Add-Type -AssemblyName System.IO.Compression.FileSystem
        $z = [IO.Compression.ZipFile]::OpenRead($zip)
        try {
            foreach ($e in $z.Entries) {
                if ($e.Name -in 'dwg2dxf.exe', 'libredwg-0.dll', 'libiconv-2.dll') {
                    [IO.Compression.ZipFileExtensions]::ExtractToFile($e, (Join-Path $bin $e.Name), $true)
                }
            }
        } finally { $z.Dispose() }
    }
    foreach ($f in 'dwg2dxf.exe', 'libredwg-0.dll', 'libiconv-2.dll') {
        if (-not (Test-Path (Join-Path $bin $f))) { throw "DWG bridge incomplete: $f" }
    }
}
Write-Host "staged: $Stage"

# ── signing helper (optional) ───────────────────────────────────────────
function Invoke-OptionalSign([string]$File) {
    if (-not $Sign) { return }
    if (-not $env:INGETRAZO_SIGN_PFX -or -not (Test-Path $env:INGETRAZO_SIGN_PFX)) {
        Write-Warning "no INGETRAZO_SIGN_PFX: $File stays unsigned"; return
    }
    $st = Get-ChildItem "${env:ProgramFiles(x86)}\Windows Kits\10\bin\*\x64\signtool.exe" -ErrorAction SilentlyContinue |
        Sort-Object FullName | Select-Object -Last 1
    if (-not $st) { Write-Warning 'signtool.exe not found: unsigned'; return }
    & $st.FullName sign /fd SHA256 /tr 'http://timestamp.digicert.com' /td SHA256 `
        /f $env:INGETRAZO_SIGN_PFX /p $env:INGETRAZO_SIGN_PASSWORD $File
    if ($LASTEXITCODE -ne 0) { throw "signing failed for $File" }
}
Invoke-OptionalSign (Join-Path $Stage 'ingetrazo.exe')

# ── 3. audit: no Python in the package ──────────────────────────────────
& (Join-Path $Repo 'packaging\audit_no_python.ps1') -Path $Stage
if ($LASTEXITCODE -ne 0) { throw 'the staged package contains Python' }

if ($SkipInstaller) { Write-Host 'installer skipped (-SkipInstaller)'; return }

# ── 4. installer ─────────────────────────────────────────────────────────
$iscc = Get-Command iscc.exe -ErrorAction SilentlyContinue | Select-Object -ExpandProperty Source
if (-not $iscc) {
    foreach ($p in "${env:ProgramFiles(x86)}\Inno Setup 6\ISCC.exe", "$env:ProgramFiles\Inno Setup 6\ISCC.exe",
                   "$env:LOCALAPPDATA\Programs\Inno Setup 6\ISCC.exe") {
        if (Test-Path $p) { $iscc = $p; break }
    }
}
if (-not $iscc) { throw 'Inno Setup 6 not found (choco install innosetup)' }
& $iscc "/DMyAppVersion=$Version" "/DStageDir=$Stage" "/DOutDir=$Out" (Join-Path $PSScriptRoot 'ingetrazo.iss')
if ($LASTEXITCODE -ne 0) { throw 'iscc failed' }

$Setup = Join-Path $Out "ingetrazo-setup-v$Version-rust.exe"
if (-not (Test-Path $Setup)) { throw "missing $Setup" }
Invoke-OptionalSign $Setup
Write-Host "installer: $Setup"
