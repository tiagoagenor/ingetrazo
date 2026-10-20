<#
.SYNOPSIS
  M9.4: fails when a staged package or an installed folder contains Python.

.DESCRIPTION
  The Rust app must ship no Python runtime, bytecode, sources or Qt/PySide.
  Walks the folder and checks (1) file and directory names against the
  forbidden patterns below, (2) .zip/.whl archive members, and (3) the
  executables and DLLs for embedded-interpreter imports (python3x.dll,
  Py_Initialize, PySide6/shiboken). Written in PowerShell on purpose: the
  audit must not need Python.

  Exit code 0 = clean, 1 = violations (listed), 2 = bad arguments.

.PARAMETER Path
  Folder to audit (the staged package, or an install directory).
.PARAMETER Installer
  Optional: an installer .exe (Inno Setup). It is installed silently into a
  temporary folder, audited, and uninstalled again.
#>
[CmdletBinding()]
param(
    [string]$Path,
    [string]$Installer
)
$ErrorActionPreference = 'Stop'

# File names (leaf) that mean Python / Qt for Python.
$ForbiddenFiles = @(
    'python*.dll', 'python*.exe', 'pythonw*.exe', 'libpython*', 'python*._pth',
    '*.pyd', '*.pyc', '*.pyo', '*.pyw', '*.py', '*.pyi', '*.whl', '*.egg',
    'pyvenv.cfg', 'base_library.zip', 'PYZ*.pyz', '*.pyz',
    'PySide6*', 'shiboken*', 'Qt6*.dll', 'Qt5*.dll',
    'ingetrazo_py*', 'ingetrazo-mcp.exe'
)
# Directory names.
$ForbiddenDirs = @('_internal', '__pycache__', 'site-packages', 'PySide6', 'shiboken6', 'Lib', 'DLLs')
# Byte strings that mean "this binary embeds / links an interpreter".
$ForbiddenImports = @('python3\d{0,2}\.dll', 'Py_Initialize', 'Py_InitializeEx', 'PyImport_', 'PySide6', 'shiboken6', 'Qt6Core\.dll')

function Test-Folder([string]$Root) {
    $bad = New-Object System.Collections.Generic.List[string]
    $Root = (Resolve-Path $Root).Path
    $items = Get-ChildItem $Root -Recurse -Force
    foreach ($it in $items) {
        $rel = $it.FullName.Substring($Root.Length).TrimStart('\')
        if ($it.PSIsContainer) {
            foreach ($d in $ForbiddenDirs) {
                if ($it.Name -ieq $d) { $bad.Add("dir   $rel") }
            }
            continue
        }
        foreach ($p in $ForbiddenFiles) {
            if ($it.Name -like $p) { $bad.Add("file  $rel   (matches $p)"); break }
        }
        $ext = $it.Extension.ToLower()
        if ($ext -in '.zip', '.whl', '.egg', '.jar') {
            Add-Type -AssemblyName System.IO.Compression.FileSystem
            try {
                $z = [IO.Compression.ZipFile]::OpenRead($it.FullName)
                foreach ($e in $z.Entries) {
                    foreach ($p in $ForbiddenFiles) {
                        if ($e.Name -like $p) { $bad.Add("zip   $rel!$($e.FullName)   (matches $p)"); break }
                    }
                }
                $z.Dispose()
            } catch { }
        }
        if ($ext -in '.exe', '.dll') {
            # ASCII scan of the image. The Rust exe is ~15 MB: fine.
            $bytes = [IO.File]::ReadAllBytes($it.FullName)
            $text = [Text.Encoding]::GetEncoding(28591).GetString($bytes)
            foreach ($rx in $ForbiddenImports) {
                if ($text -cmatch $rx) { $bad.Add("bin   $rel   (embeds '$($Matches[0])')"); break }
            }
        }
    }
    return $bad
}

function Install-Temp([string]$Setup) {
    $dir = Join-Path ([IO.Path]::GetTempPath()) ("igz-audit-" + [guid]::NewGuid().ToString('N').Substring(0, 8))
    $p = Start-Process $Setup -ArgumentList '/VERYSILENT', '/SUPPRESSMSGBOXES', '/NORESTART', '/SP-', "/DIR=`"$dir`"", '/TASKS=""' -Wait -PassThru
    if ($p.ExitCode -ne 0) { throw "installer exited with $($p.ExitCode)" }
    return $dir
}

if (-not $Path -and -not $Installer) { Write-Error 'give -Path and/or -Installer'; exit 2 }

$violations = New-Object System.Collections.Generic.List[string]
$checked = 0
if ($Path) {
    if (-not (Test-Path $Path)) { Write-Error "no such folder: $Path"; exit 2 }
    $checked += (Get-ChildItem $Path -Recurse -File -Force | Measure-Object).Count
    foreach ($v in (Test-Folder $Path)) { $violations.Add($v) }
}
if ($Installer) {
    $dir = Install-Temp (Resolve-Path $Installer).Path
    try {
        $checked += (Get-ChildItem $dir -Recurse -File -Force | Measure-Object).Count
        foreach ($v in (Test-Folder $dir)) { $violations.Add("[installed] $v") }
    } finally {
        $un = Get-ChildItem $dir -Filter 'unins*.exe' -ErrorAction SilentlyContinue | Select-Object -First 1
        if ($un) { Start-Process $un.FullName -ArgumentList '/VERYSILENT', '/SUPPRESSMSGBOXES', '/NORESTART' -Wait }
        if (Test-Path $dir) { Remove-Item $dir -Recurse -Force -ErrorAction SilentlyContinue }
    }
}

if ($violations.Count -gt 0) {
    Write-Host "audit_no_python: FAILED, $($violations.Count) violation(s) in $checked file(s):" -ForegroundColor Red
    $violations | ForEach-Object { Write-Host "  $_" }
    exit 1
}
Write-Host "audit_no_python: OK, no Python in $checked file(s)"
exit 0
