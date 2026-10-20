#!/bin/bash
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Marco Sumari Tellez and IngeTrazo contributors.
#
# M9.4 for macOS and Linux (bash twin of audit_no_python.ps1): fails when a
# staged package, a .app, an extracted AppImage or a mounted .dmg contains
# Python. No Python needed to run it.
#
#     packaging/audit_no_python.sh <folder>...
#     packaging/audit_no_python.sh --appimage <file.AppImage>
#     packaging/audit_no_python.sh --dmg <file.dmg>          (macOS only)
#
# Checks: (1) file and directory names, (2) .zip/.whl/.jar members,
# (3) executables and shared libraries for embedded-interpreter strings.
# Exit 0 = clean, 1 = violations, 2 = bad arguments.
set -uo pipefail

# Leaf names (find -iname patterns) that mean Python / Qt for Python.
FORBIDDEN_FILES=(
  'python*' 'libpython*' '*.pyd' '*.pyc' '*.pyo' '*.pyw' '*.py' '*.pyi'
  '*.whl' '*.egg' 'pyvenv.cfg' 'base_library.zip' '*.pyz'
  'PySide6*' 'shiboken*' 'libQt6*' 'libQt5*'
  'ingetrazo_py*' 'ingetrazo-mcp*'
)
FORBIDDEN_DIRS=('_internal' '__pycache__' 'site-packages' 'dist-packages' 'PySide6' 'shiboken6' 'python*')
# Strings that mean "this binary embeds or links an interpreter".
FORBIDDEN_IMPORTS='libpython3|Python\.framework|Py_Initialize|PyImport_|PySide6|shiboken6|libQt6Core'

violations=()
checked=0
mnt=""
tmp=""

audit_folder() {
  local root=$1 label=${2:-} rel f p z hit n
  if [ ! -d "$root" ]; then echo "no such folder: $root" >&2; exit 2; fi
  root=$(cd "$root" && pwd)
  n=$(find "$root" -type f | wc -l)
  checked=$((checked + n))

  for p in "${FORBIDDEN_DIRS[@]}"; do
    while IFS= read -r f; do
      violations+=("${label}dir   ${f#"$root"/}")
    done < <(find "$root" -type d -iname "$p")
  done

  for p in "${FORBIDDEN_FILES[@]}"; do
    while IFS= read -r f; do
      violations+=("${label}file  ${f#"$root"/}   (matches $p)")
    done < <(find "$root" -type f -iname "$p")
  done

  # Archive members: any zip-like file listing a forbidden leaf name.
  while IFS= read -r f; do
    rel=${f#"$root"/}
    while IFS= read -r z; do
      z=${z##*/}
      case "$z" in
        *.py|*.pyc|*.pyo|*.pyd|*.pyi|*.whl|*.pyz|python*|libpython*|PySide6*|shiboken*)
          violations+=("${label}zip   $rel!$z") ;;
      esac
    done < <(unzip -Z1 "$f" 2>/dev/null || true)
  done < <(find "$root" -type f \( -iname '*.zip' -o -iname '*.whl' -o -iname '*.egg' -o -iname '*.jar' \))

  # Binaries: files with an exec bit and shared libraries.
  while IFS= read -r f; do
    if LC_ALL=C grep -a -q -E "$FORBIDDEN_IMPORTS" "$f" 2>/dev/null; then
      hit=$(LC_ALL=C grep -a -o -E "$FORBIDDEN_IMPORTS" "$f" 2>/dev/null | head -n1)
      violations+=("${label}bin   ${f#"$root"/}   (embeds '$hit')")
    fi
  done < <(find "$root" -type f \( -perm -u+x -o -name '*.so' -o -name '*.so.*' -o -name '*.dylib' \) \
             ! -name '*.sh' ! -name '*.desktop' ! -name '*.png' ! -name '*.xml' ! -name '*.svg')
}

cleanup() {
  if [ -n "$mnt" ]; then hdiutil detach "$mnt" -quiet -force >/dev/null 2>&1 || true; fi
  if [ -n "$tmp" ]; then rm -rf "$tmp"; fi
}
trap cleanup EXIT

if [ $# -lt 1 ]; then
  echo 'usage: audit_no_python.sh <folder>... | --appimage FILE | --dmg FILE' >&2
  exit 2
fi

while [ $# -gt 0 ]; do
  case $1 in
    --appimage)
      if [ ! -f "${2:-}" ]; then echo 'no such AppImage' >&2; exit 2; fi
      tmp=$(mktemp -d)
      img=$(cd "$(dirname "$2")" && pwd)/$(basename "$2")
      (cd "$tmp" && "$img" --appimage-extract >/dev/null) || { echo 'cannot extract the AppImage' >&2; exit 2; }
      audit_folder "$tmp/squashfs-root" '[appimage] '
      shift 2 ;;
    --dmg)
      if [ ! -f "${2:-}" ]; then echo 'no such dmg' >&2; exit 2; fi
      mnt=$(mktemp -d)
      hdiutil attach "$2" -mountpoint "$mnt" -nobrowse -readonly -quiet || { echo 'cannot mount the dmg' >&2; exit 2; }
      audit_folder "$mnt" '[dmg] '
      hdiutil detach "$mnt" -quiet -force >/dev/null 2>&1 || true
      mnt=""
      shift 2 ;;
    *)
      audit_folder "$1"
      shift ;;
  esac
done

if [ ${#violations[@]} -gt 0 ]; then
  echo "audit_no_python: FAILED, ${#violations[@]} violation(s) in $checked file(s):"
  printf '  %s\n' "${violations[@]}"
  exit 1
fi
echo "audit_no_python: OK, no Python in $checked file(s)"
