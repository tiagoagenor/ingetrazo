#!/bin/bash
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Marco Sumari Tellez and IngeTrazo contributors.
#
# Installs the Rust IngeTrazo into an FHS-style prefix (M9.3). One layout for
# the AppImage (PREFIX=AppDir/usr) and the Flatpak (PREFIX=/app):
#
#   PREFIX/lib/ingetrazo/ingetrazo         the binary
#   PREFIX/lib/ingetrazo/resources/...     runtime data, next to the binary
#   PREFIX/lib/ingetrazo/vendor/libredwg/bin/dwg2dxf   (optional DWG bridge)
#   PREFIX/bin/ingetrazo                   wrapper script
#   PREFIX/share/{applications,metainfo,mime/packages,icons/hicolor/...}
#
# Usage: install-tree.sh <prefix> <ingetrazo-binary> [version]
# No Python anywhere: only the binary, data, desktop metadata and icons.
set -euo pipefail

PREFIX=${1:?usage: install-tree.sh <prefix> <binary> [version]}
BIN=${2:?usage: install-tree.sh <prefix> <binary> [version]}
VERSION=${3:-}
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
ROOT=$(cd "$HERE/../.." && pwd)
APP_ID=com.ingetrazo.IngeTrazo

if [ -z "$VERSION" ]; then
  VERSION=$(awk '/^\[workspace\.package\]/{f=1;next} f && /^version[[:space:]]*=/{gsub(/.*"|".*/,"");print;exit}' "$ROOT/rust/Cargo.toml")
fi

LIB=$PREFIX/lib/ingetrazo
SHARE=$PREFIX/share
mkdir -p "$LIB" "$PREFIX/bin" "$SHARE/applications" "$SHARE/metainfo" "$SHARE/mime/packages"

install -m 0755 "$BIN" "$LIB/ingetrazo"
install -m 0644 "$ROOT/LICENSE" "$LIB/LICENSE"

# Runtime data next to the binary (resources/blender holds a .py render
# script: deliberately left out).
for d in components textures colors styles mime ui; do
  src=$ROOT/resources/$d
  [ -d "$src" ] || continue
  mkdir -p "$LIB/resources/$d"
  cp -R "$src/." "$LIB/resources/$d/"
done
find "$LIB" \( -name '.gitkeep' -o -name '*.md' -o -name '*.py' \) -delete
if [ -n "$(find "$ROOT/materials/presets" -type f ! -name '.gitkeep' 2>/dev/null | head -n1)" ]; then
  mkdir -p "$LIB/materials"
  cp -R "$ROOT/materials/presets" "$LIB/materials/presets"
  find "$LIB/materials" -name '.gitkeep' -delete
fi
# Optional DWG bridge, when the builder provides a Linux dwg2dxf.
if [ -x "$ROOT/vendor/libredwg/bin/dwg2dxf" ]; then
  mkdir -p "$LIB/vendor/libredwg/bin"
  install -m 0755 "$ROOT/vendor/libredwg/bin/dwg2dxf" "$LIB/vendor/libredwg/bin/dwg2dxf"
fi

# Launcher: PREFIX-relative, so it works from an AppDir mounted anywhere.
cat > "$PREFIX/bin/ingetrazo" <<'LAUNCHER'
#!/bin/sh
here=$(dirname "$(readlink -f "$0")")
exec "$here/../lib/ingetrazo/ingetrazo" "$@"
LAUNCHER
chmod 0755 "$PREFIX/bin/ingetrazo"

# Desktop entry, shared MIME type, AppStream metadata.
install -m 0644 "$HERE/$APP_ID.desktop" "$SHARE/applications/$APP_ID.desktop"
install -m 0644 "$ROOT/resources/mime/ingetrazo.xml" "$SHARE/mime/packages/$APP_ID.xml"
sed "s/@VERSION@/$VERSION/g; s/@DATE@/$(date -u +%Y-%m-%d)/g" \
  "$HERE/$APP_ID.metainfo.xml" > "$SHARE/metainfo/$APP_ID.metainfo.xml"
chmod 0644 "$SHARE/metainfo/$APP_ID.metainfo.xml"

# Icons: application icon in the sizes the repo ships, plus the document
# (mimetype) icons the shared-mime-info entry points at.
for s in 16 32 48 64 128 256 512; do
  dir=$SHARE/icons/hicolor/${s}x${s}/apps
  mkdir -p "$dir"
  install -m 0644 "$ROOT/resources/icons/ingetrazo_$s.png" "$dir/$APP_ID.png"
done
mkdir -p "$SHARE/icons/hicolor/scalable/apps"
install -m 0644 "$ROOT/resources/icons/ingetrazo.svg" "$SHARE/icons/hicolor/scalable/apps/$APP_ID.svg"
for src in "$ROOT"/resources/icons/hicolor/*/mimetypes; do
  [ -d "$src" ] || continue
  size=$(basename "$(dirname "$src")")
  mkdir -p "$SHARE/icons/hicolor/$size/mimetypes"
  cp "$src"/*.png "$SHARE/icons/hicolor/$size/mimetypes/"
done
echo "installed IngeTrazo $VERSION into $PREFIX"
