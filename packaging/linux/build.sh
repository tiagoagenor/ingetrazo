#!/bin/bash
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Marco Sumari Tellez and IngeTrazo contributors.
#
# Builds the Rust IngeTrazo Linux packages (M9.3).
#
#     packaging/linux/build.sh appimage [--skip-build]
#     packaging/linux/build.sh flatpak-vendor     # prepares the offline cargo sources
#     packaging/linux/build.sh flatpak            # vendor + flatpak-builder (needs it installed)
#     packaging/linux/build.sh all
#
# The Snap is built by snapcraft from snap/snapcraft.yaml (`snapcraft` or
# snapcore/action-build); it does not go through this script.
#
# AppImage: no libraries are bundled. wgpu loads Vulkan / OpenGL / Wayland /
# X11 / xkbcommon from the host at run time (dlopen), so the GPU drivers are
# always the host's. Build on the oldest glibc you want to support (CI uses
# ubuntu-22.04).
#
# The Python packaging (packaging/build-appimage.sh, packaging/flatpak,
# packaging/snap) is untouched and keeps shipping until M10.
set -euo pipefail

ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
HERE=$ROOT/packaging/linux
OUT=$ROOT/dist/linux
APP_ID=com.ingetrazo.IngeTrazo

cmd=${1:-appimage}
shift || true
SKIP_BUILD=0
for a in "$@"; do
  case $a in
    --skip-build) SKIP_BUILD=1 ;;
    *) echo "unknown option: $a" >&2; exit 2 ;;
  esac
done

VERSION=${INGETRAZO_VERSION:-}
if [ -z "$VERSION" ]; then
  VERSION=$(awk '/^\[workspace\.package\]/{f=1;next} f && /^version[[:space:]]*=/{gsub(/.*"|".*/,"");print;exit}' "$ROOT/rust/Cargo.toml")
fi
[ -n "$VERSION" ] || { echo 'cannot read the workspace version' >&2; exit 1; }

do_appimage() {
  local arch bin appdir tool
  arch=$(uname -m)
  bin=$ROOT/rust/target/release/ingetrazo
  if [ "$SKIP_BUILD" -eq 0 ]; then
    (cd "$ROOT/rust" && cargo build --release --locked -p ingetrazo-app)
  fi
  [ -f "$bin" ] || { echo "missing $bin" >&2; exit 1; }

  appdir=$OUT/IngeTrazo.AppDir
  rm -rf "$appdir"
  mkdir -p "$appdir/usr"
  "$HERE/install-tree.sh" "$appdir/usr" "$bin" "$VERSION"

  # AppDir root entries the AppImage spec asks for.
  cp "$ROOT/resources/icons/ingetrazo_256.png" "$appdir/$APP_ID.png"
  ln -s "$APP_ID.png" "$appdir/.DirIcon"
  cp "$appdir/usr/share/applications/$APP_ID.desktop" "$appdir/$APP_ID.desktop"
  cat > "$appdir/AppRun" <<'APPRUN'
#!/bin/sh
here=$(dirname "$(readlink -f "$0")")
export XDG_DATA_DIRS="$here/usr/share:${XDG_DATA_DIRS:-/usr/local/share:/usr/share}"
exec "$here/usr/bin/ingetrazo" "$@"
APPRUN
  chmod 0755 "$appdir/AppRun"

  # M9.4: nothing Python in the AppDir.
  "$ROOT/packaging/audit_no_python.sh" "$appdir"

  tool=${APPIMAGETOOL:-}
  if [ -z "$tool" ]; then
    tool=$OUT/appimagetool
    if [ ! -x "$tool" ]; then
      curl -fsSL -o "$tool" \
        "https://github.com/AppImage/appimagetool/releases/download/continuous/appimagetool-$arch.AppImage"
      chmod +x "$tool"
    fi
  fi
  local img=$OUT/IngeTrazo-$VERSION-linux-$arch-rust.AppImage
  rm -f "$img"
  # --appimage-extract-and-run: CI runners have no FUSE.
  ARCH=$arch APPIMAGE_EXTRACT_AND_RUN=1 "$tool" --no-appstream "$appdir" "$img"
  "$ROOT/packaging/audit_no_python.sh" --appimage "$img"
  echo "AppImage: $img"
}

# Offline cargo sources for flatpak-builder (it builds with no network).
# `cargo vendor` writes rust/vendor and prints the source-replacement config,
# which goes in rust/.cargo/config.toml. Both are git-ignored build products.
flatpak_vendor() {
  mkdir -p "$ROOT/rust/.cargo"
  (cd "$ROOT/rust" && cargo vendor --locked vendor > .cargo/config.toml)
  echo "vendored: $ROOT/rust/vendor"
}

do_flatpak() {
  flatpak_vendor
  command -v flatpak-builder >/dev/null || { echo 'flatpak-builder not installed' >&2; exit 1; }
  mkdir -p "$OUT"
  flatpak-builder --force-clean --user --install-deps-from=flathub \
    --repo="$OUT/flatpak-repo" "$OUT/flatpak-build" "$HERE/flatpak/$APP_ID.yml"
  flatpak build-bundle "$OUT/flatpak-repo" "$OUT/IngeTrazo-$VERSION-rust.flatpak" "$APP_ID"
  echo "Flatpak: $OUT/IngeTrazo-$VERSION-rust.flatpak"
}

mkdir -p "$OUT"
case $cmd in
  appimage) do_appimage ;;
  flatpak-vendor) flatpak_vendor ;;
  flatpak) do_flatpak ;;
  all) do_appimage; do_flatpak ;;
  *) echo "usage: build.sh appimage|flatpak-vendor|flatpak|all" >&2; exit 2 ;;
esac
