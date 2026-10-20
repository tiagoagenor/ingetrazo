#!/bin/bash
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Marco Sumari Tellez and IngeTrazo contributors.
#
# Builds the Rust IngeTrazo macOS app (M9.2): IngeTrazo.app (universal
# binary: aarch64 + x86_64 joined with lipo) inside a .dmg, signed and
# notarised when the Apple credentials are present, ad-hoc signed otherwise.
#
#     packaging/macos/build.sh [-v VERSION] [--arch "aarch64-apple-darwin x86_64-apple-darwin"]
#                              [--skip-build] [--skip-dwg] [--skip-dmg]
#
# Environment (all optional; without them the build is ad-hoc signed and the
# first launch on another Mac is right-click > Open):
#   APPLE_SIGN_IDENTITY   codesign identity ("Developer ID Application: ... (TEAMID)").
#                         The CI sets it after importing APPLE_CERT_P12_BASE64.
#   APPLE_ID, APPLE_TEAM_ID, APPLE_APP_PASSWORD
#                         notarytool credentials (an app-specific password).
#   INGETRAZO_MAC_TARGETS Rust targets to build (default: both architectures;
#                         one target gives a per-arch app and skips lipo).
#
# The Python app's packaging (packaging/build-macos-app.sh) is untouched.
set -euo pipefail

ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
OUT=$ROOT/dist/rust
STAGE_APP=$OUT/IngeTrazo.app
TARGETS=${INGETRAZO_MAC_TARGETS:-"aarch64-apple-darwin x86_64-apple-darwin"}
VERSION=""
SKIP_BUILD=0
SKIP_DWG=0
SKIP_DMG=0
# Same deployment floor as the plist's LSMinimumSystemVersion.
MIN_MACOS=11.0

while [ $# -gt 0 ]; do
  case $1 in
    -v|--version) VERSION=$2; shift 2 ;;
    --arch) TARGETS=$2; shift 2 ;;
    --skip-build) SKIP_BUILD=1; shift ;;
    --skip-dwg) SKIP_DWG=1; shift ;;
    --skip-dmg) SKIP_DMG=1; shift ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
done

warn() { echo "warning: $*" >&2; }

# -- version ---------------------------------------------------------------
if [ -z "$VERSION" ]; then
  VERSION=$(awk '/^\[workspace\.package\]/{f=1;next} f && /^version[[:space:]]*=/{gsub(/.*"|".*/,"");print;exit}' "$ROOT/rust/Cargo.toml")
fi
[ -n "$VERSION" ] || { echo 'cannot read the workspace version' >&2; exit 1; }
echo "IngeTrazo (Rust) macOS app, version $VERSION"

# -- 1. build --------------------------------------------------------------
BIN=$OUT/ingetrazo-universal
if [ "$SKIP_BUILD" -eq 0 ]; then
  export MACOSX_DEPLOYMENT_TARGET=$MIN_MACOS
  bins=()
  for t in $TARGETS; do
    rustup target add "$t" >/dev/null 2>&1 || true
    (cd "$ROOT/rust" && cargo build --release --locked -p ingetrazo-app --target "$t")
    bins+=("$ROOT/rust/target/$t/release/ingetrazo")
  done
  mkdir -p "$OUT"
  if [ ${#bins[@]} -gt 1 ]; then
    lipo -create "${bins[@]}" -output "$BIN"
  else
    cp "${bins[0]}" "$BIN"
  fi
fi
[ -f "$BIN" ] || { echo "missing $BIN (run without --skip-build)" >&2; exit 1; }
lipo -info "$BIN"

# -- 2. stage the .app -----------------------------------------------------
rm -rf "$STAGE_APP"
mkdir -p "$STAGE_APP/Contents/MacOS" "$STAGE_APP/Contents/Resources"
cp "$BIN" "$STAGE_APP/Contents/MacOS/ingetrazo"
chmod 755 "$STAGE_APP/Contents/MacOS/ingetrazo"
cp "$ROOT/LICENSE" "$STAGE_APP/Contents/Resources/LICENSE"

# Info.plist (bundle id shared with the Python app: ingetrazo.spec).
sed "s/@VERSION@/$VERSION/g; s/@MIN_MACOS@/$MIN_MACOS/g" \
  "$ROOT/packaging/macos/Info.plist.in" > "$STAGE_APP/Contents/Info.plist"
plutil -lint "$STAGE_APP/Contents/Info.plist"
printf 'APPL????' > "$STAGE_APP/Contents/PkgInfo"

# Icon: .icns from the committed PNGs with iconutil (as build-macos-app.sh).
ICONSET=$OUT/ingetrazo.iconset
rm -rf "$ICONSET"; mkdir -p "$ICONSET"
I=$ROOT/resources/icons
cp "$I/ingetrazo_16.png"  "$ICONSET/icon_16x16.png"
cp "$I/ingetrazo_32.png"  "$ICONSET/icon_16x16@2x.png"
cp "$I/ingetrazo_32.png"  "$ICONSET/icon_32x32.png"
cp "$I/ingetrazo_64.png"  "$ICONSET/icon_32x32@2x.png"
cp "$I/ingetrazo_128.png" "$ICONSET/icon_128x128.png"
cp "$I/ingetrazo_256.png" "$ICONSET/icon_128x128@2x.png"
cp "$I/ingetrazo_256.png" "$ICONSET/icon_256x256.png"
cp "$I/ingetrazo_512.png" "$ICONSET/icon_256x256@2x.png"
cp "$I/ingetrazo_512.png" "$ICONSET/icon_512x512.png"
sips -z 1024 1024 "$I/ingetrazo_512.png" --out "$ICONSET/icon_512x512@2x.png" >/dev/null
iconutil -c icns "$ICONSET" -o "$STAGE_APP/Contents/Resources/ingetrazo.icns"
rm -rf "$ICONSET"

# Runtime data. Inside Contents/Resources, not next to the executable:
# codesign refuses loose data under Contents/MacOS. The app must resolve its
# root to ../Resources when it runs from a bundle (see the report).
# resources/blender holds a .py render script: deliberately left out.
RES=$STAGE_APP/Contents/Resources
for d in components textures colors styles mime ui; do
  src=$ROOT/resources/$d
  [ -d "$src" ] || continue
  mkdir -p "$RES/resources/$d"
  cp -R "$src/." "$RES/resources/$d/"
done
find "$RES" \( -name '.gitkeep' -o -name '*.md' -o -name '*.py' \) -delete
if [ -n "$(find "$ROOT/materials/presets" -type f ! -name '.gitkeep' 2>/dev/null | head -n1)" ]; then
  mkdir -p "$RES/materials"
  cp -R "$ROOT/materials/presets" "$RES/materials/presets"
  find "$RES/materials" -name '.gitkeep' -delete
fi

# DWG bridge: LibreDWG publishes no Mac binaries, so dwg2dxf is built from the
# release tarball (same version and hash as release-macos.yml). Best effort:
# without it the app only loses DWG import.
build_dwg() {
  local V=0.14.8597
  local SHA=af2646681858a78d756cfb9e0eeb6901f61be3d09ae8f56a98e94b1490dec4ed
  local bin=$RES/vendor/libredwg/bin
  mkdir -p "$bin"
  if [ -x "$ROOT/vendor/libredwg/bin/dwg2dxf" ]; then
    cp "$ROOT/vendor/libredwg/bin/dwg2dxf" "$bin/dwg2dxf"
    return 0
  fi
  local w archflags="" t
  w=$(mktemp -d)
  # One binary for both architectures, like the app itself.
  for t in $TARGETS; do
    case $t in aarch64*) archflags="$archflags -arch arm64" ;; x86_64*) archflags="$archflags -arch x86_64" ;; esac
  done
  (
    # No set -e inside: this runs in an if-context, so chain with &&.
    cd "$w" &&
    curl -fsSL -O "https://github.com/LibreDWG/libredwg/releases/download/$V/libredwg-$V.tar.xz" &&
    echo "$SHA  libredwg-$V.tar.xz" | shasum -a 256 -c - &&
    tar xf "libredwg-$V.tar.xz" &&
    cd "libredwg-$V" &&
    ./configure --disable-shared --disable-bindings --disable-python \
      CFLAGS="-O2 $archflags -mmacosx-version-min=$MIN_MACOS" \
      LDFLAGS="$archflags" --disable-dependency-tracking >/dev/null &&
    make -j"$(sysctl -n hw.ncpu)" >/dev/null &&
    install -m 0755 programs/dwg2dxf "$bin/dwg2dxf"
  )
  local rc=$?
  rm -rf "$w"
  return $rc
}
if [ "$SKIP_DWG" -eq 0 ]; then
  if ! build_dwg; then
    warn 'DWG bridge could not be built: the app ships without DWG import'
    rm -rf "$RES/vendor"
  fi
fi
echo "staged: $STAGE_APP"

# -- 3. audit: no Python in the bundle -------------------------------------
"$ROOT/packaging/audit_no_python.sh" "$STAGE_APP"

# -- 4. sign ---------------------------------------------------------------
IDENTITY=${APPLE_SIGN_IDENTITY:-}
ENT=$ROOT/packaging/macos/entitlements.plist
NOTARISE=0
if [ -n "$IDENTITY" ]; then
  echo "signing with: $IDENTITY"
  # Inside-out: the helper first, then the bundle. Hardened runtime + a
  # secure timestamp are what notarisation requires.
  [ -f "$RES/vendor/libredwg/bin/dwg2dxf" ] && \
    codesign --force --options runtime --timestamp --sign "$IDENTITY" "$RES/vendor/libredwg/bin/dwg2dxf"
  codesign --force --options runtime --timestamp --entitlements "$ENT" --sign "$IDENTITY" "$STAGE_APP"
  if [ -n "${APPLE_ID:-}" ] && [ -n "${APPLE_TEAM_ID:-}" ] && [ -n "${APPLE_APP_PASSWORD:-}" ]; then
    NOTARISE=1
  else
    warn 'APPLE_ID / APPLE_TEAM_ID / APPLE_APP_PASSWORD missing: signed but NOT notarised'
  fi
else
  warn 'no APPLE_SIGN_IDENTITY: ad-hoc signature only (Gatekeeper will warn on other Macs)'
  [ -f "$RES/vendor/libredwg/bin/dwg2dxf" ] && codesign --force --sign - "$RES/vendor/libredwg/bin/dwg2dxf"
  codesign --force --sign - "$STAGE_APP"
fi
codesign --verify --deep --strict --verbose=2 "$STAGE_APP"

notarise() { # notarise <file>: submit and wait; fails the build on rejection
  xcrun notarytool submit "$1" --apple-id "$APPLE_ID" --team-id "$APPLE_TEAM_ID" \
    --password "$APPLE_APP_PASSWORD" --wait
}

if [ "$NOTARISE" -eq 1 ]; then
  ZIP=$OUT/IngeTrazo-notarise.zip
  rm -f "$ZIP"
  ditto -c -k --keepParent "$STAGE_APP" "$ZIP"
  notarise "$ZIP"
  rm -f "$ZIP"
  xcrun stapler staple "$STAGE_APP"
fi

[ "$SKIP_DMG" -eq 0 ] || { echo 'dmg skipped (--skip-dmg)'; exit 0; }

# -- 5. dmg ----------------------------------------------------------------
case $TARGETS in
  *aarch64*x86_64*|*x86_64*aarch64*) SUFFIX=universal ;;
  *aarch64*) SUFFIX=arm64 ;;
  *) SUFFIX=x86_64 ;;
esac
DMG=$OUT/IngeTrazo-$VERSION-macos-$SUFFIX-rust.dmg
DMGDIR=$OUT/dmg-stage
rm -rf "$DMGDIR" "$DMG"
mkdir -p "$DMGDIR"
cp -R "$STAGE_APP" "$DMGDIR/"
ln -s /Applications "$DMGDIR/Applications"
hdiutil create -volname "IngeTrazo $VERSION" -srcfolder "$DMGDIR" -ov -format UDZO "$DMG" >/dev/null
rm -rf "$DMGDIR"
if [ -n "$IDENTITY" ]; then
  codesign --force --timestamp --sign "$IDENTITY" "$DMG"
fi
if [ "$NOTARISE" -eq 1 ]; then
  notarise "$DMG"
  xcrun stapler staple "$DMG"
  xcrun stapler validate "$DMG"
  spctl --assess --type execute --verbose=2 "$STAGE_APP" || warn 'spctl did not accept the app'
fi
"$ROOT/packaging/audit_no_python.sh" --dmg "$DMG"
echo "dmg: $DMG"
