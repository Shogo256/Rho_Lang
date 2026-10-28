#!/bin/sh
# Show .rho files with ρ's icon in GNOME Files, Dolphin and other
# freedesktop file managers: register the file type and its icon for this
# user. Undo with: install.sh --remove
set -eu
here=$(cd "$(dirname "$0")" && pwd)
data=${XDG_DATA_HOME:-$HOME/.local/share}
sizes="16 22 24 32 48 64 96 128 256 512"
if [ "${1:-}" = "--remove" ]; then
    rm -f "$data/mime/packages/rho.xml"
    for s in $sizes; do rm -f "$data/icons/hicolor/${s}x${s}/mimetypes/text-x-rho.png"; done
else
    mkdir -p "$data/mime/packages"
    cp "$here/rho.xml" "$data/mime/packages/rho.xml"
    for s in $sizes; do
        mkdir -p "$data/icons/hicolor/${s}x${s}/mimetypes"
        cp "$here/icons/text-x-rho-$s.png" "$data/icons/hicolor/${s}x${s}/mimetypes/text-x-rho.png"
    done
fi
update-mime-database "$data/mime"
[ -f "$data/icons/hicolor/index.theme" ] || cp /usr/share/icons/hicolor/index.theme "$data/icons/hicolor/" 2>/dev/null || true
gtk-update-icon-cache -f -t "$data/icons/hicolor" >/dev/null 2>&1 || true
echo "done: reopen the file manager window (or run: nautilus -q) to see it"
