#!/bin/sh
# Show .rho files with ρ's icon in GNOME Files, Dolphin and other
# freedesktop file managers: register the file type and its icon for this
# user. Undo with: install.sh --remove
#
# The icon goes into hicolor, which every theme falls back to, and also
# beside the current theme's own file-type icons: GTK 4 looks a file's
# icon names up in the current theme first, and a theme that has the
# generic text icon (Yaru does) answers with it before hicolor is reached.
set -eu
here=$(cd "$(dirname "$0")" && pwd)
data=${XDG_DATA_HOME:-$HOME/.local/share}
sizes="16 22 24 32 48 64 96 128 256 512"
theme=$(gsettings get org.gnome.desktop.interface icon-theme 2>/dev/null | tr -d "'" || true)

# the theme's mimetype directories, as (size dir, pixels)
theme_dirs() {
    [ -n "$theme" ] && [ -d "/usr/share/icons/$theme" ] || return 0
    for d in /usr/share/icons/"$theme"/*/mimetypes; do
        [ -d "$d" ] || continue
        sub=$(basename "$(dirname "$d")")
        case "$sub" in
            *x*@2x) px=$(( ${sub%%x*} * 2 )) ;;
            *x*) px=${sub%%x*} ;;
            *) continue ;;
        esac
        echo "$sub $px"
    done
}

# the closest of our sizes at or above the pixels wanted
pick() {
    for s in $sizes; do
        if [ "$s" -ge "$1" ]; then echo "$s"; return; fi
    done
    echo 512
}

if [ "${1:-}" = "--remove" ]; then
    rm -f "$data/mime/packages/rho.xml"
    for s in $sizes; do rm -f "$data/icons/hicolor/${s}x${s}/mimetypes/text-x-rho.png"; done
    [ -n "$theme" ] && find "$data/icons/$theme" -name text-x-rho.png -delete 2>/dev/null || true
else
    mkdir -p "$data/mime/packages"
    cp "$here/rho.xml" "$data/mime/packages/rho.xml"
    for s in $sizes; do
        mkdir -p "$data/icons/hicolor/${s}x${s}/mimetypes"
        cp "$here/icons/text-x-rho-$s.png" "$data/icons/hicolor/${s}x${s}/mimetypes/text-x-rho.png"
    done
    theme_dirs | while read -r sub px; do
        mkdir -p "$data/icons/$theme/$sub/mimetypes"
        cp "$here/icons/text-x-rho-$(pick "$px").png" "$data/icons/$theme/$sub/mimetypes/text-x-rho.png"
    done
fi
update-mime-database "$data/mime"
[ -f "$data/icons/hicolor/index.theme" ] || cp /usr/share/icons/hicolor/index.theme "$data/icons/hicolor/" 2>/dev/null || true
gtk-update-icon-cache -f -t "$data/icons/hicolor" >/dev/null 2>&1 || true
echo "done: close every Files window (nautilus -q) and open it again"
