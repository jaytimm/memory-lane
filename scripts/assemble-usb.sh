#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "Usage: $0 USB_ROOT WINDOWS_PACKAGE [PREPARED_GALLERY]" >&2
  echo "" >&2
  echo "USB_ROOT must already contain: Family Photos/" >&2
  echo "PREPARED_GALLERY defaults to ./FamilyPhotoArchive" >&2
  exit 2
}

[[ $# -ge 2 && $# -le 3 ]] || usage

usb_root=$(realpath "$1")
windows_package=$(realpath "$2")
project_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
prepared_gallery=$(realpath "${3:-$project_root/FamilyPhotoArchive}")
photo_root="$usb_root/Family Photos"
app_root="$usb_root/Memory Lane"
gallery_root="$app_root/Gallery"

[[ -d "$usb_root" ]] || { echo "USB root not found: $usb_root" >&2; exit 2; }
[[ -d "$photo_root" ]] || { echo "Expected photo-only folder: $photo_root" >&2; exit 2; }
[[ -f "$windows_package" ]] || { echo "Windows package not found: $windows_package" >&2; exit 2; }
[[ -f "$prepared_gallery/index.html" ]] || { echo "Prepared gallery not found: $prepared_gallery" >&2; exit 2; }

album_count=$(find "$photo_root" -mindepth 1 -maxdepth 1 -type d | wc -l)
if [[ "$album_count" -lt 1 ]]; then
  echo "No album folders found inside: $photo_root" >&2
  exit 2
fi

mkdir -p "$gallery_root"

for directory in .model-cache assets previews search thumbnails; do
  if [[ -d "$prepared_gallery/$directory" ]]; then
    mkdir -p "$gallery_root/$directory"
    # FAT/exFAT drives cannot store Unix symlinks. Dereference model-cache
    # links (and any future linked assets) so the Windows USB is self-contained.
    rsync -aL --delete "$prepared_gallery/$directory/" "$gallery_root/$directory/"
  fi
done

for file in index.html original-view.html USER-GUIDE.html README.txt; do
  if [[ -f "$prepared_gallery/$file" ]]; then
    cp -p "$prepared_gallery/$file" "$gallery_root/$file"
  fi
done

cp -p "$windows_package" "$app_root/MemoryLane-Windows-x64.zip"
cp -p "$project_root/windows/Start Memory Lane from USB.bat" "$app_root/"
cp -p "$project_root/windows/START HERE.txt" "$app_root/"
cp -p "$project_root/windows/VERSION.txt" "$app_root/"

echo ""
echo "Memory Lane assembled successfully."
echo "Albums: $album_count in $photo_root"
echo "App:    $app_root"
echo ""
echo "The original photographs were not copied into the app folder."
