"""Command-line builder for a portable, file:// compatible photo gallery."""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import re
from shutil import copy2
import sys
from urllib.parse import quote

from PIL import UnidentifiedImageError

from .ingest import (
    album_year_range,
    content_hash,
    discover_images,
    generate_derivatives,
    stable_image_id,
)
from .metadata import match_records_to_folders, read_album_records


PROJECT_ROOT = Path(__file__).resolve().parent.parent
WEB_DIR = PROJECT_ROOT / "web"


def _load_state(path: Path) -> dict:
    if not path.exists():
        return {"version": 1, "files": {}}
    try:
        state = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"version": 1, "files": {}}
    return state if state.get("version") == 1 else {"version": 1, "files": {}}


def _asset_url(path: Path) -> str:
    return quote(path.as_posix(), safe="/()._-~")


def _display_title(folder: str) -> str:
    title = folder.replace("_", " ").replace(",", ".")
    title = re.sub(r"(?<=\d)(?=[A-Za-z])", " ", title)
    title = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", title)
    return title


def _write_json_atomic(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temporary.replace(path)


def _process_source(task: tuple) -> tuple[dict | None, dict | None, bool, str | None]:
    source, cached, output_dir, copy_originals = task
    relative_key = source.relative_path.as_posix()
    unchanged = bool(
        cached
        and cached.get("sha256")
        and cached.get("size") == source.size
        and cached.get("mtime_ns") == source.mtime_ns
    )
    file_hash = cached["sha256"] if unchanged else content_hash(source.path)
    image_id = stable_image_id(file_hash, relative_key)
    thumbnail_rel = Path("thumbnails") / f"{image_id}.webp"
    preview_rel = Path("previews") / f"{image_id}.jpg"
    thumbnail_path = output_dir / thumbnail_rel
    preview_path = output_dir / preview_rel

    derivative_metadata = cached.get("image", {}) if unchanged else {}
    reused = unchanged and thumbnail_path.exists() and preview_path.exists()
    if not reused:
        try:
            derivative_metadata = generate_derivatives(
                source.path, thumbnail_path, preview_path
            )
        except (OSError, ValueError, UnidentifiedImageError) as error:
            return None, None, False, f"Skipped {relative_key}: {error}"

    original_rel = Path("photos") / source.relative_path
    if copy_originals:
        destination = output_dir / original_rel
        if not (
            unchanged
            and destination.exists()
            and destination.stat().st_size == source.size
        ):
            destination.parent.mkdir(parents=True, exist_ok=True)
            copy2(source.path, destination)

    album_start, album_end = album_year_range(source.album_folder)
    record = {
        "id": image_id,
        "album_id": source.album_folder,
        "filename": source.path.name,
        "relative_source_path": relative_key,
        "original": _asset_url(original_rel) if copy_originals else None,
        "thumbnail": _asset_url(thumbnail_rel),
        "preview": _asset_url(preview_rel),
        "width": derivative_metadata.get("width"),
        "height": derivative_metadata.get("height"),
        "exif_date": derivative_metadata.get("exif_date"),
        "date_start": album_start,
        "date_end": album_end,
        "sha256": file_hash,
    }
    state_entry = {
        "size": source.size,
        "mtime_ns": source.mtime_ns,
        "sha256": file_hash,
        "id": image_id,
        "image": derivative_metadata,
    }
    return record, state_entry, reused, None


def build_archive(
    input_dir: Path,
    output_dir: Path,
    album_filters: list[str] | None = None,
    copy_originals: bool = True,
    workers: int = 1,
) -> dict:
    input_dir = input_dir.resolve()
    output_dir = output_dir.resolve()
    if input_dir == output_dir or input_dir in output_dir.parents:
        raise ValueError("Output must not be the input directory or a directory inside it")
    if not input_dir.is_dir():
        raise ValueError(f"Input directory does not exist: {input_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    assets_dir = output_dir / "assets"
    assets_dir.mkdir(exist_ok=True)

    discovered = discover_images(input_dir)
    if album_filters:
        needles = [value.casefold() for value in album_filters]
        discovered = [
            image
            for image in discovered
            if any(needle in image.album_folder.casefold() for needle in needles)
        ]
    if not discovered:
        raise ValueError("No supported images matched the requested input and album filters")

    folders = sorted({image.album_folder for image in discovered}, key=str.casefold)
    spreadsheet = input_dir / "Album Contents.xlsx"
    curated_records = read_album_records(spreadsheet) if spreadsheet.exists() else []
    curated_by_folder = match_records_to_folders(curated_records, folders)
    counts = Counter(image.album_folder for image in discovered)

    albums: list[dict] = []
    for folder in folders:
        record = curated_by_folder.get(folder)
        date_start, date_end = album_year_range(folder)
        albums.append(
            {
                "id": folder,
                "folder": folder,
                "title": _display_title(record.years if record else folder),
                "date_start": date_start,
                "date_end": date_end,
                "description": record.description if record else "",
                "album_number": record.number if record else None,
                "expected_count": record.expected_count if record else None,
                "actual_count": counts[folder],
            }
        )

    state_path = output_dir / ".build-state.json"
    state = _load_state(state_path)
    photo_records: list[dict] = []
    warnings: list[str] = []
    reused = 0

    total = len(discovered)
    tasks = (
        (source, state["files"].get(source.relative_path.as_posix()), output_dir, copy_originals)
        for source in discovered
    )
    with ThreadPoolExecutor(max_workers=workers) as executor:
        results = executor.map(_process_source, tasks)
        for position, (source, result) in enumerate(zip(discovered, results), start=1):
            record, state_entry, was_reused, warning = result
            if warning:
                warnings.append(warning)
            elif record is not None and state_entry is not None:
                photo_records.append(record)
                state["files"][source.relative_path.as_posix()] = state_entry
                reused += int(was_reused)
            if position % 250 == 0:
                _write_json_atomic(state_path, state)
            if position == 1 or position % 25 == 0 or position == total:
                print(f"Processed {position}/{total} images", flush=True)

    manifest = {
        "version": 1,
        "title": "Family Photo Archive",
        "albums": albums,
        "photos": photo_records,
        "warnings": warnings,
    }
    manifest_text = "window.FAMILY_PHOTO_DATA = " + json.dumps(
        manifest, ensure_ascii=False, separators=(",", ":")
    ) + ";\n"
    (assets_dir / "manifest.js").write_text(manifest_text, encoding="utf-8")
    _write_json_atomic(state_path, state)
    copy2(WEB_DIR / "index.html", output_dir / "index.html")
    copy2(WEB_DIR / "USER-GUIDE.html", output_dir / "USER-GUIDE.html")
    copy2(WEB_DIR / "gallery.css", assets_dir / "gallery.css")
    copy2(WEB_DIR / "gallery.js", assets_dir / "gallery.js")
    copy2(WEB_DIR / "search-examples.js", assets_dir / "search-examples.js")
    (output_dir / "README.txt").write_text(
        "Family Photo Archive\n\n"
        "Use the Memory Lane launcher to open the gallery. No internet connection is required.\n"
        "Right-click an enlarged photograph to use the browser's Copy image or Save image as command.\n",
        encoding="utf-8",
    )

    print(
        f"Built {len(photo_records)} photos in {len(albums)} albums "
        f"({reused} unchanged previews reused; {len(warnings)} warnings)."
    )
    return manifest


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--input", required=True, type=Path, help="Source photo directory")
    result.add_argument("--output", required=True, type=Path, help="Archive output directory")
    result.add_argument(
        "--album",
        action="append",
        default=[],
        help="Build only folders containing this text; may be repeated",
    )
    result.add_argument(
        "--no-copy-originals",
        action="store_true",
        help="Omit original files (useful for a quick development build)",
    )
    result.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of parallel image workers (default: 4)",
    )
    return result


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if args.workers < 1:
        print("Error: --workers must be positive", file=sys.stderr)
        return 2
    try:
        build_archive(
            args.input,
            args.output,
            album_filters=args.album,
            copy_originals=not args.no_copy_originals,
            workers=args.workers,
        )
    except (OSError, ValueError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
