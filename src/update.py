"""Add new photos to Memory Lane and refresh the gallery on the photo USB.

One run:
  1. pulls new photos from the shared Google Drive folder with rclone (optional);
  2. compares them with the USB and prints a plain-English summary;
  3. after approval, copies them into the USB's Family Photos folders,
     converting iPhone HEIC photos to JPEG;
  4. rebuilds only what changed in a temporary staging gallery;
  5. validates the result, then updates the local cache and the USB gallery.

Photos are only ever added or replaced. Nothing is deleted from Family Photos.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass, field
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from urllib.parse import unquote
import urllib.request

from .archive_data import read_face_index, read_gallery_manifest, read_search_index
from .build import SPREADSHEET_NAME, build_albums
from .ingest import SUPPORTED_EXTENSIONS, album_year_range, content_hash, discover_images


PROJECT_ROOT = Path(__file__).resolve().parent.parent
LOCAL_ARCHIVE = PROJECT_ROOT / "FamilyPhotoArchive"
WORK_DIR = Path.home() / ".cache" / "memory-lane"
HEIC_EXTENSIONS = {".heic", ".heif"}
LEDGER_NAME = "import-ledger.json"
SEARCH_DATA_PREFIX = "window.MEMORY_LANE_SEARCH_DATA = "
FACE_MODEL_URLS = {
    "face_detection_yunet_2023mar.onnx": "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx",
    "face_recognition_sface_2021dec.onnx": "https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx",
}
# Generated gallery material. Everything else in the archive (local photo
# copies, gold labels, evaluation pages) is never touched by an update.
GENERATED_DIRS = ("thumbnails", "viewing", "search", "assets")
GENERATED_FILES = ("index.html", "original-view.html", "USER-GUIDE.html", "README.txt")
LAST_ASSETS = ("manifest.js", "search-data.js")


class UpdateError(Exception):
    pass


@dataclass
class Plan:
    add: list[tuple[Path, Path]] = field(default_factory=list)
    replace: list[tuple[Path, Path]] = field(default_factory=list)
    heic_hashes: dict[str, str] = field(default_factory=dict)
    spreadsheet: Path | None = None
    ignored: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    new_albums: set[str] = field(default_factory=set)
    gallery_missing: int = 0
    gallery_changed: int = 0
    gallery_removed: int = 0
    albums_changed: bool = False

    @property
    def copies_files(self) -> bool:
        return bool(self.add or self.replace or self.spreadsheet)

    @property
    def needs_rebuild(self) -> bool:
        return bool(
            self.copies_files
            or self.gallery_missing
            or self.gallery_changed
            or self.gallery_removed
            or self.albums_changed
        )


def _plural(number: int, word: str) -> str:
    return f"{number:,} {word}{'' if number == 1 else 's'}"


def _sha256(path: Path) -> str:
    return content_hash(path)


def find_usb(explicit: Path | None) -> Path:
    if explicit is not None:
        candidates = [explicit]
    else:
        user = os.environ.get("USER", "")
        candidates = sorted(
            path
            for base in (Path("/run/media") / user, Path("/media") / user)
            if base.is_dir()
            for path in base.iterdir()
        )
    found = [
        path
        for path in candidates
        if (path / "Family Photos").is_dir() and (path / "Memory Lane/Gallery/index.html").is_file()
    ]
    if explicit is not None and not found:
        raise UpdateError(f"{explicit} is not a Memory Lane USB (needs Family Photos and Memory Lane/Gallery)")
    if not found:
        raise UpdateError("No Memory Lane USB found. Plug in the photo USB, or pass --usb.")
    if len(found) > 1:
        names = ", ".join(str(path) for path in found)
        raise UpdateError(f"More than one Memory Lane USB is connected ({names}). Pass --usb.")
    return found[0]


def pull(remote: str, incoming: Path, shared_with_me: bool) -> None:
    if shutil.which("rclone") is None:
        raise UpdateError("rclone is not installed")
    remote_name = remote.split(":", 1)[0] + ":"
    remotes = subprocess.run(["rclone", "listremotes"], capture_output=True, text=True).stdout.split()
    if remote_name not in remotes:
        raise UpdateError(
            f"rclone has no remote called {remote_name[:-1]}. Run `rclone config` and add a "
            "read-only Google Drive remote with that name."
        )
    incoming.mkdir(parents=True, exist_ok=True)
    command = ["rclone", "copy", remote, str(incoming), "--stats-one-line", "--stats", "10s"]
    if shared_with_me:
        command.append("--drive-shared-with-me")
    print(f"Fetching new photos from {remote} ...", flush=True)
    if subprocess.run(command).returncode != 0:
        raise UpdateError("rclone could not fetch the Google Drive folder")


def _read_json(path: Path, default):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return default


def _album_summary(albums: list[dict]) -> dict[str, tuple]:
    return {album["id"]: (album["title"], album["description"]) for album in albums}


def make_plan(incoming: Path | None, photo_root: Path, archive: Path, ledger: dict) -> Plan:
    plan = Plan()
    planned_targets: dict[str, str] = {}

    if incoming is not None and incoming.is_dir():
        for path in sorted(incoming.rglob("*"), key=lambda item: item.as_posix().casefold()):
            if not path.is_file() or any(part.startswith(".") for part in path.relative_to(incoming).parts):
                continue
            relative = path.relative_to(incoming)
            if relative.as_posix() == SPREADSHEET_NAME:
                current = photo_root / SPREADSHEET_NAME
                if not current.is_file() or _sha256(current) != _sha256(path):
                    plan.spreadsheet = path
                continue
            suffix = path.suffix.casefold()
            if suffix in HEIC_EXTENSIONS:
                target = relative.with_suffix(".jpg")
            elif suffix in SUPPORTED_EXTENSIONS:
                target = relative
            else:
                plan.ignored.append(relative.as_posix())
                continue

            key = target.as_posix()
            if key.casefold() in planned_targets:
                plan.warnings.append(
                    f"Skipped {relative.as_posix()}: same name as {planned_targets[key.casefold()]}."
                )
                continue
            planned_targets[key.casefold()] = relative.as_posix()

            source_hash = _sha256(path)
            destination = photo_root / target
            if suffix in HEIC_EXTENSIONS:
                if destination.is_file() and ledger.get(relative.as_posix()) == source_hash:
                    continue
                plan.heic_hashes[relative.as_posix()] = source_hash
            elif destination.is_file() and (
                destination.stat().st_size == path.stat().st_size
                and _sha256(destination) == source_hash
            ):
                continue
            (plan.replace if destination.is_file() else plan.add).append((path, target))

            if len(target.parts) == 1:
                plan.warnings.append(
                    f"{relative.as_posix()} is not inside an album folder, so it will appear as \"Unsorted\"."
                )
            else:
                album = target.parts[0]
                if not (photo_root / album).is_dir() and album not in plan.new_albums:
                    plan.new_albums.add(album)
                    if album_year_range(album)[0] is None:
                        plan.warnings.append(
                            f"New album \"{album}\" has no year in its name, so it will not appear under a decade."
                        )

    # Photos already on the USB that the gallery does not reflect yet, such as
    # photos copied by an earlier run that stopped before rebuilding.
    manifest = read_gallery_manifest(archive)
    state = _read_json(archive / ".build-state.json", {}).get("files", {})
    on_usb = discover_images(photo_root)
    in_gallery = {photo["relative_source_path"] for photo in manifest["photos"]}
    usb_paths = set()
    for image in on_usb:
        key = image.relative_path.as_posix()
        usb_paths.add(key)
        cached = state.get(key)
        if key not in in_gallery or cached is None:
            plan.gallery_missing += 1
        elif cached.get("size") != image.size:
            plan.gallery_changed += 1
        elif cached.get("mtime_ns") != image.mtime_ns and content_hash(image.path) != cached.get("sha256"):
            plan.gallery_changed += 1
    plan.gallery_removed = len(in_gallery - usb_paths)
    expected_albums = build_albums(photo_root / SPREADSHEET_NAME, on_usb)
    plan.albums_changed = _album_summary(expected_albums) != _album_summary(
        [album for album in manifest["albums"] if album["id"] in {a["id"] for a in expected_albums}]
    )
    return plan


def describe(plan: Plan) -> str:
    lines: list[str] = []
    if plan.add or plan.replace or plan.spreadsheet:
        lines.append("From the Google Drive folder:")
        if plan.add:
            by_album = Counter(target.parts[0] if len(target.parts) > 1 else "Unsorted" for _, target in plan.add)
            details = ", ".join(
                f"{album}{' (new album)' if album in plan.new_albums else ''}: {count}"
                for album, count in sorted(by_album.items())
            )
            lines.append(f"  {_plural(len(plan.add), 'photo')} added ({details})")
        if plan.replace:
            lines.append(f"  {_plural(len(plan.replace), 'photo')} changed")
        if plan.heic_hashes:
            lines.append(f"  {_plural(len(plan.heic_hashes), 'iPhone photo')} will be converted to JPEG")
        if plan.spreadsheet:
            lines.append("  Album descriptions updated (Album Contents.xlsx)")
        if plan.ignored:
            shown = ", ".join(plan.ignored[:5]) + (" ..." if len(plan.ignored) > 5 else "")
            lines.append(
                f"  Ignored {_plural(len(plan.ignored), 'file')} that "
                f"{'is not a photo' if len(plan.ignored) == 1 else 'are not photos'}: {shown}"
            )
    if plan.gallery_missing or plan.gallery_changed or plan.gallery_removed:
        lines.append("Already on the USB, not yet in the gallery:")
        for count, word in (
            (plan.gallery_missing, "added"),
            (plan.gallery_changed, "changed"),
            (plan.gallery_removed, "removed"),
        ):
            if count:
                lines.append(f"  {_plural(count, 'photo')} {word}")
    if plan.albums_changed and not plan.spreadsheet:
        lines.append("Album descriptions on the USB differ from the gallery.")
    if plan.warnings:
        lines.append("Check:")
        lines.extend(f"  {warning}" for warning in plan.warnings)
    if not plan.needs_rebuild:
        lines.append("Nothing new. Memory Lane is up to date.")
    return "\n".join(lines)


def _copy_into_place(source: Path, destination: Path, convert: bool) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(f".{destination.name}.partial")
    if convert:
        convert_heic(source, partial)
    else:
        # copyfile, not copy2: exFAT drives reject permission changes.
        shutil.copyfile(source, partial)
    stat = source.stat()
    os.utime(partial, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    os.replace(partial, destination)


def convert_heic(source: Path, destination: Path) -> None:
    try:
        import pillow_heif
    except ImportError as error:
        raise UpdateError("Install pillow-heif to import iPhone HEIC photos") from error
    from PIL import Image, ImageOps

    pillow_heif.register_heif_opener()
    with Image.open(source) as opened:
        upright = ImageOps.exif_transpose(opened)
        exif = upright.getexif()
        exif.pop(274, None)  # orientation is already applied
        upright.convert("RGB").save(destination, "JPEG", quality=92, exif=exif.tobytes())


def apply(plan: Plan, photo_root: Path, ledger_path: Path, ledger: dict) -> None:
    items = plan.add + plan.replace
    for position, (source, target) in enumerate(items, start=1):
        convert = source.suffix.casefold() in HEIC_EXTENSIONS
        _copy_into_place(source, photo_root / target, convert)
        if position % 25 == 0 or position == len(items):
            print(f"Copied {position}/{len(items)} photos to the USB", flush=True)
    if plan.spreadsheet:
        _copy_into_place(plan.spreadsheet, photo_root / SPREADSHEET_NAME, convert=False)
    if plan.heic_hashes:
        ledger.update(plan.heic_hashes)
        temporary = ledger_path.with_name(ledger_path.name + ".tmp")
        temporary.write_text(json.dumps(ledger, indent=2, sort_keys=True), encoding="utf-8")
        os.replace(temporary, ledger_path)
    os.sync()


def face_models(archive: Path) -> tuple[Path, Path]:
    directory = archive / ".face-models"
    directory.mkdir(exist_ok=True)
    expected = {}
    faces_json = archive / "search" / "faces.json"
    if faces_json.is_file():
        model = _read_json(faces_json, {}).get("model", {})
        expected = {
            model.get("detector"): model.get("detector_sha256"),
            model.get("recognizer"): model.get("recognizer_sha256"),
        }
    for name, url in FACE_MODEL_URLS.items():
        path = directory / name
        if not path.is_file():
            print(f"Downloading face model {name} ...", flush=True)
            temporary = path.with_name(name + ".partial")
            with urllib.request.urlopen(url, timeout=120) as response, temporary.open("wb") as stream:
                shutil.copyfileobj(response, stream)
            os.replace(temporary, path)
        wanted = expected.get(name)
        if wanted and hashlib.sha256(path.read_bytes()).hexdigest() != wanted:
            raise UpdateError(f"Face model {name} does not match the one used for this index")
    return tuple(directory / name for name in FACE_MODEL_URLS)


def prepare_stage(archive: Path, photo_root: Path, stage: Path) -> None:
    if stage.exists():
        shutil.rmtree(stage)
    stage.mkdir(parents=True)
    for name in GENERATED_DIRS:
        source = archive / name
        if source.is_dir():
            # The large model folder is read-only during a build, so link it.
            shutil.copytree(source, stage / name, ignore=shutil.ignore_patterns("model") if name == "search" else None)
    (stage / "search").mkdir(exist_ok=True)
    if (archive / "search" / "model").is_dir():
        (stage / "search" / "model").symlink_to((archive / "search" / "model").resolve())
    for name in (*GENERATED_FILES, ".build-state.json"):
        if (archive / name).is_file():
            shutil.copy2(archive / name, stage / name)
    if (archive / ".model-cache").is_dir():
        (stage / ".model-cache").symlink_to((archive / ".model-cache").resolve())
    (stage / "photos").symlink_to(photo_root.resolve())


def rebuild(stage: Path, photo_root: Path, archive: Path) -> dict:
    from .build import build_archive
    from .embed import build_embeddings
    from .faces import build_face_index

    before = {photo["id"] for photo in read_gallery_manifest(stage)["photos"]}
    print("\nUpdating thumbnails and the photo list ...", flush=True)
    manifest = build_archive(photo_root, stage, copy_originals=False, workers=1)
    new_photos = sum(photo["id"] not in before for photo in manifest["photos"])
    print("\nUpdating photo search ...", flush=True)
    build_embeddings(stage, batch_size=1)
    print("\nUpdating faces ...", flush=True)
    detector, recognizer = face_models(archive)
    build_face_index(stage, detector, recognizer)
    print("\nRemoving photos that are no longer on the USB ...", flush=True)
    result = subprocess.run(
        [sys.executable, str(PROJECT_ROOT / "scripts" / "prune-gallery.py"), str(stage), str(photo_root)]
    )
    if result.returncode != 0:
        raise UpdateError("Pruning the gallery failed")
    return {"new_photos": new_photos}


def _search_data(archive: Path) -> dict:
    source = (archive / "assets" / "search-data.js").read_text(encoding="utf-8").strip()
    return json.loads(source[len(SEARCH_DATA_PREFIX) : -1])


def validate(stage: Path, photo_root: Path) -> list[str]:
    problems: list[str] = []
    manifest = read_gallery_manifest(stage)
    photos = manifest["photos"]
    ids = [photo["id"] for photo in photos]
    on_usb = {image.relative_path.as_posix() for image in discover_images(photo_root)}
    in_manifest = {photo["relative_source_path"] for photo in photos}
    if in_manifest != on_usb:
        problems.append(
            f"Gallery lists {len(in_manifest):,} photos but the USB has {len(on_usb):,} "
            f"({len(on_usb - in_manifest)} missing, {len(in_manifest - on_usb)} extra)"
        )
    if manifest.get("warnings"):
        problems.extend(f"Build warning: {warning}" for warning in manifest["warnings"])
    if sum(album["actual_count"] for album in manifest["albums"]) != len(photos):
        problems.append("Album counts do not add up to the photo count")
    for field_name, directory_name in (("thumbnail", "thumbnails"), ("viewing", "viewing")):
        expected = {Path(unquote(photo[field_name])).name for photo in photos if photo.get(field_name)}
        directory = stage / directory_name
        present = {path.name for path in directory.iterdir()} if directory.is_dir() else set()
        if expected - present:
            problems.append(f"{len(expected - present)} {field_name} files are missing")
    for photo in photos:
        if not (photo_root / photo["relative_source_path"]).is_file():
            problems.append(f"Missing original: {photo['relative_source_path']}")
            break
    metadata, _ = read_search_index(stage)
    if [item["id"] for item in metadata["items"]] != ids:
        problems.append("Photo search index does not match the photo list")
    faces, _ = read_face_index(stage)
    if [item["id"] for item in faces["processed"]] != ids:
        problems.append("Face index does not match the photo list")
    search_data = _search_data(stage)
    for key in ("similar", "faces", "text"):
        if not search_data.get(key):
            problems.append(f"Browser search data is missing {key}")
    if search_data.get("similar") and search_data["similar"]["count"] != len(photos):
        problems.append("Browser photo search does not cover every photo")
    return problems


def sync(stage: Path, destination: Path, usb: bool) -> None:
    # exFAT keeps no permissions or symlinks, and its timestamps are coarse.
    options = ["-rt", "--modify-window=2"] if usb else ["-a"]
    for name in GENERATED_DIRS:
        source = stage / name
        if not source.is_dir():
            continue
        (destination / name).mkdir(parents=True, exist_ok=True)
        excludes = ["--exclude=/model"] if name == "search" else []
        if name == "assets":
            excludes += [f"--exclude=/{asset}" for asset in LAST_ASSETS]
        command = ["rsync", *options, "--delete", *excludes, f"{source}/", f"{destination / name}/"]
        if subprocess.run(command).returncode != 0:
            raise UpdateError(f"Copying {name} to {destination} failed")
    for name in GENERATED_FILES + (() if usb else (".build-state.json",)):
        if (stage / name).is_file():
            _copy_into_place(stage / name, destination / name, convert=False)
    # The photo list and search pointer go last, so an interrupted copy never
    # points the gallery at files that are not there yet.
    for asset in LAST_ASSETS:
        _copy_into_place(stage / "assets" / asset, destination / "assets" / asset, convert=False)
    os.sync()


def verify_copy(stage: Path, destination: Path) -> list[str]:
    problems = []
    for relative in ("assets/manifest.js", "assets/search-data.js", "assets/search-similar.js", "search/index.json"):
        if _sha256(stage / relative) != _sha256(destination / relative):
            problems.append(f"{relative} on {destination} does not match")
    thumbnails = len(list((stage / "thumbnails").iterdir()))
    if len(list((destination / "thumbnails").iterdir())) != thumbnails:
        problems.append(f"Thumbnail count on {destination} does not match")
    return problems


def check_in_step(archive: Path, gallery: Path) -> None:
    """Refuse to run when the USB gallery and the local copy have drifted apart."""
    for relative in ("assets/manifest.js", "assets/search-data.js"):
        if _sha256(archive / relative) != _sha256(gallery / relative):
            raise UpdateError(
                f"The USB gallery and the local copy differ ({relative}). "
                "Bring them back in step before updating."
            )


def run(args: argparse.Namespace) -> int:
    archive = args.archive.resolve()
    usb = find_usb(args.usb)
    photo_root = usb / "Family Photos"
    gallery = usb / "Memory Lane" / "Gallery"
    print(f"Photo USB: {usb}")
    check_in_step(archive, gallery)

    incoming = args.source
    if not args.remote and not args.source:
        print("No Google Drive folder given; checking the USB for photos added by hand.")
    if args.remote:
        incoming = args.work / "incoming"
        pull(args.remote, incoming, args.shared_with_me)

    ledger_path = gallery / LEDGER_NAME
    ledger = _read_json(ledger_path, {})
    print("Comparing photos ...", flush=True)
    plan = make_plan(incoming, photo_root, archive, ledger)
    print()
    print(describe(plan))
    if not plan.needs_rebuild or args.dry_run:
        return 0
    if not args.yes:
        answer = input("\nUpdate Memory Lane on the USB? [y/N] ").strip().casefold()
        if answer not in {"y", "yes"}:
            print("Nothing was changed.")
            return 0

    if plan.copies_files:
        apply(plan, photo_root, ledger_path, ledger)
    stage = args.work / "stage"
    prepare_stage(archive, photo_root, stage)
    rebuild(stage, photo_root, archive)
    problems = validate(stage, photo_root)
    if problems:
        print("\nThe rebuilt gallery did not pass its checks, so nothing else was changed:")
        print("\n".join(f"  {problem}" for problem in problems))
        print(f"The staging copy is kept at {stage}.")
        return 1

    print("\nUpdating the local copy ...", flush=True)
    sync(stage, archive, usb=False)
    print("Updating the USB gallery ...", flush=True)
    sync(stage, gallery, usb=True)
    problems = verify_copy(stage, archive) + verify_copy(stage, gallery)
    if problems:
        print("\nCopying finished with problems:")
        print("\n".join(f"  {problem}" for problem in problems))
        return 1
    shutil.rmtree(stage)
    count = len(read_gallery_manifest(archive)["photos"])
    print(f"\nDone. Memory Lane now has {_plural(count, 'photo')}. Eject the USB before unplugging it.")
    return 0


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    result.add_argument("--remote", help='rclone path of the shared folder, e.g. "gdrive:New Photos"')
    result.add_argument("--shared-with-me", action="store_true", help="The folder is under Shared with me")
    result.add_argument("--source", type=Path, help="Use a local folder of new photos instead of rclone")
    result.add_argument("--usb", type=Path, help="Photo USB (found automatically when only one is connected)")
    result.add_argument("--archive", type=Path, default=LOCAL_ARCHIVE, help=argparse.SUPPRESS)
    result.add_argument("--work", type=Path, default=WORK_DIR, help=argparse.SUPPRESS)
    result.add_argument("--dry-run", action="store_true", help="Show the summary only")
    result.add_argument("--yes", action="store_true", help="Do not ask for approval")
    return result


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if args.remote and args.source:
        print("Error: use --remote or --source, not both", file=sys.stderr)
        return 2
    try:
        return run(args)
    except UpdateError as error:
        print(f"Error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
