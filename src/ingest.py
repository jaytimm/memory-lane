"""Image discovery, hashing, metadata extraction, and preview generation."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from hashlib import sha256
from pathlib import Path
import re

from PIL import Image, ImageOps


SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff"}
EXIF_DATE_TAGS = (36867, 36868, 306)  # original, digitized, modified


@dataclass(frozen=True)
class SourceImage:
    path: Path
    relative_path: Path
    album_folder: str
    size: int
    mtime_ns: int


def discover_images(input_dir: str | Path) -> list[SourceImage]:
    root = Path(input_dir).resolve()
    images: list[SourceImage] = []
    for path in root.rglob("*"):
        if not path.is_file() or path.suffix.casefold() not in SUPPORTED_EXTENSIONS:
            continue
        relative = path.relative_to(root)
        stat = path.stat()
        album = relative.parts[0] if len(relative.parts) > 1 else "Unsorted"
        images.append(SourceImage(path, relative, album, stat.st_size, stat.st_mtime_ns))
    return sorted(images, key=lambda item: item.relative_path.as_posix().casefold())


def content_hash(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def stable_image_id(file_hash: str, relative_path: str) -> str:
    """Create a content-led ID with a path suffix to distinguish exact duplicates."""
    path_hash = sha256(relative_path.encode("utf-8")).hexdigest()
    return f"{file_hash[:20]}-{path_hash[:8]}"


def album_year_range(folder_name: str) -> tuple[int | None, int | None]:
    start_match = re.search(r"(?:19|20)\d{2}", folder_name)
    if not start_match:
        return None, None
    start = int(start_match.group())
    end = start
    end_match = re.search(r"[-–][^/]*?[._](\d{2})(?:\D*)$", folder_name)
    if end_match:
        short_end = int(end_match.group(1))
        end = (start // 100) * 100 + short_end
        while end < start:
            end += 100
    return start, end


def extract_exif_date(image: Image.Image) -> str | None:
    try:
        exif = image.getexif()
    except Exception:
        return None
    for tag in EXIF_DATE_TAGS:
        raw = exif.get(tag)
        if not raw:
            continue
        try:
            return datetime.strptime(str(raw), "%Y:%m:%d %H:%M:%S").isoformat()
        except ValueError:
            continue
    return None


def _rgb(image: Image.Image) -> Image.Image:
    if image.mode in {"RGBA", "LA"} or "transparency" in image.info:
        rgba = image.convert("RGBA")
        background = Image.new("RGBA", rgba.size, "white")
        background.alpha_composite(rgba)
        return background.convert("RGB")
    return image.convert("RGB")


def generate_derivatives(
    source: Path,
    thumbnail_path: Path,
    preview_path: Path,
    thumbnail_size: int = 400,
    preview_size: int = 1800,
) -> dict:
    thumbnail_path.parent.mkdir(parents=True, exist_ok=True)
    preview_path.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(source) as opened:
        exif_date = extract_exif_date(opened)
        corrected = ImageOps.exif_transpose(opened)
        width, height = corrected.size

        thumbnail = _rgb(corrected.copy())
        thumbnail.thumbnail((thumbnail_size, thumbnail_size), Image.Resampling.LANCZOS)
        thumbnail.save(thumbnail_path, "WEBP", quality=82, method=6)

        preview = _rgb(corrected.copy())
        preview.thumbnail((preview_size, preview_size), Image.Resampling.LANCZOS)
        preview.save(preview_path, "JPEG", quality=88, optimize=True, progressive=True)

    return {"width": width, "height": height, "exif_date": exif_date}

