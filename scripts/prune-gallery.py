#!/usr/bin/env python3
"""Remove cached gallery records whose source photos no longer exist."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from urllib.parse import unquote

import numpy as np


MANIFEST_PREFIX = "window.FAMILY_PHOTO_DATA = "


def write_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    temporary.replace(path)


def write_array(path: Path, value: np.ndarray) -> None:
    temporary = path.with_suffix(".tmp.npy")
    np.save(temporary, value, allow_pickle=False)
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("archive", type=Path)
    parser.add_argument("photo_root", type=Path)
    parser.add_argument(
        "--preserve-album",
        action="append",
        default=[],
        help="Keep a cached album whose source files were renamed but verified separately",
    )
    args = parser.parse_args()

    archive = args.archive.resolve()
    photo_root = args.photo_root.resolve()
    manifest_path = archive / "assets" / "manifest.js"
    source = manifest_path.read_text(encoding="utf-8").strip()
    if not source.startswith(MANIFEST_PREFIX) or not source.endswith(";"):
        raise ValueError(f"Unrecognized manifest: {manifest_path}")
    manifest = json.loads(source[len(MANIFEST_PREFIX) : -1])

    original_photos = manifest["photos"]
    kept_photos = []
    removed_photos = []
    for photo in original_photos:
        relative = Path(photo["relative_source_path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"Unsafe source path: {relative}")
        exists = (photo_root / relative).is_file()
        preserve_album = photo["album_id"] in args.preserve_album
        (kept_photos if exists or preserve_album else removed_photos).append(photo)

    kept_ids = {photo["id"] for photo in kept_photos}
    counts: dict[str, int] = {}
    for photo in kept_photos:
        counts[photo["album_id"]] = counts.get(photo["album_id"], 0) + 1
    for album in manifest["albums"]:
        album["actual_count"] = counts.get(album["id"], 0)
    manifest["photos"] = kept_photos

    manifest_temporary = manifest_path.with_suffix(".js.tmp")
    manifest_temporary.write_text(
        MANIFEST_PREFIX
        + json.dumps(manifest, ensure_ascii=False, separators=(",", ":"))
        + ";\n",
        encoding="utf-8",
    )
    manifest_temporary.replace(manifest_path)

    search_dir = archive / "search"
    index_path = search_dir / "index.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    embeddings_path = search_dir / "embeddings.npy"
    embeddings = np.load(embeddings_path, allow_pickle=False)
    if len(index["items"]) != embeddings.shape[0]:
        raise ValueError("Semantic metadata and embeddings are misaligned")
    semantic_mask = np.asarray(
        [item["id"] in kept_ids for item in index["items"]], dtype=bool
    )
    index["items"] = [
        item for item, keep in zip(index["items"], semantic_mask, strict=True) if keep
    ]
    write_array(embeddings_path, embeddings[semantic_mask])
    write_json(index_path, index)

    faces_path = search_dir / "faces.json"
    faces = json.loads(faces_path.read_text(encoding="utf-8"))
    face_embeddings_path = search_dir / "faces.npy"
    face_embeddings = np.load(face_embeddings_path, allow_pickle=False)
    if len(faces["items"]) != face_embeddings.shape[0]:
        raise ValueError("Face metadata and embeddings are misaligned")
    face_mask = np.asarray(
        [item["photo_id"] in kept_ids for item in faces["items"]], dtype=bool
    )
    faces["items"] = [
        item for item, keep in zip(faces["items"], face_mask, strict=True) if keep
    ]
    faces["processed"] = [
        item for item in faces["processed"] if item["id"] in kept_ids
    ]
    faces["photo_count"] = len(kept_photos)
    faces["face_count"] = len(faces["items"])
    write_array(face_embeddings_path, face_embeddings[face_mask])
    write_json(faces_path, faces)

    for photo in removed_photos:
        for field in ("preview", "thumbnail"):
            relative = photo.get(field)
            if relative:
                (archive / unquote(relative)).unlink(missing_ok=True)

    print(f"Kept {len(kept_photos):,} photos; removed {len(removed_photos):,} stale records.")
    print(f"Kept {len(faces['items']):,} faces.")


if __name__ == "__main__":
    main()
