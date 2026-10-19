"""Generate an incremental OpenCLIP image-embedding index for an archive."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from urllib.parse import unquote

import numpy as np
from PIL import Image, ImageOps

from .archive_data import read_gallery_manifest
from .browser_index import write_browser_index
from .clip_model import load_clip, model_identity, preserve_portable_weights
from .text_model import write_text_model


def _read_vector_files(
    metadata_path: Path, embeddings_path: Path, identity: dict
) -> dict[tuple[str, str], np.ndarray]:
    if not metadata_path.exists() or not embeddings_path.exists():
        return {}
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("model") != identity:
            return {}
        embeddings = np.load(embeddings_path, allow_pickle=False)
        if len(metadata.get("items", [])) != embeddings.shape[0]:
            return {}
    except (OSError, ValueError, json.JSONDecodeError):
        return {}
    return {
        (item["id"], item["sha256"]): embeddings[index]
        for index, item in enumerate(metadata["items"])
    }


def _existing_vectors(search_dir: Path, identity: dict) -> dict[tuple[str, str], np.ndarray]:
    vectors = _read_vector_files(
        search_dir / "index.json", search_dir / "embeddings.npy", identity
    )
    vectors.update(
        _read_vector_files(
            search_dir / ".partial-index.json",
            search_dir / ".partial-embeddings.npy",
            identity,
        )
    )
    return vectors


def _save_array_atomic(path: Path, array: np.ndarray) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        np.save(stream, array, allow_pickle=False)
    temporary.replace(path)


def _save_partial(
    search_dir: Path, identity: dict, photos: list[dict], vectors: list[np.ndarray | None]
) -> None:
    completed = [index for index, vector in enumerate(vectors) if vector is not None]
    partial_vectors = np.stack([vectors[index] for index in completed]).astype(
        np.float32, copy=False
    )
    partial_items = [
        {"id": photos[index]["id"], "sha256": photos[index]["sha256"]}
        for index in completed
    ]
    _save_array_atomic(search_dir / ".partial-embeddings.npy", partial_vectors)
    metadata_path = search_dir / ".partial-index.json"
    temporary = metadata_path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps({"model": identity, "items": partial_items}), encoding="utf-8"
    )
    temporary.replace(metadata_path)


def _photo_path(archive: Path, photo: dict) -> Path:
    if not photo.get("original"):
        raise ValueError(f"Photo {photo['id']} has no archived original")
    return archive / Path(unquote(photo["original"]))


def _encode_batch(model, preprocess, paths: list[Path]) -> np.ndarray:
    import torch

    tensors = []
    for path in paths:
        with Image.open(path) as opened:
            corrected = ImageOps.exif_transpose(opened)
            tensors.append(preprocess(corrected.convert("RGB")))
    batch = torch.stack(tensors)
    with torch.inference_mode():
        features = model.encode_image(batch)
        features = features / features.norm(dim=-1, keepdim=True)
    return features.cpu().numpy().astype(np.float32, copy=False)


def build_embeddings(archive: Path, batch_size: int = 8) -> dict:
    archive = archive.resolve()
    manifest = read_gallery_manifest(archive)
    search_dir = archive / "search"
    search_dir.mkdir(parents=True, exist_ok=True)

    print("Loading OpenCLIP model...", flush=True)
    cache_dir = archive / ".model-cache"
    model, preprocess, tokenizer, preprocess_config = load_clip(cache_dir)
    preserve_portable_weights(cache_dir)
    identity = model_identity()
    existing = _existing_vectors(search_dir, identity)
    existing_by_sha = {sha256: vector for (_, sha256), vector in existing.items()}

    photos = manifest.get("photos", [])
    vectors: list[np.ndarray | None] = [None] * len(photos)
    pending: list[tuple[int, Path]] = []
    reused = 0
    for index, photo in enumerate(photos):
        cached = existing.get((photo["id"], photo["sha256"]))
        if cached is None:
            # A renamed file has a new gallery id but identical pixels and can
            # safely reuse the same CLIP image vector.
            cached = existing_by_sha.get(photo["sha256"])
        if cached is not None:
            vectors[index] = cached
            reused += 1
        else:
            pending.append((index, _photo_path(archive, photo)))

    total_new = len(pending)
    for offset in range(0, total_new, batch_size):
        batch_entries = pending[offset : offset + batch_size]
        encoded = _encode_batch(model, preprocess, [entry[1] for entry in batch_entries])
        for (index, _), vector in zip(batch_entries, encoded, strict=True):
            vectors[index] = vector
        completed_new = min(offset + batch_size, total_new)
        print(f"Embedded {completed_new}/{total_new} new images", flush=True)
        if completed_new % 256 < batch_size or completed_new == total_new:
            _save_partial(search_dir, identity, photos, vectors)
            print(f"Checkpointed {reused + completed_new}/{len(photos)} embeddings", flush=True)

    if vectors:
        embeddings = np.stack(vectors).astype(np.float32, copy=False)
    else:
        embeddings = np.empty((0, 0), dtype=np.float32)
    items = [
        {
            key: photo.get(key)
            for key in (
                "id", "sha256", "album_id", "filename", "original", "preview",
                "thumbnail", "date_start", "date_end",
            )
        }
        for photo in photos
    ]
    metadata = {
        "version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model": identity,
        "preprocessing": preprocess_config,
        "normalized": True,
        "dtype": "float32",
        "embedding_dimension": int(embeddings.shape[1]) if embeddings.size else 0,
        "items": items,
    }
    _save_array_atomic(search_dir / "embeddings.npy", embeddings)
    metadata_path = search_dir / "index.json"
    metadata_temporary = metadata_path.with_suffix(".json.tmp")
    metadata_temporary.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    metadata_temporary.replace(metadata_path)
    for partial_path in (
        search_dir / ".partial-index.json",
        search_dir / ".partial-embeddings.npy",
    ):
        partial_path.unlink(missing_ok=True)
    write_text_model(archive, model, tokenizer, identity)
    write_browser_index(archive)
    print(
        f"Indexed {len(items)} photos ({reused} reused, {total_new} newly embedded) "
        f"into {search_dir}."
    )
    return metadata


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--archive", required=True, type=Path)
    result.add_argument("--batch-size", type=int, default=8)
    return result


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if args.batch_size < 1:
        print("Error: --batch-size must be positive", file=sys.stderr)
        return 2
    try:
        build_embeddings(args.archive, args.batch_size)
    except (OSError, RuntimeError, ValueError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
