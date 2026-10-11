"""Build a compact face index for person search in a generated archive."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
from urllib.parse import unquote

import cv2
import numpy as np

from .archive_data import read_gallery_manifest
from .embed import _save_array_atomic


INDEX_VERSION = 1


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _model_identity(detector_model: Path, recognizer_model: Path) -> dict:
    return {
        "detector": detector_model.name,
        "detector_sha256": _file_sha256(detector_model),
        "recognizer": recognizer_model.name,
        "recognizer_sha256": _file_sha256(recognizer_model),
    }


def _resized(image: np.ndarray, max_side: int) -> tuple[np.ndarray, float]:
    height, width = image.shape[:2]
    scale = min(1.0, max_side / max(height, width))
    if scale == 1.0:
        return image, scale
    return (
        cv2.resize(
            image,
            (round(width * scale), round(height * scale)),
            interpolation=cv2.INTER_AREA,
        ),
        scale,
    )


def _read_cached_groups(search_dir: Path, identity: dict) -> tuple[set[tuple[str, str]], dict]:
    processed: set[tuple[str, str]] = set()
    groups: dict[tuple[str, str], list[tuple[dict, np.ndarray]]] = {}
    for stem in ("faces", ".partial-faces"):
        metadata_path = search_dir / f"{stem}.json"
        vectors_path = search_dir / f"{stem}.npy"
        if not metadata_path.exists() or not vectors_path.exists():
            continue
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            vectors = np.load(vectors_path, allow_pickle=False)
            if metadata.get("version") != INDEX_VERSION or metadata.get("model") != identity:
                continue
            if len(metadata.get("items", [])) != len(vectors):
                continue
        except (OSError, ValueError, json.JSONDecodeError):
            continue
        processed.update((item["id"], item["sha256"]) for item in metadata["processed"])
        loaded_groups: dict[tuple[str, str], list[tuple[dict, np.ndarray]]] = {}
        for item, vector in zip(metadata["items"], vectors, strict=True):
            key = (item["photo_id"], item["photo_sha256"])
            loaded_groups.setdefault(key, []).append((item, vector))
        groups.update(loaded_groups)
    return processed, groups


def _save(
    search_dir: Path,
    stem: str,
    identity: dict,
    processed: list[dict],
    items: list[dict],
    vectors: list[np.ndarray],
    settings: dict,
) -> None:
    matrix = np.stack(vectors).astype(np.float32, copy=False) if vectors else np.empty((0, 128), dtype=np.float32)
    _save_array_atomic(search_dir / f"{stem}.npy", matrix)
    metadata = {
        "version": INDEX_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model": identity,
        "settings": settings,
        "normalized": True,
        "dtype": "float32",
        "embedding_dimension": int(matrix.shape[1]),
        "photo_count": len(processed),
        "face_count": len(items),
        "processed": processed,
        "items": items,
    }
    path = search_dir / f"{stem}.json"
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(metadata, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


def build_face_index(
    archive: Path,
    detector_model: Path,
    recognizer_model: Path,
    confidence: float = 0.70,
    max_side: int = 1280,
    checkpoint_every: int = 100,
) -> dict:
    archive = archive.resolve()
    detector_model = detector_model.resolve()
    recognizer_model = recognizer_model.resolve()
    manifest = read_gallery_manifest(archive)
    photos = manifest.get("photos", [])
    search_dir = archive / "search"
    search_dir.mkdir(parents=True, exist_ok=True)
    identity = _model_identity(detector_model, recognizer_model)
    settings = {
        "confidence": confidence,
        "max_side": max_side,
        "minimum_detected_face": 22,
    }
    cached, groups = _read_cached_groups(search_dir, identity)
    cached_shas = {sha256 for _, sha256 in cached}
    groups_by_sha: dict[str, list[tuple[dict, np.ndarray]]] = {}
    for (_, sha256), entries in groups.items():
        groups_by_sha.setdefault(sha256, entries)

    cv2.setNumThreads(1)

    detector = cv2.FaceDetectorYN.create(
        str(detector_model), "", (320, 320), confidence, 0.3, 5000
    )
    recognizer = cv2.FaceRecognizerSF.create(str(recognizer_model), "")
    processed: list[dict] = []
    items: list[dict] = []
    vectors: list[np.ndarray] = []
    reused = 0
    newly_processed = 0
    last_checkpoint = 0

    for position, photo in enumerate(photos, 1):
        key = (photo["id"], photo["sha256"])
        processed.append({"id": photo["id"], "sha256": photo["sha256"]})
        if key in cached:
            for item, vector in groups.get(key, []):
                items.append(item)
                vectors.append(vector)
            reused += 1
        elif photo["sha256"] in cached_shas:
            # Renames change gallery ids but not face locations or vectors.
            # Reuse the cached faces while attaching them to the new photo id.
            for face_number, (item, vector) in enumerate(
                groups_by_sha.get(photo["sha256"], [])
            ):
                items.append(
                    {
                        **item,
                        "id": f"{photo['id']}:{face_number}",
                        "photo_id": photo["id"],
                        "photo_sha256": photo["sha256"],
                    }
                )
                vectors.append(vector)
            reused += 1
        else:
            if not photo.get("original"):
                raise ValueError(f"Photo {photo['id']} has no archived original")
            source = archive / Path(unquote(photo["original"]))
            image = cv2.imread(str(source), cv2.IMREAD_COLOR)
            if image is None:
                print(f"Warning: could not read {source}", file=sys.stderr)
            else:
                detection_image, scale = _resized(image, max_side)
                detector.setInputSize((detection_image.shape[1], detection_image.shape[0]))
                _, detections = detector.detect(detection_image)
                if detections is not None:
                    image_height, image_width = image.shape[:2]
                    accepted = 0
                    for detected in detections:
                        if min(detected[2], detected[3]) < 22:
                            continue
                        face = detected.copy()
                        face[:14] /= scale
                        try:
                            aligned = recognizer.alignCrop(image, face)
                            vector = recognizer.feature(aligned).reshape(-1).astype(np.float32)
                        except cv2.error:
                            continue
                        norm = float(np.linalg.norm(vector))
                        if not np.isfinite(norm) or norm == 0:
                            continue
                        vector /= norm
                        x, y, width, height = face[:4]
                        normalized_x = max(0.0, float(x) / image_width)
                        normalized_y = max(0.0, float(y) / image_height)
                        bbox = [
                            normalized_x,
                            normalized_y,
                            min(1.0 - normalized_x, float(width) / image_width),
                            min(1.0 - normalized_y, float(height) / image_height),
                        ]
                        items.append(
                            {
                                "id": f"{photo['id']}:{accepted}",
                                "photo_id": photo["id"],
                                "photo_sha256": photo["sha256"],
                                "bbox": [round(value, 6) for value in bbox],
                                "detection_score": round(float(detected[-1]), 6),
                            }
                        )
                        vectors.append(vector)
                        accepted += 1
            newly_processed += 1

        if position % 50 == 0 or position == len(photos):
            print(
                f"Processed {position:,}/{len(photos):,} photos; found {len(items):,} faces",
                flush=True,
            )
        if newly_processed and (
            newly_processed - last_checkpoint >= checkpoint_every
            or position == len(photos)
        ):
            _save(
                search_dir,
                ".partial-faces",
                identity,
                processed,
                items,
                vectors,
                settings,
            )
            print(f"Checkpointed at {position:,} photos", flush=True)
            last_checkpoint = newly_processed

    _save(search_dir, "faces", identity, processed, items, vectors, settings)
    (search_dir / ".partial-faces.json").unlink(missing_ok=True)
    (search_dir / ".partial-faces.npy").unlink(missing_ok=True)
    print(
        f"Indexed {len(items):,} faces from {len(photos):,} photos "
        f"({reused:,} reused, {newly_processed:,} newly processed).",
        flush=True,
    )
    return {"photo_count": len(photos), "face_count": len(items), "model": identity}


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--archive", required=True, type=Path)
    result.add_argument("--detector-model", required=True, type=Path)
    result.add_argument("--recognizer-model", required=True, type=Path)
    result.add_argument("--confidence", type=float, default=0.70)
    result.add_argument("--max-side", type=int, default=1280)
    result.add_argument("--checkpoint-every", type=int, default=100)
    return result


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if not 0 < args.confidence <= 1 or args.max_side < 320 or args.checkpoint_every < 1:
        print("Error: invalid face-index settings", file=sys.stderr)
        return 2
    try:
        build_face_index(
            args.archive,
            args.detector_model,
            args.recognizer_model,
            args.confidence,
            args.max_side,
            args.checkpoint_every,
        )
    except (OSError, RuntimeError, ValueError, cv2.error) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
