"""Helpers for reading the generated gallery and semantic index."""

from __future__ import annotations

import json
from pathlib import Path


MANIFEST_PREFIX = "window.FAMILY_PHOTO_DATA = "


def read_gallery_manifest(archive: str | Path) -> dict:
    path = Path(archive) / "assets" / "manifest.js"
    if not path.exists():
        raise ValueError(f"Gallery manifest not found: {path}")
    source = path.read_text(encoding="utf-8").strip()
    if not source.startswith(MANIFEST_PREFIX) or not source.endswith(";"):
        raise ValueError(f"Unrecognized gallery manifest format: {path}")
    return json.loads(source[len(MANIFEST_PREFIX) : -1])


def read_search_index(archive: str | Path) -> tuple[dict, "object"]:
    import numpy as np

    search_dir = Path(archive) / "search"
    metadata_path = search_dir / "index.json"
    embeddings_path = search_dir / "embeddings.npy"
    if not metadata_path.exists() or not embeddings_path.exists():
        raise ValueError(
            f"Semantic index not found in {search_dir}. Run python -m src.embed first."
        )
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    embeddings = np.load(embeddings_path, allow_pickle=False)
    if embeddings.ndim != 2 or embeddings.shape[0] != len(metadata.get("items", [])):
        raise ValueError("Semantic index metadata and embedding rows do not match")
    return metadata, embeddings


def read_face_index(archive: str | Path) -> tuple[dict, "object"]:
    import numpy as np

    search_dir = Path(archive) / "search"
    metadata_path = search_dir / "faces.json"
    embeddings_path = search_dir / "faces.npy"
    if not metadata_path.exists() or not embeddings_path.exists():
        raise ValueError(f"Face index not found in {search_dir}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    embeddings = np.load(embeddings_path, allow_pickle=False)
    if embeddings.ndim != 2 or embeddings.shape[0] != len(metadata.get("items", [])):
        raise ValueError("Face index metadata and embedding rows do not match")
    return metadata, embeddings
