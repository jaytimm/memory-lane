"""Export search vectors as scripts the gallery can load without a server.

Browsers block fetch() for pages opened from file://, but still run <script>
files. Writing the precomputed vectors as scripts lets "Find similar photos"
and "Find this person" work from a USB on any computer, including a Chromebook.
"""

from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path
import sys

import numpy as np

from .archive_data import read_face_index, read_search_index


SEARCH_DATA_FILE = "search-data.js"
SIMILAR_FILE = "search-similar.js"
FACES_FILE = "search-faces.js"


def _vectors_base64(matrix: np.ndarray) -> str:
    # Little-endian float32 matches JavaScript's Float32Array on every
    # platform the gallery targets.
    data = np.ascontiguousarray(matrix, dtype="<f4").tobytes()
    return base64.b64encode(data).decode("ascii")


def _write_script(path: Path, variable: str, value: dict) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        f"window.{variable} = "
        + json.dumps(value, ensure_ascii=False, separators=(",", ":"))
        + ";\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def write_browser_index(archive: str | Path) -> dict:
    """Write browser search scripts for whichever indexes the archive has."""
    archive = Path(archive).resolve()
    assets = archive / "assets"
    assets.mkdir(parents=True, exist_ok=True)
    search_dir = archive / "search"
    config: dict = {"version": 1, "similar": None, "faces": None}

    if (search_dir / "index.json").exists():
        metadata, embeddings = read_search_index(archive)
        if embeddings.size:
            _write_script(
                assets / SIMILAR_FILE,
                "MEMORY_LANE_SIMILAR_VECTORS",
                {
                    "dimension": int(embeddings.shape[1]),
                    "ids": [item["id"] for item in metadata["items"]],
                    "vectors": _vectors_base64(embeddings),
                },
            )
            config["similar"] = {"file": SIMILAR_FILE, "count": len(metadata["items"])}

    if (search_dir / "faces.json").exists():
        metadata, embeddings = read_face_index(archive)
        if embeddings.size:
            items = metadata["items"]
            _write_script(
                assets / FACES_FILE,
                "MEMORY_LANE_FACE_VECTORS",
                {
                    "dimension": int(embeddings.shape[1]),
                    "ids": [item["id"] for item in items],
                    "photo_ids": [item["photo_id"] for item in items],
                    "bboxes": [item["bbox"] for item in items],
                    "vectors": _vectors_base64(embeddings),
                },
            )
            config["faces"] = {"file": FACES_FILE, "count": len(items)}

    # Write the small pointer file last so it never names a missing data file.
    _write_script(assets / SEARCH_DATA_FILE, "MEMORY_LANE_SEARCH_DATA", config)
    if config["similar"] is None:
        (assets / SIMILAR_FILE).unlink(missing_ok=True)
    if config["faces"] is None:
        (assets / FACES_FILE).unlink(missing_ok=True)
    return config


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--archive", required=True, type=Path)
    return result


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        config = write_browser_index(args.archive)
    except (OSError, ValueError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 2
    similar = config["similar"]["count"] if config["similar"] else 0
    faces = config["faces"]["count"] if config["faces"] else 0
    print(f"Wrote browser search data: {similar:,} photos, {faces:,} faces.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
