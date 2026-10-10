"""Create an offline visual page of top semantic-search results."""

from __future__ import annotations

import argparse
from html import escape
import os
from pathlib import Path
import sys
from urllib.parse import quote, unquote

from .archive_data import read_gallery_manifest
from .clip_model import load_clip
from .search import search_index


DEFAULT_QUERIES = [
    "children playing in the snow",
    "Christmas morning opening presents",
    "a family standing outside a house",
    "birthday cake and candles",
    "people swimming in a lake",
    "dogs playing in the backyard",
    "a parrot",
    "children playing outside",
    "a wedding celebration",
]


def _relative_url(archive: Path, output: Path, archive_url: str) -> str:
    target = archive / Path(unquote(archive_url))
    relative = os.path.relpath(target, output.parent)
    return quote(Path(relative).as_posix(), safe="/()._-~")


def build_evaluation(
    archive: Path, queries: list[str], top_k: int, output: Path | None = None
) -> Path:
    archive = archive.resolve()
    output = (output or archive / "search-evaluation.html").resolve()
    manifest = read_gallery_manifest(archive)
    photo_count = len(manifest.get("photos", []))
    album_count = len(manifest.get("albums", []))
    gallery_url = quote(
        Path(os.path.relpath(archive / "index.html", output.parent)).as_posix(),
        safe="/()._-~",
    )
    model, _, tokenizer, _ = load_clip(archive / ".model-cache")
    sections: list[str] = []
    for query in queries:
        print(f"Searching: {query}", flush=True)
        results = search_index(
            archive, query, top_k=top_k, loaded_model=(model, tokenizer)
        )
        cards = []
        for rank, result in enumerate(results, 1):
            image_url = _relative_url(archive, output, result["thumbnail"])
            preview_url = _relative_url(archive, output, result["preview"])
            cards.append(
                f'<figure><a href="{preview_url}" target="_blank"><img src="{image_url}" alt="" loading="lazy"></a>'
                f'<figcaption><strong>{rank}. {escape(result["filename"])}</strong>'
                f'<span>{escape(result["album_id"])} · {result["score"]:.3f}</span>'
                "</figcaption></figure>"
            )
        sections.append(
            f'<section><h2>{escape(query)}</h2><div class="grid">'
            + "".join(cards)
            + "</div></section>"
        )
    document = """<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Semantic search evaluation</title><style>
body{margin:0;background:#f2f3ee;color:#17201d;font-family:system-ui,sans-serif}
header,section{width:min(1400px,calc(100% - 40px));margin:auto}header{padding:42px 0 20px}
h1{font:500 3rem Georgia,serif;margin:0}p{color:#65706b}section{padding:25px 0;border-top:1px solid #d9ddd7}
h2{font:500 2rem Georgia,serif}.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(180px,1fr));gap:12px}
figure{margin:0;background:white;border-radius:8px;overflow:hidden}img{display:block;width:100%;aspect-ratio:1;object-fit:cover}
figcaption{padding:10px;font-size:.78rem}strong,span{display:block;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
span{color:#65706b;margin-top:4px}.badge{display:inline-block;margin:18px 0 4px;padding:8px 12px;border-radius:999px;background:#173d36;color:white;font-weight:750;letter-spacing:.05em;text-transform:uppercase}
header a{color:#b94f32}</style></head><body><header><h1>Semantic search evaluation</h1>"""
    document += (
        f'<p class="badge">Full collection · {photo_count:,} photographs · {album_count} albums</p>'
        '<p>Results are ranked only by CLIP image similarity. Scores are comparative, not probabilities. '
        f'<a href="{gallery_url}">Return to gallery</a>.</p></header>'
    )
    document += "".join(sections) + "</body></html>"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(document, encoding="utf-8")
    print(f"Wrote {output}")
    return output


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--archive", required=True, type=Path)
    result.add_argument("--query", action="append", dest="queries")
    result.add_argument("--top-k", type=int, default=20)
    result.add_argument("--output", type=Path)
    return result


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        build_evaluation(
            args.archive, args.queries or DEFAULT_QUERIES, args.top_k, args.output
        )
    except (OSError, RuntimeError, ValueError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
