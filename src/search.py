"""Search a photo archive with natural language and optional metadata filters."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from threading import Lock

import numpy as np

from .archive_data import read_search_index
from .clip_model import TEXT_PROMPT_TEMPLATE, load_clip, model_identity


def rank_embeddings(
    embeddings: np.ndarray,
    query_vector: np.ndarray,
    candidate_indices: np.ndarray,
    top_k: int,
) -> list[tuple[int, float]]:
    if candidate_indices.size == 0:
        return []
    scores = embeddings[candidate_indices] @ query_vector
    count = min(top_k, scores.size)
    if count == 0:
        return []
    selected = np.argpartition(scores, -count)[-count:]
    selected = selected[np.argsort(scores[selected])[::-1]]
    return [(int(candidate_indices[index]), float(scores[index])) for index in selected]


def _candidate_indices(
    items: list[dict], album: str | None, date_from: int | None, date_to: int | None
) -> np.ndarray:
    candidates: list[int] = []
    album_query = album.casefold() if album else None
    for index, item in enumerate(items):
        if album_query and album_query not in (item.get("album_id") or "").casefold():
            continue
        item_start = item.get("date_start")
        item_end = item.get("date_end") or item_start
        if date_from is not None and (item_end is None or item_end < date_from):
            continue
        if date_to is not None and (item_start is None or item_start > date_to):
            continue
        candidates.append(index)
    return np.asarray(candidates, dtype=np.int64)


def encode_query(model, tokenizer, query: str) -> np.ndarray:
    import torch

    prompted = TEXT_PROMPT_TEMPLATE.format(query.strip())
    tokens = tokenizer([prompted])
    with torch.inference_mode():
        features = model.encode_text(tokens)
        features = features / features.norm(dim=-1, keepdim=True)
    return features[0].cpu().numpy().astype(np.float32, copy=False)


class SemanticSearchEngine:
    """Keep the model and index resident for low-latency repeated searches."""

    def __init__(self, archive: Path):
        self.archive = Path(archive).resolve()
        self.metadata, self.embeddings = read_search_index(self.archive)
        identity = model_identity()
        if self.metadata.get("model") != identity:
            raise ValueError("Installed OpenCLIP model version does not match this index")
        self.model, _, self.tokenizer, _ = load_clip(self.archive / ".model-cache")
        self._model_lock = Lock()

    @property
    def photo_count(self) -> int:
        return len(self.metadata["items"])

    def search(
        self,
        query: str,
        top_k: int = 80,
        album: str | None = None,
        date_from: int | None = None,
        date_to: int | None = None,
    ) -> list[dict]:
        with self._model_lock:
            query_vector = encode_query(self.model, self.tokenizer, query)
        candidates = _candidate_indices(
            self.metadata["items"], album, date_from, date_to
        )
        ranked = rank_embeddings(self.embeddings, query_vector, candidates, top_k)
        return [
            {
                **self.metadata["items"][index],
                "score": round(score, 6),
                "query": query,
            }
            for index, score in ranked
        ]


def search_index(
    archive: Path,
    query: str,
    top_k: int = 20,
    album: str | None = None,
    date_from: int | None = None,
    date_to: int | None = None,
    loaded_model=None,
) -> list[dict]:
    metadata, embeddings = read_search_index(archive)
    identity = model_identity()
    if metadata.get("model") != identity:
        raise ValueError("Installed OpenCLIP model version does not match this index")
    if loaded_model is None:
        model, _, tokenizer, _ = load_clip(Path(archive) / ".model-cache")
    else:
        model, tokenizer = loaded_model
    query_vector = encode_query(model, tokenizer, query)
    candidates = _candidate_indices(metadata["items"], album, date_from, date_to)
    ranked = rank_embeddings(embeddings, query_vector, candidates, top_k)
    return [
        {**metadata["items"][index], "score": round(score, 6), "query": query}
        for index, score in ranked
    ]


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--archive", required=True, type=Path)
    result.add_argument("--query", required=True)
    result.add_argument("--top-k", type=int, default=20)
    result.add_argument("--album")
    result.add_argument("--date-from", type=int)
    result.add_argument("--date-to", type=int)
    result.add_argument("--json", action="store_true")
    return result


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if args.top_k < 1:
        print("Error: --top-k must be positive", file=sys.stderr)
        return 2
    try:
        results = search_index(
            args.archive, args.query, args.top_k, args.album, args.date_from, args.date_to
        )
    except (OSError, RuntimeError, ValueError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 2
    if args.json:
        print(json.dumps(results, ensure_ascii=False, indent=2))
    else:
        print(f"Top {len(results)} results for {args.query!r}:")
        for rank, result in enumerate(results, 1):
            print(
                f"{rank:>2}. {result['score']:.4f}  "
                f"{result['album_id']}/{result['filename']}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
