"""Build a read-only audit of face similarity for one named person."""

from __future__ import annotations

import argparse
from html import escape
import json
import os
from pathlib import Path
import sys
from urllib.parse import quote

import numpy as np

from .archive_data import read_face_index, read_gallery_manifest


def _centroid(embeddings: np.ndarray, rows: list[int]) -> np.ndarray:
    vector = embeddings[rows].mean(axis=0)
    norm = float(np.linalg.norm(vector))
    if not norm:
        raise ValueError("Cannot build a profile from zero-length face vectors")
    return vector / norm


def label_profiles(
    labels: dict, embeddings: np.ndarray, rows_by_id: dict[str, int]
) -> tuple[dict[str, list[np.ndarray]], set[str]]:
    """Build one prototype per labeled cluster or accepted recovery batch."""
    profiles: dict[str, list[np.ndarray]] = {}
    decided: set[str] = set()
    for cluster in labels.get("clusters", []):
        face_ids = [face_id for face_id in cluster.get("face_ids", []) if face_id in rows_by_id]
        decided.update(face_ids)
        name = cluster.get("name")
        if name and face_ids:
            profiles.setdefault(name, []).append(
                _centroid(embeddings, [rows_by_id[face_id] for face_id in face_ids])
            )
    recovery = labels.get("recovery", {})
    for name, batches in recovery.get("accepted", {}).items():
        for batch in batches:
            face_ids = [face_id for face_id in batch if face_id in rows_by_id]
            decided.update(face_ids)
            if face_ids:
                profiles.setdefault(name, []).append(
                    _centroid(embeddings, [rows_by_id[face_id] for face_id in face_ids])
                )
    return profiles, decided


def score_identity(
    embeddings: np.ndarray,
    profiles: dict[str, list[np.ndarray]],
    person: str,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Return target score, lead over the closest other person, and competitor."""
    if person not in profiles:
        raise ValueError(f"No labeled profiles found for {person}")
    target = np.stack(profiles[person])
    target_scores = np.max(embeddings @ target.T, axis=1)
    other_names = [name for name in profiles if name != person]
    if not other_names:
        return target_scores, np.full(len(embeddings), np.inf), [""] * len(embeddings)
    other_vectors = np.stack([vector for name in other_names for vector in profiles[name]])
    vector_names = [name for name in other_names for _ in profiles[name]]
    other_scores = embeddings @ other_vectors.T
    closest_columns = np.argmax(other_scores, axis=1)
    closest_scores = other_scores[np.arange(len(embeddings)), closest_columns]
    closest_names = [vector_names[int(column)] for column in closest_columns]
    return target_scores, target_scores - closest_scores, closest_names


def _asset_url(path: Path, output: Path) -> str:
    relative = os.path.relpath(path.resolve(), output.parent.resolve())
    return quote(Path(relative).as_posix(), safe="/()._-~")


def build_identity_audit(
    archive: Path,
    labels_path: Path,
    person: str,
    output: Path,
    strong_score: float = 0.60,
    strong_margin: float = 0.08,
    review_score: float = 0.50,
    review_margin: float = 0.03,
) -> Path:
    archive, labels_path, output = archive.resolve(), labels_path.resolve(), output.resolve()
    labels = json.loads(labels_path.read_text(encoding="utf-8"))
    metadata, embeddings = read_face_index(archive)
    manifest = read_gallery_manifest(archive)
    items = metadata["items"]
    rows_by_id = {item["id"]: row for row, item in enumerate(items)}
    profiles, decided = label_profiles(labels, embeddings, rows_by_id)
    target_scores, margins, closest_names = score_identity(embeddings, profiles, person)
    dismissed = set(labels.get("recovery", {}).get("dismissed", {}).get(person, []))
    photos = {photo["id"]: photo for photo in manifest["photos"]}

    candidates = []
    for row, item in enumerate(items):
        if item["id"] in decided or item["id"] in dismissed:
            continue
        photo = photos.get(item["photo_id"])
        if not photo:
            continue
        candidates.append(
            {
                "face_id": item["id"],
                "bbox": item["bbox"],
                "score": float(target_scores[row]),
                "margin": float(margins[row]),
                "closest": closest_names[row],
                "preview": _asset_url(archive / photo["preview"], output),
                "filename": photo["filename"],
                "album": photo["album_id"],
                "year": photo.get("date_start"),
                "width": photo.get("width") or 1,
                "height": photo.get("height") or 1,
            }
        )
    candidates.sort(key=lambda candidate: candidate["score"], reverse=True)
    strong = [
        candidate
        for candidate in candidates
        if candidate["score"] >= strong_score and candidate["margin"] >= strong_margin
    ]
    review = [
        candidate
        for candidate in candidates
        if candidate not in strong
        and candidate["score"] >= review_score
        and candidate["margin"] >= review_margin
    ]
    approved_count = len(labels.get("people", {}).get(person, {}).get("face_ids", []))
    payload = {"person": person, "strong": strong, "review": review}
    title = f"{person} face-assignment audit"
    document = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{escape(title)}</title><style>
:root{{--ink:#17201d;--muted:#68736e;--paper:#f2f3ee;--accent:#b94f32;--green:#173d36}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--paper);color:var(--ink);font:16px/1.45 system-ui,sans-serif}}header,main{{width:min(1500px,calc(100% - 36px));margin:auto}}header{{padding:32px 0 24px}}
h1{{font:500 clamp(2.2rem,6vw,4rem)/1 Georgia,serif;margin:0 0 10px}}p{{color:var(--muted);max-width:950px}}.summary{{font-size:1.08rem}}
section{{border-top:1px solid #d2d8d2;padding:24px 0 40px}}h2{{font:500 2rem Georgia,serif;margin:0 0 5px}}.grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(135px,1fr));gap:12px;margin-top:18px}}
figure{{margin:0;background:#fff;border-radius:9px;overflow:hidden}}figure a{{display:block}}svg{{display:block;width:100%;aspect-ratio:1;background:#ddd}}figcaption{{padding:8px;font-size:.75rem}}strong,span{{display:block}}figcaption strong,figcaption span{{white-space:nowrap;overflow:hidden;text-overflow:ellipsis}}span{{color:var(--muted)}}.empty{{padding:24px;background:#fff;border-radius:9px}}
</style></head><body><header><h1>{escape(title)}</h1>
<p class="summary"><b>{approved_count} approved {escape(person)} faces</b> produced <b>{len(strong)} strong automatic candidates</b> and <b>{len(review)} review-zone candidates</b>.</p>
<p>Similarity and separation are match scores, not probabilities. Strong candidates require at least {strong_score:.0%} similarity and an {strong_margin:.0%} lead over every other named person. This page is read-only; no identity assignments have been changed.</p>
</header><main><section><h2>Strong automatic candidates</h2><p>These are the proposed automatic assignments to inspect first.</p><div class="grid" id="strong"></div></section>
<section><h2>Review zone</h2><p>These are plausible but will remain unassigned unless explicitly approved.</p><div class="grid" id="review"></div></section></main>
<script>const DATA={json.dumps(payload, ensure_ascii=False, separators=(',', ':'))};
const esc=s=>String(s??'').replace(/[&<>"']/g,c=>({{'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}}[c]));
function crop(face){{const [bx,by,bw,bh]=face.bbox,p=.42,x=Math.max(0,bx-bw*p),y=Math.max(0,by-bh*p),w=Math.min(1-x,bw*(1+2*p)),h=Math.min(1-y,bh*(1+2*p));return `<svg viewBox="${{x*face.width}} ${{y*face.height}} ${{w*face.width}} ${{h*face.height}}" preserveAspectRatio="xMidYMid slice"><image href="${{face.preview}}" width="${{face.width}}" height="${{face.height}}"/></svg>`}}
function render(id,items){{const target=document.querySelector('#'+id);target.innerHTML=items.map((face,index)=>`<figure><a href="${{face.preview}}" target="_blank">${{crop(face)}}</a><figcaption><strong>#${{index+1}} · similarity ${{Math.round(face.score*100)}}%</strong><span>Lead ${{Math.round(face.margin*100)}}% over ${{esc(face.closest)}}</span><span>${{esc(face.album)}}${{face.year?' · '+face.year:''}}</span></figcaption></figure>`).join('')||'<p class="empty">No faces in this section.</p>'}}render('strong',DATA.strong);render('review',DATA.review);
</script></body></html>"""
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(document, encoding="utf-8")
    print(
        f"Wrote {output} ({len(strong)} strong; {len(review)} review-zone candidates)",
        flush=True,
    )
    return output


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--archive", required=True, type=Path)
    result.add_argument("--labels", required=True, type=Path)
    result.add_argument("--person", required=True)
    result.add_argument("--output", type=Path)
    result.add_argument("--strong-score", type=float, default=0.60)
    result.add_argument("--strong-margin", type=float, default=0.08)
    result.add_argument("--review-score", type=float, default=0.50)
    result.add_argument("--review-margin", type=float, default=0.03)
    return result


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    output = args.output or args.archive / f"{args.person.casefold()}-identity-audit.html"
    try:
        build_identity_audit(
            args.archive,
            args.labels,
            args.person,
            output,
            args.strong_score,
            args.strong_margin,
            args.review_score,
            args.review_margin,
        )
    except (OSError, ValueError, json.JSONDecodeError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
