"""Train one multiclass identity classifier from confirmed face labels."""

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


def confirmed_labels(labels: dict, rows_by_id: dict[str, int]) -> dict[str, str]:
    """Return confirmed face-to-person labels, rejecting conflicting identities."""
    result: dict[str, str] = {}

    def add(person: str, face_ids: list[str]) -> None:
        for face_id in face_ids:
            if face_id not in rows_by_id:
                continue
            previous = result.get(face_id)
            if previous and previous != person:
                raise ValueError(f"Face {face_id} is labeled as both {previous} and {person}")
            result[face_id] = person

    for cluster in labels.get("clusters", []):
        if cluster.get("name"):
            add(cluster["name"], cluster.get("face_ids", []))
    for person, batches in labels.get("recovery", {}).get("accepted", {}).items():
        for batch in batches:
            add(person, batch)
    return result


def train_balanced_ridge(
    features: np.ndarray,
    targets: np.ndarray,
    class_count: int,
    regularization: float = 1.0,
) -> np.ndarray:
    """Fit a class-balanced multiclass ridge classifier with an intercept."""
    if len(features) != len(targets) or not len(features):
        raise ValueError("Training features and labels must be non-empty and aligned")
    counts = np.bincount(targets, minlength=class_count)
    if np.any(counts == 0):
        raise ValueError("Every identity must have at least one training face")
    design = np.column_stack((features, np.ones(len(features), dtype=features.dtype)))
    weights = len(targets) / (class_count * counts[targets])
    outcomes = np.eye(class_count, dtype=np.float32)[targets]
    penalty = np.eye(design.shape[1], dtype=np.float32) * regularization
    penalty[-1, -1] = 0
    system = design.T @ (weights[:, None] * design) + penalty
    response = design.T @ (weights[:, None] * outcomes)
    return np.linalg.solve(system, response).astype(np.float32)


def predict_identities(
    features: np.ndarray, coefficients: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return winning class, its score, and its lead over the runner-up."""
    scores = features @ coefficients[:-1] + coefficients[-1]
    order = np.argsort(scores, axis=1)
    winners = order[:, -1]
    rows = np.arange(len(features))
    winning_scores = scores[rows, winners]
    margins = winning_scores - scores[rows, order[:, -2]]
    return winners, winning_scores, margins


def _asset_url(path: Path, output: Path) -> str:
    relative = os.path.relpath(path.resolve(), output.parent.resolve())
    return quote(Path(relative).as_posix(), safe="/()._-~")


def _write_audit(
    archive: Path,
    output: Path,
    classes: list[str],
    assignments: list[dict],
    confirmed_counts: dict[str, int],
) -> None:
    manifest = read_gallery_manifest(archive)
    photos = {photo["id"]: photo for photo in manifest["photos"]}
    candidates: dict[str, list[dict]] = {person: [] for person in classes}
    for assignment in assignments:
        if assignment["confirmed"]:
            continue
        photo = photos.get(assignment["photo_id"])
        if not photo:
            continue
        candidates[assignment["person"]].append(
            {
                **assignment,
                "preview": _asset_url(archive / photo["preview"], output),
                "filename": photo["filename"],
                "album": photo["album_id"],
                "year": photo.get("date_start"),
                "width": photo.get("width") or 1,
                "height": photo.get("height") or 1,
            }
        )
    for values in candidates.values():
        values.sort(key=lambda value: (value["margin"], value["score"]), reverse=True)
    payload = {"classes": classes, "candidates": candidates, "confirmed": confirmed_counts}
    document = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Identity model audit</title><style>
:root{{--ink:#17201d;--muted:#68736e;--paper:#f2f3ee;--accent:#b94f32;--green:#173d36}}*{{box-sizing:border-box}}body{{margin:0;background:var(--paper);color:var(--ink);font:16px/1.45 system-ui,sans-serif}}header,main{{width:min(1500px,calc(100% - 36px));margin:auto}}header{{padding:28px 0 20px}}h1{{font:500 clamp(2.2rem,6vw,4rem)/1 Georgia,serif;margin:0 0 10px}}p{{color:var(--muted)}}select{{font:inherit;padding:9px;border:1px solid #bcc4bd;border-radius:7px;background:#fff}}.summary{{font-weight:700;color:var(--green)}}.grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(135px,1fr));gap:12px;margin:18px 0 40px}}figure{{margin:0;background:#fff;border-radius:9px;overflow:hidden}}svg{{display:block;width:100%;aspect-ratio:1;background:#ddd}}figcaption{{padding:8px;font-size:.75rem}}figcaption strong,figcaption span{{display:block;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}}figcaption span{{color:var(--muted)}}
</style></head><body><header><h1>Identity model audit</h1><p>One shared classifier trained on confirmed faces for {len(classes)} people. Predictions shown here were not used for training.</p><select id="person"></select><p class="summary" id="summary"></p></header><main><div class="grid" id="grid"></div></main><script>
const DATA={json.dumps(payload, ensure_ascii=False, separators=(',', ':'))};const esc=s=>String(s??'').replace(/[&<>"']/g,c=>({{'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}}[c]));
function crop(f){{const [bx,by,bw,bh]=f.bbox,p=.42,x=Math.max(0,bx-bw*p),y=Math.max(0,by-bh*p),w=Math.min(1-x,bw*(1+2*p)),h=Math.min(1-y,bh*(1+2*p));return `<svg viewBox="${{x*f.width}} ${{y*f.height}} ${{w*f.width}} ${{h*f.height}}" preserveAspectRatio="xMidYMid slice"><image href="${{f.preview}}" width="${{f.width}}" height="${{f.height}}"/></svg>`}}const select=document.querySelector('#person');select.innerHTML=DATA.classes.map(n=>`<option>${{esc(n)}}</option>`).join('');function render(){{const n=select.value,items=DATA.candidates[n];document.querySelector('#summary').textContent=`${{DATA.confirmed[n]}} confirmed training faces · ${{items.length}} new high-confidence assignments`;document.querySelector('#grid').innerHTML=items.map((f,i)=>`<figure>${{crop(f)}}<figcaption><strong>#${{i+1}} · model score ${{f.score.toFixed(2)}}</strong><span>Lead over next person ${{f.margin.toFixed(2)}}</span><span>${{esc(f.album)}}${{f.year?' · '+f.year:''}}</span></figcaption></figure>`).join('')}}select.onchange=render;select.value=DATA.classes.includes('Anthony')?'Anthony':DATA.classes[0];render();
</script></body></html>"""
    output.write_text(document, encoding="utf-8")


def train_identity_model(
    archive: Path,
    labels_path: Path,
    model_output: Path,
    assignments_output: Path,
    audit_output: Path,
    regularization: float = 1.0,
    minimum_score: float = 0.45,
    minimum_margin: float = 0.25,
) -> dict:
    archive, labels_path = archive.resolve(), labels_path.resolve()
    labels = json.loads(labels_path.read_text(encoding="utf-8"))
    metadata, embeddings = read_face_index(archive)
    items = metadata["items"]
    rows_by_id = {item["id"]: row for row, item in enumerate(items)}
    confirmed = confirmed_labels(labels, rows_by_id)
    classes = sorted(set(confirmed.values()))
    class_rows = {person: index for index, person in enumerate(classes)}
    training_ids = list(confirmed)
    training_rows = np.asarray([rows_by_id[face_id] for face_id in training_ids])
    targets = np.asarray([class_rows[confirmed[face_id]] for face_id in training_ids])
    coefficients = train_balanced_ridge(
        embeddings[training_rows], targets, len(classes), regularization
    )
    winners, scores, margins = predict_identities(embeddings, coefficients)
    assignments = []
    for row, item in enumerate(items):
        is_confirmed = item["id"] in confirmed
        if not is_confirmed and (scores[row] < minimum_score or margins[row] < minimum_margin):
            continue
        person = confirmed[item["id"]] if is_confirmed else classes[int(winners[row])]
        assignments.append(
            {
                "face_id": item["id"],
                "photo_id": item["photo_id"],
                "bbox": item["bbox"],
                "person": person,
                "score": float(scores[row]),
                "margin": float(margins[row]),
                "confirmed": is_confirmed,
            }
        )
    confirmed_counts = {
        person: sum(value == person for value in confirmed.values()) for person in classes
    }
    new_counts = {
        person: sum(not item["confirmed"] and item["person"] == person for item in assignments)
        for person in classes
    }
    summary = {
        "version": 1,
        "method": "class-balanced-multiclass-ridge",
        "classes": classes,
        "regularization": regularization,
        "minimum_score": minimum_score,
        "minimum_margin": minimum_margin,
        "confirmed_counts": confirmed_counts,
        "automatic_counts": new_counts,
        "assignments": assignments,
    }
    for path in (model_output, assignments_output, audit_output):
        path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        model_output,
        coefficients=coefficients,
        classes=np.asarray(classes),
        regularization=np.asarray(regularization),
        minimum_score=np.asarray(minimum_score),
        minimum_margin=np.asarray(minimum_margin),
    )
    assignments_output.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    _write_audit(archive, audit_output, classes, assignments, confirmed_counts)
    print(
        f"Trained {len(classes)} identities from {len(confirmed):,} confirmed faces; "
        f"assigned {sum(new_counts.values()):,} additional faces.",
        flush=True,
    )
    return summary


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--archive", required=True, type=Path)
    result.add_argument("--labels", required=True, type=Path)
    result.add_argument("--regularization", type=float, default=1.0)
    result.add_argument("--minimum-score", type=float, default=0.45)
    result.add_argument("--minimum-margin", type=float, default=0.25)
    return result


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        train_identity_model(
            args.archive,
            args.labels,
            args.archive / "search" / "identity-model.npz",
            args.archive / "search" / "identity-assignments.json",
            args.archive / "identity-model-audit.html",
            args.regularization,
            args.minimum_score,
            args.minimum_margin,
        )
    except (OSError, ValueError, json.JSONDecodeError, np.linalg.LinAlgError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
