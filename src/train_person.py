"""Train a binary face-identity classifier for one confirmed person."""

from __future__ import annotations

import argparse
import hashlib
from html import escape
import json
import os
from pathlib import Path
import sys
from urllib.parse import quote

import numpy as np

from .archive_data import read_face_index, read_gallery_manifest


def confirmed_examples(
    labels: dict, rows_by_id: dict[str, int]
) -> tuple[dict[str, str], dict[str, str]]:
    """Return confirmed identities and independent cluster/recovery group IDs."""
    identities: dict[str, str] = {}
    groups: dict[str, str] = {}

    def add(person: str, face_ids: list[str], group: str) -> None:
        for face_id in face_ids:
            if face_id not in rows_by_id:
                continue
            previous = identities.get(face_id)
            if previous and previous != person:
                raise ValueError(f"Face {face_id} is labeled as both {previous} and {person}")
            identities[face_id] = person
            groups[face_id] = group

    for cluster in labels.get("clusters", []):
        if cluster.get("name"):
            add(
                cluster["name"],
                cluster.get("face_ids", []),
                f"cluster:{cluster.get('cluster_id', 'unknown')}",
            )
    for person, batches in labels.get("recovery", {}).get("accepted", {}).items():
        for number, batch in enumerate(batches):
            add(person, batch, f"recovery:{person}:{number}")
    return identities, groups


def train_binary_ridge(
    features: np.ndarray,
    positive: np.ndarray,
    regularization: float = 1.0,
) -> np.ndarray:
    """Fit a balanced binary ridge classifier with an intercept."""
    positive_count = int(positive.sum())
    negative_count = len(positive) - positive_count
    if not positive_count or not negative_count:
        raise ValueError("Binary training requires positive and negative examples")
    design = np.column_stack((features, np.ones(len(features), dtype=features.dtype)))
    weights = np.where(
        positive,
        len(positive) / (2 * positive_count),
        len(positive) / (2 * negative_count),
    )
    targets = np.where(positive, 1.0, -1.0)
    penalty = np.eye(design.shape[1], dtype=np.float32) * regularization
    penalty[-1, -1] = 0
    system = design.T @ (weights[:, None] * design) + penalty
    response = design.T @ (weights * targets)
    return np.linalg.solve(system, response).astype(np.float32)


def predict_scores(features: np.ndarray, coefficients: np.ndarray) -> np.ndarray:
    return features @ coefficients[:-1] + coefficients[-1]


def grouped_validation_scores(
    features: np.ndarray,
    positive: np.ndarray,
    groups: list[str],
    regularization: float = 1.0,
) -> np.ndarray:
    """Hold out each positive batch and similarly grouped negatives in turn."""
    positive_groups = sorted({group for group, value in zip(groups, positive) if value})
    if len(positive_groups) < 2:
        raise ValueError("At least two independent positive groups are required")
    group_count = len(positive_groups)
    folds = np.empty(len(groups), dtype=np.int32)
    positive_fold = {group: fold for fold, group in enumerate(positive_groups)}
    for row, (group, value) in enumerate(zip(groups, positive)):
        if value:
            folds[row] = positive_fold[group]
        else:
            digest = hashlib.sha1(group.encode("utf-8")).hexdigest()
            folds[row] = int(digest[:8], 16) % group_count
    scores = np.empty(len(features), dtype=np.float32)
    for fold in range(group_count):
        test = folds == fold
        model = train_binary_ridge(features[~test], positive[~test], regularization)
        scores[test] = predict_scores(features[test], model)
    return scores


def _asset_url(path: Path, output: Path) -> str:
    relative = os.path.relpath(path.resolve(), output.parent.resolve())
    return quote(Path(relative).as_posix(), safe="/()._-~")


def _write_audit(archive: Path, output: Path, payload: dict) -> None:
    manifest = read_gallery_manifest(archive)
    photos = {photo["id"]: photo for photo in manifest["photos"]}
    for section in ("automatic", "review"):
        enriched = []
        for candidate in payload[section]:
            photo = photos.get(candidate["photo_id"])
            if not photo:
                continue
            enriched.append(
                {
                    **candidate,
                    "preview": _asset_url(archive / photo["preview"], output),
                    "album": photo["album_id"],
                    "year": photo.get("date_start"),
                    "width": photo.get("width") or 1,
                    "height": photo.get("height") or 1,
                }
            )
        payload[section] = enriched
    title = f"{payload['person']} identity model"
    document = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>{escape(title)}</title><style>
:root{{--ink:#17201d;--muted:#68736e;--paper:#f2f3ee;--green:#173d36}}*{{box-sizing:border-box}}body{{margin:0;background:var(--paper);color:var(--ink);font:16px/1.45 system-ui,sans-serif}}header,main{{width:min(1500px,calc(100% - 36px));margin:auto}}header{{padding:28px 0 20px}}h1{{font:500 clamp(2.2rem,6vw,4rem)/1 Georgia,serif;margin:0 0 10px}}h2{{font:500 2rem Georgia,serif;margin:0}}p{{color:var(--muted);max-width:950px}}section{{border-top:1px solid #d2d8d2;padding:24px 0 40px}}.summary{{font-weight:700;color:var(--green)}}.grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(135px,1fr));gap:12px;margin-top:18px}}figure{{margin:0;background:#fff;border-radius:9px;overflow:hidden}}svg{{display:block;width:100%;aspect-ratio:1;background:#ddd}}figcaption{{padding:8px;font-size:.75rem}}figcaption strong,figcaption span{{display:block;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}}figcaption span{{color:var(--muted)}}</style></head><body><header><h1>{escape(title)}</h1><p class="summary">{payload['confirmed_count']} confirmed training faces · {len(payload['automatic'])} automatic assignments · {len(payload['review'])} lower-scoring possibilities</p><p>This is one binary {escape(payload['person'])}-versus-not-{escape(payload['person'])} model. Other named people were used only as negative training evidence.</p><p>Grouped validation recognized {payload['validation']['true_positive']} of {payload['validation']['positive_count']} held-out {escape(payload['person'])} faces. Predictions were not fed back into training.</p></header><main><section><h2>Automatic assignments</h2><p>Above the selected model boundary.</p><div class="grid" id="automatic"></div></section><section><h2>Review band</h2><p>Below the assignment boundary but still on the {escape(payload['person'])} side of the classifier.</p><div class="grid" id="review"></div></section></main><script>
const DATA={json.dumps(payload, ensure_ascii=False, separators=(',', ':'))};const esc=s=>String(s??'').replace(/[&<>"']/g,c=>({{'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}}[c]));function crop(f){{const [bx,by,bw,bh]=f.bbox,p=.42,x=Math.max(0,bx-bw*p),y=Math.max(0,by-bh*p),w=Math.min(1-x,bw*(1+2*p)),h=Math.min(1-y,bh*(1+2*p));return `<svg viewBox="${{x*f.width}} ${{y*f.height}} ${{w*f.width}} ${{h*f.height}}" preserveAspectRatio="xMidYMid slice"><image href="${{f.preview}}" width="${{f.width}}" height="${{f.height}}"/></svg>`}}function render(id){{document.querySelector('#'+id).innerHTML=DATA[id].map((f,i)=>`<figure>${{crop(f)}}<figcaption><strong>#${{i+1}} · score ${{f.score.toFixed(2)}}</strong><span>${{esc(f.album)}}${{f.year?' · '+f.year:''}}</span></figcaption></figure>`).join('')}}render('automatic');render('review');
</script></body></html>"""
    output.write_text(document, encoding="utf-8")


def train_person_model(
    archive: Path,
    labels_path: Path,
    person: str,
    model_output: Path,
    assignments_output: Path,
    audit_output: Path,
    regularization: float = 1.0,
    assignment_threshold: float = 0.20,
    review_threshold: float = 0.0,
) -> dict:
    archive, labels_path = archive.resolve(), labels_path.resolve()
    labels = json.loads(labels_path.read_text(encoding="utf-8"))
    metadata, embeddings = read_face_index(archive)
    items = metadata["items"]
    rows_by_id = {item["id"]: row for row, item in enumerate(items)}
    identities, example_groups = confirmed_examples(labels, rows_by_id)
    training_ids = list(identities)
    training_rows = np.asarray([rows_by_id[face_id] for face_id in training_ids])
    positive = np.asarray([identities[face_id] == person for face_id in training_ids])
    groups = [example_groups[face_id] for face_id in training_ids]
    validation_scores = grouped_validation_scores(
        embeddings[training_rows], positive, groups, regularization
    )
    coefficients = train_binary_ridge(
        embeddings[training_rows], positive, regularization
    )
    scores = predict_scores(embeddings, coefficients)
    training_set = set(training_ids)
    automatic, review = [], []
    for row, item in enumerate(items):
        if item["id"] in training_set or scores[row] < review_threshold:
            continue
        candidate = {
            "face_id": item["id"],
            "photo_id": item["photo_id"],
            "bbox": item["bbox"],
            "score": float(scores[row]),
        }
        (automatic if scores[row] >= assignment_threshold else review).append(candidate)
    automatic.sort(key=lambda candidate: candidate["score"], reverse=True)
    review.sort(key=lambda candidate: candidate["score"], reverse=True)
    validation = {
        "positive_count": int(positive.sum()),
        "negative_count": int((~positive).sum()),
        "true_positive": int(np.sum(positive & (validation_scores >= assignment_threshold))),
        "false_positive": int(np.sum((~positive) & (validation_scores >= assignment_threshold))),
    }
    payload = {
        "version": 1,
        "method": "class-balanced-binary-ridge",
        "person": person,
        "regularization": regularization,
        "assignment_threshold": assignment_threshold,
        "review_threshold": review_threshold,
        "confirmed_count": int(positive.sum()),
        "validation": validation,
        "automatic": automatic,
        "review": review,
    }
    for path in (model_output, assignments_output, audit_output):
        path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        model_output,
        coefficients=coefficients,
        person=np.asarray(person),
        assignment_threshold=np.asarray(assignment_threshold),
        review_threshold=np.asarray(review_threshold),
    )
    assignments_output.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    _write_audit(archive, audit_output, payload)
    print(
        f"Trained {person} from {positive.sum()} positive and {(~positive).sum()} negative faces; "
        f"assigned {len(automatic)} and placed {len(review)} in review.",
        flush=True,
    )
    return payload


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--archive", required=True, type=Path)
    result.add_argument("--labels", required=True, type=Path)
    result.add_argument("--person", required=True)
    result.add_argument("--regularization", type=float, default=1.0)
    result.add_argument("--assignment-threshold", type=float, default=0.20)
    result.add_argument("--review-threshold", type=float, default=0.0)
    return result


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    slug = args.person.casefold().replace(" ", "-")
    try:
        train_person_model(
            args.archive,
            args.labels,
            args.person,
            args.archive / "search" / f"{slug}-identity-model.npz",
            args.archive / "search" / f"{slug}-identity-assignments.json",
            args.archive / f"{slug}-identity-model.html",
            args.regularization,
            args.assignment_threshold,
            args.review_threshold,
        )
    except (OSError, ValueError, json.JSONDecodeError, np.linalg.LinAlgError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
