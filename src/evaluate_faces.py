"""Build a disposable, interactive evaluation of face similarity search.

This is intentionally a builder-side tool. OpenCV and the ONNX models are not
needed by the gallery after face embeddings have been generated.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from html import escape
import json
from pathlib import Path
import sys
from urllib.parse import quote, unquote

import cv2
import numpy as np

from .archive_data import read_gallery_manifest


def stratified_sample(photos: list[dict], limit: int) -> list[dict]:
    """Select photos evenly through every album, then fill any spare slots."""
    if limit >= len(photos):
        return list(photos)
    by_album: dict[str, list[dict]] = defaultdict(list)
    for photo in photos:
        by_album[photo["album_id"]].append(photo)

    album_count = len(by_album)
    target = max(1, limit // album_count)
    selected: list[dict] = []
    selected_ids: set[str] = set()
    for album_photos in by_album.values():
        count = min(target, len(album_photos))
        positions = np.linspace(0, len(album_photos) - 1, count, dtype=int)
        for position in positions:
            photo = album_photos[int(position)]
            if photo["id"] not in selected_ids:
                selected.append(photo)
                selected_ids.add(photo["id"])

    if len(selected) < limit:
        remaining = [photo for photo in photos if photo["id"] not in selected_ids]
        count = min(limit - len(selected), len(remaining))
        positions = np.linspace(0, len(remaining) - 1, count, dtype=int)
        selected.extend(remaining[int(position)] for position in positions)
    return selected[:limit]


def resized_for_detection(image: np.ndarray, max_side: int) -> tuple[np.ndarray, float]:
    height, width = image.shape[:2]
    scale = min(1.0, max_side / max(height, width))
    if scale == 1.0:
        return image, scale
    resized = cv2.resize(
        image, (round(width * scale), round(height * scale)), interpolation=cv2.INTER_AREA
    )
    return resized, scale


def original_face_row(face: np.ndarray, scale: float) -> np.ndarray:
    result = face.copy()
    result[:14] /= scale
    return result


def crop_face(image: np.ndarray, face: np.ndarray, padding: float = 0.45) -> np.ndarray:
    height, width = image.shape[:2]
    x, y, box_width, box_height = face[:4]
    pad_x, pad_y = box_width * padding, box_height * padding
    x1 = max(0, int(x - pad_x))
    y1 = max(0, int(y - pad_y))
    x2 = min(width, int(x + box_width + pad_x))
    y2 = min(height, int(y + box_height + pad_y))
    return image[y1:y2, x1:x2]


def _url(path: Path, relative_to: Path) -> str:
    return quote(path.resolve().relative_to(relative_to.resolve()).as_posix(), safe="/()._-~")


def build_face_evaluation(
    archive: Path,
    detector_model: Path,
    recognizer_model: Path,
    sample_size: int,
    output: Path,
    confidence: float = 0.70,
    max_side: int = 1280,
    date_from: int | None = None,
    date_to: int | None = None,
) -> Path:
    archive = archive.resolve()
    output = output.resolve()
    assets = output.parent / f"{output.stem}-assets"
    assets.mkdir(parents=True, exist_ok=True)

    manifest = read_gallery_manifest(archive)
    eligible_photos = [
        photo
        for photo in manifest["photos"]
        if (date_from is None or (photo.get("date_start") or 0) >= date_from)
        and (date_to is None or (photo.get("date_start") or 9999) <= date_to)
    ]
    if not eligible_photos:
        raise ValueError("No photos match the requested date range")
    photos = stratified_sample(eligible_photos, sample_size)
    detector = cv2.FaceDetectorYN.create(
        str(detector_model), "", (320, 320), confidence, 0.3, 5000
    )
    recognizer = cv2.FaceRecognizerSF.create(str(recognizer_model), "")

    faces: list[dict] = []
    vectors: list[np.ndarray] = []
    failed = 0
    for photo_number, photo in enumerate(photos, 1):
        source = archive / Path(unquote(photo["original"]))
        image = cv2.imread(str(source), cv2.IMREAD_COLOR)
        if image is None:
            failed += 1
            continue
        detection_image, scale = resized_for_detection(image, max_side)
        detector.setInputSize((detection_image.shape[1], detection_image.shape[0]))
        _, detections = detector.detect(detection_image)
        if detections is not None:
            for face_number, detected in enumerate(detections):
                if min(detected[2], detected[3]) < 22:
                    continue
                face = original_face_row(detected, scale)
                try:
                    aligned = recognizer.alignCrop(image, face)
                    vector = recognizer.feature(aligned).reshape(-1).astype(np.float32)
                except cv2.error:
                    continue
                norm = float(np.linalg.norm(vector))
                if not np.isfinite(norm) or norm == 0:
                    continue
                vector /= norm
                crop = crop_face(image, face)
                crop_name = f"{photo['id']}-{face_number}.jpg"
                cv2.imwrite(
                    str(assets / crop_name),
                    crop,
                    [cv2.IMWRITE_JPEG_QUALITY, 88],
                )
                faces.append(
                    {
                        "photo_id": photo["id"],
                        "filename": photo["filename"],
                        "album_id": photo["album_id"],
                        "year": photo.get("date_start"),
                        "preview": _url(archive / photo["preview"], output.parent),
                        "crop": _url(assets / crop_name, output.parent),
                        "score": round(float(detected[-1]), 4),
                    }
                )
                vectors.append(vector)
        if photo_number % 50 == 0 or photo_number == len(photos):
            print(
                f"Processed {photo_number:,}/{len(photos):,} photos; "
                f"found {len(faces):,} faces",
                flush=True,
            )

    if not vectors:
        raise RuntimeError("No faces were detected in the sample")
    matrix = np.stack(vectors)
    payload = {
        "sampledPhotos": len(photos),
        "failedPhotos": failed,
        "faces": faces,
        "vectors": np.round(matrix, 5).tolist(),
    }
    date_label = ""
    if date_from is not None or date_to is not None:
        date_label = f" · {date_from or 'earliest'}–{date_to or 'latest'}"
    title = f"Face-search evaluation{date_label}"
    document = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{title}</title><style>
:root{{--ink:#17201d;--muted:#68736e;--paper:#f2f3ee;--accent:#b94f32}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--paper);color:var(--ink);font-family:system-ui,sans-serif}}
header,main{{width:min(1500px,calc(100% - 36px));margin:auto}}header{{padding:32px 0 20px}}
h1{{font:500 clamp(2rem,5vw,3.5rem) Georgia,serif;margin:0 0 8px}}p{{color:var(--muted)}}
.picker,.results{{display:grid;grid-template-columns:repeat(auto-fill,minmax(115px,1fr));gap:10px}}
.picker{{max-height:42vh;overflow:auto;padding:3px 3px 18px}}
button{{border:0;background:white;padding:0;cursor:pointer;border-radius:8px;overflow:hidden;box-shadow:0 1px 3px #0002}}
button:hover,button.selected{{outline:3px solid var(--accent)}}img{{display:block;width:100%;aspect-ratio:1;object-fit:cover}}
.results figure{{margin:0;background:white;border-radius:8px;overflow:hidden}}.results a{{display:block}}
figcaption{{padding:8px;font-size:.75rem}}strong,span{{display:block;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}}
span{{color:var(--muted);margin-top:3px}}h2{{font:500 1.8rem Georgia,serif;border-top:1px solid #d5d9d3;padding-top:22px}}
.empty{{grid-column:1/-1;padding:30px;background:#fff;border-radius:8px}}.note{{max-width:850px}}
</style></head><body><header><h1>{title}</h1>
<p class="note">Click a detected face. The page ranks every other face by cosine similarity, then keeps the best face from each photo. This sample contains <b>{len(faces):,} faces</b> detected in <b>{len(photos):,} photos</b> spread across all albums.</p>
</header><main><h2>Choose a face</h2><div class="picker" id="picker"></div>
<h2 id="result-title">Nearest matches</h2><div class="results" id="results"><p class="empty">Choose any face above.</p></div></main>
<script>const DATA={json.dumps(payload, ensure_ascii=False, separators=(',', ':'))};
const picker=document.querySelector('#picker'),results=document.querySelector('#results'),titleEl=document.querySelector('#result-title');
const dot=(a,b)=>{{let s=0;for(let i=0;i<a.length;i++)s+=a[i]*b[i];return s}};
const info=f=>`${{f.album_id}}${{f.year?' · '+f.year:''}}`;
function choose(index,button){{document.querySelector('.selected')?.classList.remove('selected');button.classList.add('selected');
 const query=DATA.vectors[index],best=new Map();DATA.faces.forEach((f,i)=>{{if(i===index||f.photo_id===DATA.faces[index].photo_id)return;const score=dot(query,DATA.vectors[i]);if(!best.has(f.photo_id)||best.get(f.photo_id).score<score)best.set(f.photo_id,{{...f,score}})}});
 const ranked=[...best.values()].sort((a,b)=>b.score-a.score).slice(0,40);titleEl.textContent=`Nearest matches for ${{DATA.faces[index].filename}}`;
 results.innerHTML=ranked.map((f,i)=>`<figure><a href="${{f.preview}}" target="_blank"><img src="${{f.crop}}" loading="lazy"></a><figcaption><strong>${{i+1}}. ${{f.filename}}</strong><span>${{info(f)}} · ${{f.score.toFixed(3)}}</span></figcaption></figure>`).join('')}}
const hash=s=>{{let h=2166136261;for(let i=0;i<s.length;i++)h=Math.imul(h^s.charCodeAt(i),16777619);return h>>>0}};
DATA.faces.map((f,i)=>i).sort((a,b)=>hash(DATA.faces[a].photo_id+'-'+a)-hash(DATA.faces[b].photo_id+'-'+b)).forEach(i=>{{const f=DATA.faces[i],b=document.createElement('button');b.title=`${{f.filename}} · ${{info(f)}}`;b.innerHTML=`<img src="${{f.crop}}" loading="lazy">`;b.onclick=()=>choose(i,b);picker.appendChild(b)}});
</script></body></html>"""
    output.write_text(document, encoding="utf-8")
    print(f"Wrote {output}")
    return output


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--archive", required=True, type=Path)
    result.add_argument("--detector-model", required=True, type=Path)
    result.add_argument("--recognizer-model", required=True, type=Path)
    result.add_argument("--sample-size", type=int, default=900)
    result.add_argument("--confidence", type=float, default=0.70)
    result.add_argument("--max-side", type=int, default=1280)
    result.add_argument("--date-from", type=int)
    result.add_argument("--date-to", type=int)
    result.add_argument("--output", type=Path)
    return result


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if args.sample_size < 1 or args.max_side < 320 or not 0 < args.confidence <= 1:
        print("Error: invalid sampling or detection arguments", file=sys.stderr)
        return 2
    if args.date_from is not None and args.date_to is not None and args.date_from > args.date_to:
        print("Error: --date-from must not be later than --date-to", file=sys.stderr)
        return 2
    output = args.output or args.archive / "face-search-evaluation.html"
    try:
        build_face_evaluation(
            args.archive,
            args.detector_model,
            args.recognizer_model,
            args.sample_size,
            output,
            args.confidence,
            args.max_side,
            args.date_from,
            args.date_to,
        )
    except (OSError, RuntimeError, ValueError, cv2.error) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
