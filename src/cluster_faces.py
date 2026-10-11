"""Build a local contact-sheet evaluator for conservative face clusters."""

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


def nearest_faces(
    embeddings: np.ndarray,
    photo_ids: list[str],
    neighbors: int = 6,
    block_size: int = 512,
) -> tuple[np.ndarray, np.ndarray]:
    """Return each face's nearest faces, excluding faces in the same photograph."""
    count = len(embeddings)
    if embeddings.ndim != 2 or count != len(photo_ids):
        raise ValueError("Face metadata and embeddings do not match")
    if count < 2 or neighbors < 1 or block_size < 1:
        raise ValueError("At least two faces and positive search settings are required")
    neighbor_count = min(neighbors, count - 1)
    indices = np.empty((count, neighbor_count), dtype=np.int32)
    scores = np.empty((count, neighbor_count), dtype=np.float32)
    rows_by_photo: dict[str, list[int]] = {}
    for row, photo_id in enumerate(photo_ids):
        rows_by_photo.setdefault(photo_id, []).append(row)

    for start in range(0, count, block_size):
        end = min(count, start + block_size)
        similarities = embeddings[start:end] @ embeddings.T
        for row in range(start, end):
            similarities[row - start, rows_by_photo[photo_ids[row]]] = -np.inf
        selected = np.argpartition(similarities, -neighbor_count, axis=1)[
            :, -neighbor_count:
        ]
        selected_scores = np.take_along_axis(similarities, selected, axis=1)
        order = np.argsort(selected_scores, axis=1)[:, ::-1]
        indices[start:end] = np.take_along_axis(selected, order, axis=1)
        scores[start:end] = np.take_along_axis(selected_scores, order, axis=1)
    return indices, scores


def conservative_clusters(
    items: list[dict],
    embeddings: np.ndarray,
    threshold: float = 0.72,
    neighbors: int = 6,
    minimum_size: int = 3,
    block_size: int = 512,
) -> list[dict]:
    """Join reciprocal close neighbors and return stable candidate clusters."""
    if not 0 < threshold <= 1 or minimum_size < 2:
        raise ValueError("Invalid clustering threshold or minimum size")
    photo_ids = [item["photo_id"] for item in items]
    nearest, scores = nearest_faces(embeddings, photo_ids, neighbors, block_size)
    count = len(items)
    parent = np.arange(count, dtype=np.int32)
    sizes = np.ones(count, dtype=np.int32)

    def find(row: int) -> int:
        while parent[row] != row:
            parent[row] = parent[parent[row]]
            row = int(parent[row])
        return row

    def union(first: int, second: int) -> None:
        first_root, second_root = find(first), find(second)
        if first_root == second_root:
            return
        if sizes[first_root] < sizes[second_root]:
            first_root, second_root = second_root, first_root
        parent[second_root] = first_root
        sizes[first_root] += sizes[second_root]

    reciprocal = [set(map(int, row)) for row in nearest]
    edge_scores: dict[tuple[int, int], float] = {}
    for first in range(count):
        for position, raw_second in enumerate(nearest[first]):
            second = int(raw_second)
            score = float(scores[first, position])
            if score < threshold or first not in reciprocal[second]:
                continue
            pair = (min(first, second), max(first, second))
            edge_scores[pair] = score
            union(first, second)

    members_by_root: dict[int, list[int]] = {}
    for row in range(count):
        members_by_root.setdefault(find(row), []).append(row)

    clusters: list[dict] = []
    for members in members_by_root.values():
        if len(members) < minimum_size:
            continue
        member_set = set(members)
        internal_scores = [
            score
            for (first, second), score in edge_scores.items()
            if first in member_set and second in member_set
        ]
        centroid = embeddings[members].mean(axis=0)
        norm = float(np.linalg.norm(centroid))
        cohesion = embeddings[members] @ (centroid / norm) if norm else np.zeros(len(members))
        face_ids = sorted(items[row]["id"] for row in members)
        cluster_id = hashlib.sha1("\n".join(face_ids).encode()).hexdigest()[:12]
        clusters.append(
            {
                "id": cluster_id,
                "members": members,
                "size": len(members),
                "edge_min": min(internal_scores) if internal_scores else 0.0,
                "edge_median": float(np.median(internal_scores)) if internal_scores else 0.0,
                "cohesion_min": float(np.min(cohesion)),
                "cohesion_median": float(np.median(cohesion)),
            }
        )
    clusters.sort(key=lambda cluster: (-cluster["size"], cluster["id"]))
    return clusters


def _asset_url(path: Path, output: Path) -> str:
    relative = os.path.relpath(path.resolve(), output.parent.resolve())
    return quote(Path(relative).as_posix(), safe="/()._-~")


def build_cluster_evaluation(
    archive: Path,
    output: Path,
    threshold: float = 0.72,
    neighbors: int = 6,
    minimum_size: int = 3,
    block_size: int = 512,
) -> Path:
    archive, output = archive.resolve(), output.resolve()
    manifest = read_gallery_manifest(archive)
    face_metadata, embeddings = read_face_index(archive)
    items = face_metadata["items"]
    photos = {photo["id"]: photo for photo in manifest["photos"]}
    print(f"Finding reciprocal neighbors among {len(items):,} faces...", flush=True)
    clusters = conservative_clusters(
        items, embeddings, threshold, neighbors, minimum_size, block_size
    )

    payload_clusters = []
    for cluster in clusters:
        members = []
        for row in cluster["members"]:
            face = items[row]
            photo = photos.get(face["photo_id"])
            if not photo:
                continue
            members.append(
                {
                    "face_id": face["id"],
                    "photo_id": face["photo_id"],
                    "bbox": face["bbox"],
                    "preview": _asset_url(archive / photo["preview"], output),
                    "filename": photo["filename"],
                    "album": photo["album_id"],
                    "year": photo.get("date_start"),
                    "width": photo.get("width") or 1,
                    "height": photo.get("height") or 1,
                }
            )
        payload_clusters.append(
            {
                **{key: value for key, value in cluster.items() if key != "members"},
                "members": members,
            }
        )

    payload = {
        "archive": archive.name,
        "faceCount": len(items),
        "threshold": threshold,
        "neighbors": neighbors,
        "minimumSize": minimum_size,
        "clusteredFaces": sum(cluster["size"] for cluster in clusters),
        "clusters": payload_clusters,
    }
    title = "Face cluster review"
    document = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{escape(title)}</title><style>
:root{{--ink:#17201d;--muted:#68736e;--paper:#f2f3ee;--card:#fff;--accent:#b94f32;--green:#173d36}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--paper);color:var(--ink);font:16px/1.45 system-ui,sans-serif}}
button,input,select{{font:inherit}}header{{padding:25px clamp(18px,4vw,55px);background:#fff;border-bottom:1px solid #d9ddd8}}
h1{{font:500 clamp(2rem,5vw,3.8rem)/1 Georgia,serif;margin:0 0 9px}}p{{color:var(--muted);margin:5px 0}}
.toolbar{{display:flex;flex-wrap:wrap;gap:10px;margin-top:18px}}button,.button{{border:0;border-radius:8px;padding:10px 14px;background:var(--green);color:#fff;cursor:pointer}}
button.secondary{{background:#e5e8e3;color:var(--ink)}}button.danger{{background:#8a3d32}}button:disabled{{opacity:.45}}
main{{display:grid;grid-template-columns:minmax(260px,340px) 1fr;min-height:calc(100vh - 180px)}}
aside{{border-right:1px solid #d9ddd8;padding:14px;max-height:calc(100vh - 180px);overflow:auto}}
.cluster-list{{display:grid;gap:9px}}.cluster-button{{display:grid;grid-template-columns:54px 1fr;gap:10px;text-align:left;background:#fff;color:var(--ink);padding:7px;width:100%}}
.cluster-button.active{{outline:3px solid var(--accent)}}.mini{{width:54px;height:54px;border-radius:6px;overflow:hidden;background:#ddd}}
.cluster-button span,.cluster-button small{{display:block;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}}small{{color:var(--muted)}}
.review{{padding:clamp(18px,3vw,42px);min-width:0}}.review-head{{display:flex;flex-wrap:wrap;justify-content:space-between;gap:15px;align-items:start}}
h2{{font:500 clamp(1.6rem,4vw,2.6rem) Georgia,serif;margin:0}}.label-form{{display:flex;flex-wrap:wrap;gap:8px;margin:20px 0;padding:14px;background:#fff;border-radius:10px}}
input,select{{border:1px solid #bcc4bd;border-radius:7px;padding:9px;background:#fff}}input{{min-width:220px;flex:1}}
.faces{{display:grid;grid-template-columns:repeat(auto-fill,minmax(125px,1fr));gap:11px}}
figure{{margin:0;background:#fff;border-radius:9px;overflow:hidden}}figure a{{display:block}}svg{{display:block;width:100%;aspect-ratio:1;background:#ddd}}
figcaption{{padding:8px;font-size:.75rem}}figcaption strong,figcaption span{{display:block;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}}figcaption span{{color:var(--muted)}}
.empty{{padding:35px;background:#fff;border-radius:10px}}.badge{{display:inline-block;padding:3px 8px;border-radius:99px;background:#e5e8e3;margin-left:5px;font-size:.8rem}}
@media(max-width:760px){{main{{display:block}}aside{{max-height:35vh;border-right:0;border-bottom:1px solid #d9ddd8}}}}
</style></head><body><header><h1>{escape(title)}</h1>
<p><b>{len(clusters):,} candidate clusters</b> containing <b>{payload['clusteredFaces']:,}</b> of {len(items):,} faces. Reciprocal top-{neighbors} neighbors; minimum similarity {threshold:.2f}; minimum cluster size {minimum_size}.</p>
<p>Name or reject each cluster once. Decisions stay in this browser until exported; photographs are not modified.</p>
<div class="toolbar"><button id="export">Export labels</button><button id="show-unreviewed" class="secondary">Unreviewed</button><button id="show-all" class="secondary">All clusters</button></div>
</header><main><aside><div class="cluster-list" id="cluster-list"></div></aside>
<section class="review" id="review"><p class="empty">Choose a cluster to inspect.</p></section></main>
<script>const DATA={json.dumps(payload, ensure_ascii=False, separators=(',', ':'))};
const KEY=`memory-lane-clusters:${{DATA.archive}}:${{DATA.threshold}}:${{DATA.neighbors}}`,labels=JSON.parse(localStorage.getItem(KEY)||'{{}}');
const list=document.querySelector('#cluster-list'),review=document.querySelector('#review');let current=null,onlyUnreviewed=true;
const esc=s=>String(s??'').replace(/[&<>"']/g,c=>({{'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}}[c]));
function crop(face,mini=false){{const [bx,by,bw,bh]=face.bbox,p=.42,x=Math.max(0,bx-bw*p),y=Math.max(0,by-bh*p),w=Math.min(1-x,bw*(1+2*p)),h=Math.min(1-y,bh*(1+2*p));return `<svg class="${{mini?'mini':''}}" viewBox="${{x*face.width}} ${{y*face.height}} ${{w*face.width}} ${{h*face.height}}" preserveAspectRatio="xMidYMid slice"><image href="${{face.preview}}" width="${{face.width}}" height="${{face.height}}"/></svg>`}}
function save(){{localStorage.setItem(KEY,JSON.stringify(labels));renderList();}}
function renderList(){{list.innerHTML='';DATA.clusters.filter(c=>!onlyUnreviewed||!labels[c.id]).forEach(c=>{{const label=labels[c.id],b=document.createElement('button');b.className='cluster-button'+(current===c.id?' active':'');b.innerHTML=crop(c.members[0],true)+`<span><b>${{esc(label?.name||(label?.rejected?'Rejected':'Cluster '+c.id))}}</b><small>${{c.size}} faces · cohesion ${{c.cohesion_median.toFixed(2)}}</small></span>`;b.onclick=()=>show(c.id);list.append(b)}});if(!list.children.length)list.innerHTML='<p>Every candidate cluster has been reviewed.</p>'}}
function show(id){{current=id;const c=DATA.clusters.find(x=>x.id===id),label=labels[id]||{{}};renderList();review.innerHTML=`<div class="review-head"><div><h2>${{esc(label.name||'Unlabeled cluster')}} <span class="badge">${{c.size}} faces</span></h2><p>Median cohesion ${{c.cohesion_median.toFixed(3)}} · weakest member ${{c.cohesion_min.toFixed(3)}} · cluster ${{c.id}}</p></div><div><button id="previous" class="secondary">← Previous</button> <button id="next" class="secondary">Next →</button></div></div><div class="label-form"><input id="person-name" placeholder="Person's name" value="${{esc(label.name||'')}}"><select id="variant"><option value="">No profile label</option>${{['young','adult','profile','other'].map(v=>`<option value="${{v}}" ${{label.variant===v?'selected':''}}>${{v[0].toUpperCase()+v.slice(1)}}</option>`).join('')}}</select><button id="save-label">Name cluster</button><button id="reject" class="danger">Reject cluster</button></div><div class="faces">${{c.members.slice(0,160).map(f=>`<figure><a href="${{f.preview}}" target="_blank">${{crop(f)}}</a><figcaption><strong>${{esc(f.filename)}}</strong><span>${{esc(f.album)}}${{f.year?' · '+f.year:''}}</span></figcaption></figure>`).join('')}}</div>${{c.size>160?`<p>Showing the first 160 of ${{c.size}} faces.</p>`:''}}`;document.querySelector('#save-label').onclick=()=>{{const name=document.querySelector('#person-name').value.trim();if(!name)return;labels[id]={{name,variant:document.querySelector('#variant').value,rejected:false}};save();show(id)}};document.querySelector('#reject').onclick=()=>{{labels[id]={{rejected:true}};save();show(id)}};const at=DATA.clusters.findIndex(x=>x.id===id);document.querySelector('#previous').onclick=()=>show(DATA.clusters[(at-1+DATA.clusters.length)%DATA.clusters.length].id);document.querySelector('#next').onclick=()=>show(DATA.clusters[(at+1)%DATA.clusters.length].id)}}
document.querySelector('#show-unreviewed').onclick=()=>{{onlyUnreviewed=true;renderList()}};document.querySelector('#show-all').onclick=()=>{{onlyUnreviewed=false;renderList()}};
document.querySelector('#export').onclick=()=>{{const clusters=Object.entries(labels).map(([cluster_id,label])=>({{cluster_id,...label,face_ids:DATA.clusters.find(c=>c.id===cluster_id)?.members.map(f=>f.face_id)||[]}}));const blob=new Blob([JSON.stringify({{version:1,archive:DATA.archive,settings:{{threshold:DATA.threshold,neighbors:DATA.neighbors,minimum_size:DATA.minimumSize}},clusters}},null,2)],{{type:'application/json'}}),a=document.createElement('a');a.href=URL.createObjectURL(blob);a.download='people-cluster-labels.json';a.click();URL.revokeObjectURL(a.href)}};
renderList();if(DATA.clusters.length)show(DATA.clusters[0].id);</script></body></html>"""
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(document, encoding="utf-8")
    print(f"Wrote {output}")
    return output


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--archive", required=True, type=Path)
    result.add_argument("--output", type=Path)
    result.add_argument("--threshold", type=float, default=0.72)
    result.add_argument("--neighbors", type=int, default=6)
    result.add_argument("--minimum-size", type=int, default=3)
    result.add_argument("--block-size", type=int, default=512)
    return result


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    output = args.output or args.archive / "face-cluster-evaluation.html"
    try:
        build_cluster_evaluation(
            args.archive,
            output,
            args.threshold,
            args.neighbors,
            args.minimum_size,
            args.block_size,
        )
    except (OSError, RuntimeError, ValueError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
