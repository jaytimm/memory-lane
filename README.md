# Memory Lane

Memory Lane is a local-first image gallery with album browsing and semantic
search.

The repository contains the gallery builder and launchers. User images,
metadata, search indexes, and model files are not included.

## Development

Build a gallery:

```bash
python3 -m src.build --input "/path/to/images" --output ./PhotoArchive
```

Run the tests:

```bash
python3 -m unittest discover -s tests -v
```

Build a local face-cluster review page from an archive that already has a face
index:

```bash
python3 -m src.cluster_faces --archive ./PhotoArchive
```

The generated `face-cluster-evaluation.html` is a curator tool, not part of the
family-facing gallery. It presents conservative face clusters for one-time
labeling. Reusing the same name across clusters creates one person with several
internal representations. Progress is saved in browser storage and can be
exported as JSON. It does not modify photographs.

After a cluster is named, **Recover more faces** ranks every face that did not
enter a conservative cluster against that person's labeled clusters and accepted
recovery batches. Correct faces can be added in batches, false matches can be
hidden, and the most recent accepted batch can be undone. Recovery decisions are
saved alongside the cluster labels and included in the JSON export.

Build a read-only audit of automatic assignments for one named person:

```bash
python3 -m src.audit_identity --archive ./PhotoArchive \
  --labels ./people-cluster-labels.json --person Anthony
```

The audit separates strong automatic candidates from a lower-confidence review
zone and compares each face with every other named person before proposing an
assignment.

Normalize spelling aliases in a completed export before treating it as the gold
label set:

```bash
python3 -m src.finalize_labels --labels ./people-cluster-labels.json \
  --output ./PhotoArchive/people-labels-gold.json \
  --alias Jackiea=Jackie --alias Jeffe=Jeff
```

Train a binary identity classifier for one person and apply it across the full
face index. Other confirmed people serve only as negative examples:

```bash
python3 -m src.train_person --archive ./PhotoArchive \
  --labels ./people-cluster-labels.json --person Anthony
```

The compact model is trained only from confirmed clusters and recovery batches.
It writes a model, assignment index, and visual audit page with automatic matches
and a lower-scoring review band. Predictions remain separate from training
examples and are never fed back into the model automatically.

The Windows companion is built by the workflow in `.github/workflows`.
