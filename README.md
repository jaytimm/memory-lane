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

The Windows companion is built by the workflow in `.github/workflows`.
