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
family-facing gallery. It first builds strict face clusters, then groups related
age, pose, and appearance clusters into broader candidate people. Names are
saved in browser storage while working and can be exported as JSON. Proposed
people can be rejected or split back into their strict subclusters. It does not
modify photographs.

The Windows companion is built by the workflow in `.github/workflows`.
