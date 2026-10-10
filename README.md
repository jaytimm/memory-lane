# Memory Lane

Memory Lane is a local-first family photo gallery with album browsing and
natural-language search.

The repository contains the gallery builder and launchers. Family photographs,
archive metadata, search indexes, and model files are not included.

## Development

Build a gallery:

```bash
python3 -m src.build --input "/path/to/photos" --output ./FamilyPhotoArchive
```

Run the tests:

```bash
python3 -m unittest discover -s tests -v
```

The Windows companion is built by the workflow in `.github/workflows`.
