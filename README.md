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

Opening the gallery's `index.html` directly, without the launcher, still
supports browsing, **Find similar photos**, **Find this person**, and original
photographs. Typed search needs the launcher. The embedding, face, and prune
steps keep the browser search data current; to regenerate it by hand:

```bash
python3 -m src.browser_index --archive ./PhotoArchive
```

Run the tests:

```bash
python3 -m unittest discover -s tests -v
```

The Windows companion is built by the workflow in `.github/workflows`.

## Assemble a USB drive

Keep the originals in `Family Photos/` and all generated app material in
`Memory Lane/`. After the album folders have been copied, assemble the app side
from Linux with:

```bash
./scripts/assemble-usb.sh "/media/user/USB" ./MemoryLane-Windows-x64.zip
```

The script copies the prepared gallery cache without duplicating the original
photographs. The Windows launcher, ZIP, version, and start instructions are kept
together inside `Memory Lane/`.

For later photo additions and corrections, follow
[`PHOTO-UPDATE-RUNBOOK.md`](PHOTO-UPDATE-RUNBOOK.md). It documents the safe
single-worker incremental rebuild and the external-drive sandbox caveat.
