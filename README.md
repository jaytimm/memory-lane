# Family Photo Archive

This project builds a portable, offline HTML gallery from nested photo folders.
It preserves source files, generates correctly oriented thumbnails and previews,
and imports album descriptions from `Album Contents.xlsx` when present.

## Build

Python 3.10+ and Pillow are required.

```bash
python3 -m src.build \
  --input "/path/to/photos" \
  --output "./FamilyPhotoArchive"
```

For a smaller review build, repeat `--album` with text found in folder names:

```bash
python3 -m src.build \
  --input "/path/to/photos" \
  --output "./FamilyPhotoArchive-sample" \
  --album "1969" \
  --album "1976" \
  --album "2024"
```

Open `index.html` directly in a browser. The gallery does not use a server,
network connection, remote font, or external JavaScript library.

Builds are incremental. Unchanged source images reuse their hashes and generated
previews. Exact duplicate files in different folders receive distinct IDs while
retaining the content hash as the leading part of the ID.

## Test

```bash
python3 -m unittest discover -v
```

The source directory is only read. Originals copied into the generated archive
are byte-for-byte copies and are never edited in place.

## Semantic search

Semantic indexing uses OpenCLIP `ViT-B-32` with the
`laion2b_s34b_b79k` checkpoint. The image and text encoders always use the same
checkpoint, and the index records its package versions and preprocessing values.

Create the isolated environment and install CPU-only PyTorch plus OpenCLIP:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install \
  --index-url https://download.pytorch.org/whl/cpu \
  torch==2.9.1 torchvision==0.24.1
.venv/bin/python -m pip install open_clip_torch==3.3.0
```

Build or incrementally update an image index:

```bash
.venv/bin/python -m src.embed --archive ./FamilyPhotoArchive
```

Search with optional album and year filters:

```bash
.venv/bin/python -m src.search \
  --archive ./FamilyPhotoArchive \
  --query "children playing outside" \
  --top-k 20
```

Create a visual evaluation page for the standard query set:

```bash
.venv/bin/python -m src.evaluate --archive ./FamilyPhotoArchive
```

The first indexing command downloads the selected model weights and preserves
them in `search/model/open_clip_model.safetensors`. Later indexing and search
prefer that standalone file; `.model-cache` is only a download cache and does
not need to travel with the finished archive.

## Launch the searchable gallery

The static gallery still opens directly without search. To keep the CLIP text
model loaded and add live semantic search to the same interface, run:

```bash
./start-gallery.sh
```

or equivalently:

```bash
.venv/bin/python -m src.serve --archive ./FamilyPhotoArchive
```

The launcher binds only to `127.0.0.1`, opens the gallery in the default browser,
and serves search locally. Queries and photographs are never sent elsewhere.
Press Ctrl+C in the launcher terminal to stop it.

## Build the portable Windows launcher

The manual GitHub Actions workflow in
`.github/workflows/build-windows-portable.yml` builds a Windows x64 runtime
without uploading any photographs. Run **Build portable Windows gallery
search** from the repository's Actions tab, then download the
`FamilyGallery-Windows-x64` artifact.

Extract the artifact so these items are next to each other:

```text
Start Family Gallery.bat
WindowsSearch/
FamilyPhotoArchive/
```

The archive must contain
`search/model/open_clip_model.safetensors`. The embedding command now preserves
that checkpoint as a regular file rather than relying on cache symlinks, making
it safe to copy to FAT-formatted USB drives and load on Windows without network
access.
