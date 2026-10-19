# Project: Offline Family Photo Gallery with Semantic Search

## 1. Objective

Build a portable, offline photo archive for approximately 5,000 digitized family photographs (~1.5 GB).

The archive should provide:

- A clean HTML gallery for browsing photos.
- Albums based on the existing directory structure.
- Semantic text-to-image search using pretrained CLIP embeddings.
- Optional filtering by album and date.
- Access to the original full-resolution photographs.

The eventual distribution medium is a USB drive shared with nontechnical family members.

**Primary requirement:** The gallery must remain usable even if semantic search is unavailable.

Do not build a cloud service or require user accounts.

## 2. Technical approach

Use Python for preprocessing and indexing, and HTML/CSS/JavaScript for the gallery.

Preferred technologies:

| Component | Technology |
|---|---|
| Photo ingestion | Python, Pillow |
| Embeddings | OpenCLIP |
| Semantic similarity | NumPy initially |
| Metadata | JSON or SQLite during indexing |
| Gallery | Static HTML, CSS, JavaScript |
| Thumbnails | WebP |
| Search model at distribution | Explore ONNX-based browser inference |
| Packaging | Portable directory suitable for USB |

Avoid unnecessary frameworks, database servers, and infrastructure.

The original photographs must never be modified.

## 3. Phase 1 — Photo ingestion and static gallery

Build a Python command-line tool that accepts an input directory containing nested photo folders.

### Ingestion

- Recursively discover supported image files.
- Support JPEG, PNG, WebP, and common TIFF scans.
- Preserve original relative paths and folder names.
- Assign a stable unique ID to each image.
- Extract available dates from EXIF metadata.
- Treat scan dates as potentially unreliable.
- Handle images with no dates.
- Correct image orientation for generated previews.
- Generate thumbnails, approximately 300–400 pixels wide.
- Generate an image manifest containing file paths, album names, dates, and thumbnail paths.
- Support incremental processing so unchanged files are skipped.

Use source file hashes or other stable identifiers rather than relying only on filenames.

### Gallery

Generate a static website with:

1. Album browsing based on existing folders.
2. A grid of photographs with lazy-loaded thumbnails.
3. An all-photos view.
4. A full-size image viewer with previous/next controls.
5. Album navigation.
6. A clear option to open the original image.
7. Basic date filtering when trustworthy dates exist.

The interface should be simple enough for someone unfamiliar with computers to use.

Avoid unnecessary settings and technical terminology.

### Offline requirements

The gallery must function without an internet connection.

Do not load fonts, JavaScript libraries, stylesheets, or images from remote servers.

The core gallery should work when opening `index.html` directly from a USB drive through a `file://` URL.

Avoid browser features that require an HTTP server for basic browsing.

Generate locally loadable JavaScript metadata if necessary rather than depending on browser `fetch()` access to local JSON files.

For large collections, use pagination or incremental rendering rather than displaying all 5,000 DOM elements simultaneously.

## 4. Phase 2 — Semantic indexing and retrieval

Implement text-to-image retrieval using a pretrained CLIP model.

### Image indexing

- Use a publicly available pretrained OpenCLIP checkpoint.
- Record the exact model architecture, weights identifier, and preprocessing configuration.
- Batch-process all discovered images.
- Generate one embedding per image.
- L2-normalize embeddings.
- Store embeddings in a NumPy array.
- Maintain a consistent mapping between embedding rows and image IDs.
- Save model and index version metadata.
- Support incremental indexing without reprocessing unchanged images.

Use the same model checkpoint for both image and text representations.

### Text search

Implement a Python search interface that:

1. Accepts a natural-language query.
2. Encodes it using the model's text encoder.
3. Normalizes the text embedding.
4. Computes cosine similarity against indexed image embeddings.
5. Returns the top K matching images.

Support optional album and date filters.

Apply metadata filters before final ranking where practical.

Example queries:

- "Children playing in the snow"
- "Christmas morning opening presents"
- "A family standing outside a house"
- "Birthday cakes and candles"
- "People swimming in a lake"
- "Dogs playing in the backyard"

Return image paths, scores, and associated metadata.

### Evaluation

Provide a simple visual evaluation tool that displays the top 20 results for a query.

Test search quality against a sample of actual digitized photographs.

Do not add captions, face recognition, OCR, or other models at this stage.

## 5. Phase 3 — Integrate semantic search into the gallery

The desired experience is a single photo gallery with a text search box.

Investigate running the CLIP text encoder locally in the browser using ONNX Runtime Web or a comparable runtime.

Requirements:

- Reuse the same embedding space as the precomputed image vectors.
- Package the tokenizer and model weights locally.
- Do not download model files at runtime.
- Do not send search queries or photos to external services.
- Run vector similarity calculations locally.
- Return thumbnails ranked by cosine similarity.
- Permit filtering results by album.
- Remain usable without a GPU.

**Important:** Verify that the model, tokenizer, and required runtime files can load in the intended USB environment.

Do not assume a browser can load ONNX models directly through `file://`.

Build an isolated proof of concept and test browser restrictions before integrating it into the gallery.

If fully portable browser inference is impractical, evaluate a small local launcher as an alternative.

The static gallery must continue to function independently.

## 6. Expected output structure

```text
FamilyPhotoArchive/
├── index.html
├── photos/
│   ├── Album A/
│   ├── Album B/
│   └── ...
├── thumbnails/
├── assets/
│   ├── gallery.css
│   ├── gallery.js
│   └── manifest.js
├── search/
│   ├── embeddings.bin
│   ├── index.json
│   ├── model/
│   └── search.js
└── README.txt
```

The build tooling should live separately from the distributable archive.

Example development structure:

```text
project/
├── src/
│   ├── ingest.py
│   ├── thumbnails.py
│   ├── embed.py
│   ├── search.py
│   └── build.py
├── web/
├── tests/
├── requirements.txt
└── README.md
```

## 7. CLI requirements

Provide commands equivalent to:

```bash
python -m src.build \
  --input "/path/to/photos" \
  --output "./FamilyPhotoArchive"

python -m src.embed \
  --archive "./FamilyPhotoArchive"

python -m src.search \
  --archive "./FamilyPhotoArchive" \
  --query "children playing outside" \
  --top-k 20
```

Commands should provide progress information and useful error messages.

Do not overwrite original photographs.

The build should be repeatable and support incremental updates.

## 8. Performance targets

Design for approximately 5,000 photographs and 1.5 GB of original image files.

Targets:

- Gallery opens locally without requiring an installation.
- Album navigation and thumbnail browsing remain responsive.
- Do not preload all full-resolution images.
- Search similarity computation is effectively instantaneous once the text query is embedded.
- The gallery remains functional offline.
- The archive can be copied to another USB drive without changing absolute paths.

Prefer relative paths everywhere.

The build should handle individual unreadable or unsupported files without aborting the entire process.

## 9. Testing

Include tests for:

- Nested folders and duplicate filenames.
- Unicode, spaces, and unusual characters in paths.
- EXIF orientation.
- Missing or unreliable dates.
- TIFF conversion for browser-compatible previews.
- Corrupt or unreadable files.
- Incremental indexing.
- Correct correspondence between image IDs and embedding rows.
- Normalized embedding vectors.
- Search result ordering.
- Gallery behavior without internet access.
- Moving the finished archive to a different directory or drive.

The gallery must work with sample data before attempting the full collection.

Test the final archive using an actual USB drive if available.

## 10. Acceptance criteria

### Phase 1 complete when:

- A directory of photographs can be converted into an offline HTML gallery.
- Existing folder organization is retained as albums.
- Thumbnails and original images are accessible.
- The gallery functions by opening `index.html`.
- No internet access or external software is required for browsing.

### Phase 2 complete when:

- The collection can be embedded using pretrained OpenCLIP.
- Natural-language queries return semantically relevant photos.
- Search results can be inspected visually.
- Embeddings can be reused without reprocessing the entire collection.
- Album/date filters work when metadata is available.

### Phase 3 complete when:

- Semantic search is available from the gallery interface.
- Search inference and similarity scoring run locally.
- No external network requests are required.
- Portability has been tested on the intended operating systems.
- A failure in search does not prevent ordinary photo browsing.

## 11. Implementation instructions

Work incrementally.

Start by inspecting the repository and establishing the project structure.

Then:

1. Implement ingestion and gallery generation.
2. Test on a small synthetic or sample photo collection.
3. Implement and evaluate CLIP semantic retrieval.
4. Experiment with browser-based text encoding.
5. Integrate search only after the browser inference approach has been demonstrated.
6. Package the final archive.

Use maintained libraries where practical, and keep dependencies minimal.

Document all assumptions, commands, and limitations.

Do not introduce face recognition, generative AI, cloud infrastructure, or a vector database unless there is a demonstrated need.

**First deliverable:** A working, fully offline HTML photo gallery generator with album navigation and thumbnails, together with a small CLIP semantic-search prototype that runs locally from Python.

Do not attempt full USB-compatible browser inference until both components work independently.
