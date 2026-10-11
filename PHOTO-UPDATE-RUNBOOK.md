# Photo update runbook

Use this procedure whenever photographs in the USB drive's `Family Photos/`
folder have been added, removed, renamed, or corrected. The USB copy is the
authoritative source. `FamilyPhotoArchive/photos/` is an older local source copy
and must not be used to decide what changed.

## Important: do not repeat the read-only detour

The normal restricted development shell deliberately presents external media
under `/media/` as read-only. A non-escalated `findmnt` can therefore report
`ro` even while the actual host mount is healthy and writable.

- Do not diagnose or repair the USB based on a sandboxed `findmnt` result.
- Check real mount flags only in an approved host-side command.
- The expected host result is `rw`.
- Run `fsck.exfat` only if the host mount, copy operation, or kernel log reports
  a real error. Do not infer an exFAT problem from the sandbox view.

## Safe incremental update

1. Find the mounted USB and confirm these folders exist:

   ```text
   USB/
     Family Photos/
     Memory Lane/
   ```

2. Compare `Family Photos/` with
   `FamilyPhotoArchive/assets/manifest.js` and `.build-state.json`. Report
   additions, removals, renames, and same-path content changes before writing.
   Hash only candidates whose size or modification time changed.

3. Create a temporary staging gallery. Seed its generated directories from the
   current local `FamilyPhotoArchive`, and link `stage/photos` to the USB's
   `Family Photos/` folder. Preserve these local items outside the generated
   sync:

   - `FamilyPhotoArchive/photos/`
   - `FamilyPhotoArchive/gold-face-labels/`
   - evaluation files

4. Rebuild previews with one worker and low priority:

   ```bash
   OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
   NUMEXPR_NUM_THREADS=1 nice -n 15 ionice -c 3 \
   .venv/bin/python -m src.build \
     --input "/media/user/USB/Family Photos" \
     --output "$STAGE" --no-copy-originals --workers 1
   ```

   Stop if the reported reuse count does not match the preflight comparison.

5. Refresh semantic embeddings with one thread and batch size 1. The indexer
   must reuse unchanged `(id, sha256)` records and process only changed photos.

6. Refresh the face index with one OpenCV thread. It must reuse unchanged
   records and process only changed photos. A corrected orientation may change
   the total number of detected faces; that is expected.

7. Run `scripts/prune-gallery.py` with positional arguments to remove orphaned
   records and derivatives:

   ```bash
   .venv/bin/python scripts/prune-gallery.py \
     "$STAGE" "/media/user/USB/Family Photos"
   ```

8. Validate before copying:

   - manifest photo count equals the source count;
   - album counts sum to the photo count;
   - manifest has zero warnings;
   - preview and thumbnail counts equal the photo count;
   - semantic metadata count equals the embedding matrix rows;
   - face metadata count equals the face matrix rows;
   - every manifest photo resolves to a real USB original.

9. Synchronize only generated material back to the local cache:

   ```text
   .model-cache/  assets/  previews/  search/  thumbnails/
   .build-state.json  index.html  original-view.html  USER-GUIDE.html  README.txt
   ```

   Never broadly `rsync --delete` the entire `FamilyPhotoArchive`; that would
   endanger the preserved local photos, gold labels, and evaluation files.

10. Synchronize the same generated material to `USB/Memory Lane/Gallery/`.
    Do not modify `USB/Family Photos/` during this step. Flush writes, validate
    the USB-side counts again, then safely unmount and power off the drive.

## Orientation-only corrections

If corrected files differ only by EXIF orientation, they still require new
thumbnails, previews, semantic embeddings, and face processing because the
displayed orientation changed. A recent example involved 11 corrected files:
5,148 of 5,159 records were reused and only those 11 were processed.

