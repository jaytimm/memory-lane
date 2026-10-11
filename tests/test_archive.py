from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from zipfile import ZIP_DEFLATED, ZipFile

from PIL import Image

from src.build import build_archive
from src.cluster_faces import conservative_clusters, nearest_faces
from src.embed import _existing_vectors, _save_partial
from src.ingest import album_year_range, stable_image_id
from src.metadata import match_records_to_folders, read_album_records
from src.search import (
    FaceSearchIndex,
    SemanticSearchEngine,
    _candidate_indices,
    rank_embeddings,
)
from src.serve import (
    parse_face_search_request,
    parse_photo_faces_request,
    parse_search_request,
    parse_similar_request,
)


class IngestTests(unittest.TestCase):
    def test_album_year_ranges_include_century_rollover(self):
        self.assertEqual(album_year_range("1976_WEDDING"), (1976, 1976))
        self.assertEqual(album_year_range("1999_12.99-06.00"), (1999, 2000))
        self.assertEqual(album_year_range("2018_5.18-1.19"), (2018, 2019))
        self.assertEqual(album_year_range("undated"), (None, None))

    def test_duplicate_content_in_different_paths_has_unique_id(self):
        digest = "a" * 64
        first = stable_image_id(digest, "Album A/photo.jpg")
        second = stable_image_id(digest, "Album B/photo.jpg")
        self.assertNotEqual(first, second)
        self.assertTrue(first.startswith(digest[:20]))


class MetadataTests(unittest.TestCase):
    def _write_workbook(self, path: Path):
        strings = ["Alb", "No", "Years", "Content", "1976 Wedding", "Our Wedding", "more notes"]
        shared_items = "".join(f"<si><t>{value}</t></si>" for value in strings)
        shared = (
            '<?xml version="1.0" encoding="UTF-8"?>'
            '<sst xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
            f"{shared_items}</sst>"
        )
        sheet = (
            '<?xml version="1.0" encoding="UTF-8"?>'
            '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"><sheetData>'
            '<row r="1"><c r="A1" t="s"><v>0</v></c><c r="B1" t="s"><v>1</v></c>'
            '<c r="C1" t="s"><v>2</v></c><c r="D1" t="s"><v>3</v></c></row>'
            '<row r="2"><c r="A2"><v>3</v></c><c r="B2"><v>54</v></c>'
            '<c r="C2" t="s"><v>4</v></c><c r="D2" t="s"><v>5</v></c></row>'
            '<row r="3"><c r="D3" t="s"><v>6</v></c></row>'
            '</sheetData></worksheet>'
        )
        with ZipFile(path, "w", ZIP_DEFLATED) as archive:
            archive.writestr("xl/sharedStrings.xml", shared)
            archive.writestr("xl/worksheets/sheet1.xml", sheet)

    def test_spreadsheet_wrapped_rows_and_folder_match(self):
        with tempfile.TemporaryDirectory() as temporary:
            workbook = Path(temporary) / "albums.xlsx"
            self._write_workbook(workbook)
            records = read_album_records(workbook)
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0].expected_count, 54)
        self.assertEqual(records[0].description, "Our Wedding more notes")
        matches = match_records_to_folders(records, ["1976_WEDDING"])
        self.assertEqual(matches["1976_WEDDING"].number, "3")


class BuildIntegrationTests(unittest.TestCase):
    def test_build_handles_paths_orientation_tiff_corruption_and_incremental_reuse(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            album = source / "1976_WEDDING"
            nested = album / "Unicode café 深い"
            nested.mkdir(parents=True)

            Image.new("RGB", (30, 20), "red").save(album / "same name.jpg")
            Image.new("RGB", (20, 30), "blue").save(nested / "same name.jpg")
            Image.new("RGB", (24, 16), "green").save(album / "no date.png")
            Image.new("RGB", (18, 12), "purple").save(album / "scan.tiff")

            oriented = Image.new("RGB", (40, 20), "orange")
            exif = Image.Exif()
            exif[274] = 6
            oriented.save(album / "oriented.jpg", exif=exif)
            (album / "broken.jpg").write_bytes(b"not an image")

            output = root / "archive"
            first = build_archive(source, output, workers=3)
            self.assertEqual(len(first["photos"]), 5)
            self.assertEqual(len({photo["id"] for photo in first["photos"]}), 5)
            self.assertEqual(len(first["warnings"]), 1)
            self.assertTrue((output / "index.html").exists())
            self.assertTrue((output / "USER-GUIDE.html").exists())
            self.assertTrue((output / "assets/search-examples.js").exists())
            self.assertTrue((output / "photos/1976_WEDDING/scan.tiff").exists())

            unicode_record = next(
                photo for photo in first["photos"] if "Unicode" in photo["relative_source_path"]
            )
            self.assertIn("caf%C3%A9", unicode_record["original"])
            no_date = next(photo for photo in first["photos"] if photo["filename"] == "no date.png")
            self.assertIsNone(no_date["exif_date"])
            self.assertEqual(no_date["date_start"], 1976)

            oriented_record = next(photo for photo in first["photos"] if photo["filename"] == "oriented.jpg")
            with Image.open(output / oriented_record["thumbnail"]) as generated:
                self.assertGreater(generated.height, generated.width)

            thumbnail = output / first["photos"][0]["thumbnail"]
            before = thumbnail.stat().st_mtime_ns
            second = build_archive(source, output)
            self.assertEqual(len(second["photos"]), 5)
            self.assertEqual(thumbnail.stat().st_mtime_ns, before)

            manifest = (output / "assets/manifest.js").read_text(encoding="utf-8")
            payload = manifest.removeprefix("window.FAMILY_PHOTO_DATA = ").removesuffix(";\n")
            self.assertEqual(len(json.loads(payload)["photos"]), 5)


class SemanticRankingTests(unittest.TestCase):
    def test_conservative_face_clusters_require_reciprocal_neighbors(self):
        import numpy as np

        items = [
            {"id": "a", "photo_id": "one"},
            {"id": "b", "photo_id": "two"},
            {"id": "c", "photo_id": "three"},
            {"id": "d", "photo_id": "four"},
        ]
        embeddings = np.asarray(
            [[1, 0], [.99, .01], [.98, .02], [0, 1]], dtype=np.float32
        )
        embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)
        clusters = conservative_clusters(
            items, embeddings, threshold=.95, neighbors=2, minimum_size=3, block_size=2
        )
        self.assertEqual(len(clusters), 1)
        self.assertEqual(clusters[0]["members"], [0, 1, 2])

    def test_nearest_faces_excludes_faces_from_same_photo(self):
        import numpy as np

        embeddings = np.asarray([[1, 0], [1, 0], [.8, .2]], dtype=np.float32)
        nearest, _ = nearest_faces(embeddings, ["one", "one", "two"], neighbors=1)
        self.assertEqual(nearest[:2, 0].tolist(), [2, 2])
        self.assertIn(int(nearest[2, 0]), (0, 1))

    def test_cosine_ranking_and_metadata_filters(self):
        import numpy as np

        embeddings = np.asarray([[1.0, 0.0], [0.8, 0.2], [0.0, 1.0]], dtype=np.float32)
        query = np.asarray([1.0, 0.0], dtype=np.float32)
        ranked = rank_embeddings(embeddings, query, np.asarray([0, 1, 2]), top_k=2)
        self.assertEqual([index for index, _ in ranked], [0, 1])

        items = [
            {"album_id": "1976_WEDDING", "date_start": 1976, "date_end": 1976},
            {"album_id": "1999_12.99-06.00", "date_start": 1999, "date_end": 2000},
            {"album_id": "2024_10.24-10.26", "date_start": 2024, "date_end": 2026},
        ]
        candidates = _candidate_indices(items, "wedding", 1970, 1980)
        self.assertEqual(candidates.tolist(), [0])
        rollover = _candidate_indices(items, None, 2000, 2000)
        self.assertEqual(rollover.tolist(), [1])

    def test_partial_embedding_checkpoint_can_resume(self):
        import numpy as np

        identity = {"architecture": "test-model"}
        photos = [
            {"id": "one", "sha256": "aaa"},
            {"id": "two", "sha256": "bbb"},
            {"id": "three", "sha256": "ccc"},
        ]
        vectors = [
            np.asarray([1.0, 0.0], dtype=np.float32),
            None,
            np.asarray([0.0, 1.0], dtype=np.float32),
        ]
        with tempfile.TemporaryDirectory() as temporary:
            search_dir = Path(temporary)
            _save_partial(search_dir, identity, photos, vectors)
            restored = _existing_vectors(search_dir, identity)
        self.assertEqual(set(restored), {("one", "aaa"), ("three", "ccc")})
        self.assertTrue(np.array_equal(restored[("one", "aaa")], vectors[0]))

    def test_launcher_request_validation_and_limits(self):
        parsed = parse_search_request(
            {"query": "  a parrot  ", "top_k": 500, "date_from": 1980}
        )
        self.assertEqual(parsed["query"], "a parrot")
        self.assertEqual(parsed["top_k"], 100)
        self.assertEqual(parsed["date_from"], 1980)
        with self.assertRaisesRegex(ValueError, "non-empty"):
            parse_search_request({"query": "  "})
        with self.assertRaisesRegex(ValueError, "year"):
            parse_search_request({"query": "dog", "date_to": "1990"})

    def test_face_index_groups_results_by_photo(self):
        import numpy as np

        with tempfile.TemporaryDirectory() as temporary:
            archive = Path(temporary)
            search = archive / "search"
            search.mkdir()
            items = [
                {"id": "p1:0", "photo_id": "p1", "bbox": [0, 0, .2, .2], "detection_score": .9},
                {"id": "p2:0", "photo_id": "p2", "bbox": [0, 0, .2, .2], "detection_score": .9},
                {"id": "p2:1", "photo_id": "p2", "bbox": [.2, 0, .2, .2], "detection_score": .9},
                {"id": "p3:0", "photo_id": "p3", "bbox": [0, 0, .2, .2], "detection_score": .9},
            ]
            (search / "faces.json").write_text(json.dumps({"items": items}))
            np.save(
                search / "faces.npy",
                np.asarray([[1, 0], [.9, .1], [.7, .3], [0, 1]], dtype=np.float32),
            )
            photos = [{"id": "p1"}, {"id": "p2"}, {"id": "p3"}]
            index = FaceSearchIndex(archive, photos)
            results = index.search("p1:0", top_k=10)
        self.assertEqual([result["id"] for result in results], ["p2", "p3"])
        self.assertEqual(len(index.faces_for_photo("p2")), 2)

    def test_face_request_validation(self):
        self.assertEqual(parse_photo_faces_request({"photo_id": "abc"}), "abc")
        self.assertEqual(parse_face_search_request({"face_id": "abc:0", "top_k": 500}), ("abc:0", 100))
        with self.assertRaisesRegex(ValueError, "photo_id"):
            parse_photo_faces_request({"photo_id": ""})
        with self.assertRaisesRegex(ValueError, "integer"):
            parse_face_search_request({"face_id": "abc:0", "top_k": True})

    def test_similar_photo_ranking_and_validation(self):
        import numpy as np

        engine = SemanticSearchEngine.__new__(SemanticSearchEngine)
        engine.metadata = {"items": [{"id": "one"}, {"id": "two"}, {"id": "three"}]}
        engine.embeddings = np.asarray(
            [[1.0, 0.0], [0.8, 0.2], [0.0, 1.0]], dtype=np.float32
        )
        engine._photo_rows = {"one": 0, "two": 1, "three": 2}
        self.assertEqual([item["id"] for item in engine.similar("one", 2)], ["two", "three"])
        self.assertEqual(parse_similar_request({"photo_id": "one", "top_k": 500}), ("one", 100))
        with self.assertRaisesRegex(ValueError, "Photo not found"):
            engine.similar("missing")


if __name__ == "__main__":
    unittest.main()
