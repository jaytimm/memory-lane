"""Launch the gallery with a local, persistent semantic-search service."""

from __future__ import annotations

import argparse
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import sys
from threading import Timer
import webbrowser

from .search import SemanticSearchEngine


MAX_REQUEST_BYTES = 32_768


def parse_search_request(payload: object) -> dict:
    if not isinstance(payload, dict):
        raise ValueError("Request body must be a JSON object")
    query = payload.get("query")
    if not isinstance(query, str) or not query.strip():
        raise ValueError("A non-empty query is required")
    query = query.strip()
    if len(query) > 500:
        raise ValueError("Query is too long")

    top_k = payload.get("top_k", 80)
    if not isinstance(top_k, int) or isinstance(top_k, bool):
        raise ValueError("top_k must be an integer")
    top_k = min(100, max(1, top_k))

    album = payload.get("album") or None
    if album is not None and not isinstance(album, str):
        raise ValueError("album must be text")

    parsed = {"query": query, "top_k": top_k, "album": album}
    for name in ("date_from", "date_to"):
        value = payload.get(name)
        if value is not None and (not isinstance(value, int) or isinstance(value, bool)):
            raise ValueError(f"{name} must be a year")
        parsed[name] = value
    return parsed


def make_handler(archive: Path, engine: SemanticSearchEngine):
    class GalleryHandler(SimpleHTTPRequestHandler):
        def log_message(self, format: str, *args) -> None:
            # Keep the family-facing launcher quiet; explicit startup/errors remain visible.
            return

        def _json(self, status: int, payload: dict) -> None:
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:
            if self.path == "/api/status":
                self._json(
                    200,
                    {
                        "available": True,
                        "photo_count": engine.photo_count,
                        "model": engine.metadata["model"],
                    },
                )
                return
            super().do_GET()

        def do_POST(self) -> None:
            if self.path != "/api/search":
                self._json(404, {"error": "Not found"})
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if length < 1 or length > MAX_REQUEST_BYTES:
                    raise ValueError("Invalid request size")
                payload = json.loads(self.rfile.read(length).decode("utf-8"))
                request = parse_search_request(payload)
                results = engine.search(**request)
                self._json(
                    200,
                    {"query": request["query"], "count": len(results), "results": results},
                )
            except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
                self._json(400, {"error": str(error)})
            except Exception as error:
                print(f"Search error: {error}", file=sys.stderr)
                self._json(500, {"error": "Search failed"})

    return partial(GalleryHandler, directory=str(archive))


def serve(archive: Path, host: str, port: int, open_browser: bool = True) -> None:
    archive = archive.resolve()
    if not (archive / "index.html").exists():
        raise ValueError(f"Gallery not found: {archive}")
    print("Loading semantic search. This may take several seconds...", flush=True)
    engine = SemanticSearchEngine(archive)
    server = ThreadingHTTPServer((host, port), make_handler(archive, engine))
    actual_port = server.server_address[1]
    url = f"http://{host}:{actual_port}/"
    print(f"Family Photo Archive is ready: {url}", flush=True)
    print("Press Ctrl+C to stop it.", flush=True)
    if open_browser:
        Timer(0.25, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping gallery.")
    finally:
        server.server_close()


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--archive", required=True, type=Path)
    result.add_argument("--host", default="127.0.0.1")
    result.add_argument("--port", type=int, default=8765)
    result.add_argument("--no-browser", action="store_true")
    return result


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if not 0 <= args.port <= 65535:
        print("Error: --port must be between 0 and 65535", file=sys.stderr)
        return 2
    try:
        serve(args.archive, args.host, args.port, not args.no_browser)
    except (OSError, RuntimeError, ValueError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
