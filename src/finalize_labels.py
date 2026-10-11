"""Normalize aliases and write a canonical face-label export."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


def normalize_aliases(labels: dict, aliases: dict[str, str]) -> dict:
    """Merge person-name aliases throughout an exported label document."""
    result = json.loads(json.dumps(labels))
    for cluster in result.get("clusters", []):
        name = cluster.get("name")
        if name in aliases:
            cluster["name"] = aliases[name]

    recovery = result.setdefault("recovery", {})
    for section in ("accepted", "dismissed"):
        values = recovery.setdefault(section, {})
        merged: dict[str, list] = {}
        for name, entries in values.items():
            merged.setdefault(aliases.get(name, name), []).extend(entries)
        recovery[section] = merged

    people: dict[str, dict[str, list[str]]] = {}
    for cluster in result.get("clusters", []):
        name = cluster.get("name")
        if not name:
            continue
        person = people.setdefault(name, {"cluster_ids": [], "face_ids": []})
        person["cluster_ids"].append(cluster["cluster_id"])
        person["face_ids"].extend(cluster.get("face_ids", []))
    for name, batches in recovery.get("accepted", {}).items():
        person = people.setdefault(name, {"cluster_ids": [], "face_ids": []})
        for batch in batches:
            person["face_ids"].extend(batch)
    for person in people.values():
        person["cluster_ids"] = list(dict.fromkeys(person["cluster_ids"]))
        person["face_ids"] = list(dict.fromkeys(person["face_ids"]))
    result["people"] = people
    result["normalization"] = {"aliases": aliases}
    return result


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--labels", required=True, type=Path)
    result.add_argument("--output", required=True, type=Path)
    result.add_argument(
        "--alias",
        action="append",
        default=[],
        metavar="OLD=NEW",
        help="merge OLD into NEW; may be repeated",
    )
    return result


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        aliases = dict(value.split("=", 1) for value in args.alias)
        if any(not old or not new for old, new in aliases.items()):
            raise ValueError("Aliases require non-empty OLD=NEW names")
        labels = json.loads(args.labels.read_text(encoding="utf-8"))
        normalized = normalize_aliases(labels, aliases)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(normalized, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    except (OSError, ValueError, json.JSONDecodeError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 2
    print(f"Wrote {args.output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
