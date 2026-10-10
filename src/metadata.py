"""Read the curated album spreadsheet without requiring an Excel dependency."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from difflib import SequenceMatcher
from pathlib import Path
import re
from xml.etree import ElementTree as ET
from zipfile import ZipFile


MAIN_NS = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
NS = {"m": MAIN_NS}


@dataclass
class AlbumRecord:
    number: str
    expected_count: int | None
    years: str
    description: str

    def to_dict(self) -> dict:
        return asdict(self)


def _column_number(reference: str) -> int:
    letters = "".join(char for char in reference if char.isalpha())
    number = 0
    for char in letters.upper():
        number = number * 26 + ord(char) - 64
    return number


def _xlsx_rows(path: Path) -> list[list[str]]:
    with ZipFile(path) as workbook:
        strings: list[str] = []
        if "xl/sharedStrings.xml" in workbook.namelist():
            root = ET.fromstring(workbook.read("xl/sharedStrings.xml"))
            strings = [
                "".join(node.text or "" for node in item.findall(".//m:t", NS))
                for item in root.findall("m:si", NS)
            ]

        sheet = ET.fromstring(workbook.read("xl/worksheets/sheet1.xml"))
        rows: list[list[str]] = []
        for row_node in sheet.findall(".//m:sheetData/m:row", NS):
            row: list[str] = []
            for cell in row_node.findall("m:c", NS):
                column = _column_number(cell.attrib.get("r", ""))
                while len(row) < column:
                    row.append("")
                value_node = cell.find("m:v", NS)
                value = "" if value_node is None else (value_node.text or "")
                if cell.attrib.get("t") == "s" and value:
                    value = strings[int(value)]
                elif cell.attrib.get("t") == "inlineStr":
                    value = "".join(
                        node.text or "" for node in cell.findall(".//m:t", NS)
                    )
                row[column - 1] = " ".join(value.split())
            rows.append(row)
    return rows


def _album_number(value: str) -> str | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return str(int(number)) if number.is_integer() else str(number)


def _integer(value: str) -> int | None:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def read_album_records(path: str | Path) -> list[AlbumRecord]:
    """Parse album records, joining wrapped spreadsheet rows into descriptions."""
    records: list[AlbumRecord] = []
    current: AlbumRecord | None = None

    for cells in _xlsx_rows(Path(path)):
        cells = cells + [""] * (4 - len(cells))
        number = _album_number(cells[0])
        if number is not None:
            current = AlbumRecord(
                number=number,
                expected_count=_integer(cells[1]),
                years=cells[2].strip(),
                description=cells[3].strip(),
            )
            records.append(current)
            continue

        if cells[1].strip().lower() in {"no. of pics", "size", "no. of books"}:
            current = None
            continue

        if current is not None:
            continuation = " ".join(
                value.strip() for value in (cells[2], cells[3]) if value.strip()
            )
            if continuation:
                current.description = " ".join(
                    part for part in (current.description, continuation) if part
                )

    return records


def normalized_label(value: str) -> str:
    value = value.casefold().replace("googlephotos", "")
    return re.sub(r"[^a-z0-9]+", "", value)


def match_records_to_folders(
    records: list[AlbumRecord], folders: list[str]
) -> dict[str, AlbumRecord]:
    """Match curated year labels to folder names using a one-to-one fuzzy match."""
    candidates: list[tuple[float, str, int]] = []
    for folder in folders:
        folder_key = normalized_label(folder)
        for index, record in enumerate(records):
            record_key = normalized_label(record.years)
            if not folder_key or not record_key:
                continue
            score = SequenceMatcher(None, folder_key, record_key).ratio()
            # A shared four-digit starting year is strong evidence.
            folder_year = re.search(r"(?:19|20)\d{2}", folder)
            record_year = re.search(r"(?:19|20)\d{2}", record.years)
            if folder_year and record_year and folder_year.group() == record_year.group():
                score += 0.3
            candidates.append((score, folder, index))

    matches: dict[str, AlbumRecord] = {}
    used_records: set[int] = set()
    for score, folder, index in sorted(candidates, reverse=True):
        if score < 0.65 or folder in matches or index in used_records:
            continue
        matches[folder] = records[index]
        used_records.add(index)
    return matches

