"""Bounded conversion of declared GOV.UK ODS/XLSX cells to published text.

This is not a spreadsheet engine: values are publisher-stored, never evaluated.
All sheets (including hidden ones), cell coordinates and notes are retained.
Graphics are excluded from text reuse, as on the HTML route; source packages
remain governed observations. No links, macros or embedded objects are run.
"""
from __future__ import annotations

import io
import json
import posixpath
import re
import zipfile
from dataclasses import dataclass
from datetime import datetime
from urllib.parse import unquote, urlsplit

from lxml import etree

from newsroom.authority.canonical import digest_canonical
from .govuk_evidence import (
    GovUkContentDocument, GovUkContentHold, MAX_BODY_BYTES, _api_url,
    _exclusion_signals, _instant, _organisation_names, _safe_attachment_location,
    _unique_object, _html_text, parse_govuk_content_document,
)

VERSION = "hermes-govuk-spreadsheet-text-v1"
MAX_EXPANDED_BYTES = 8 * 1_048_576
MAX_XML_BYTES = 4 * 1_048_576
MAX_XML_NODES = 200_000
MAX_PARTS = 256
MAX_CELLS = 100_000
MAX_SHEETS = 32
MAX_ROWS = 1_048_576
MAX_COLUMNS = 16_384
MIME_TYPES = {
    ".ods": "application/vnd.oasis.opendocument.spreadsheet",
    ".xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
}
POLICY_DIGEST = digest_canonical({
    "version": VERSION, "origin": "https://assets.publishing.service.gov.uk",
    "declaration": "exact-current-GOVUK-parent", "max_bytes": MAX_BODY_BYTES,
    "max_expanded_bytes": MAX_EXPANDED_BYTES, "max_parts": MAX_PARTS,
    "max_xml_bytes": MAX_XML_BYTES, "max_xml_nodes": MAX_XML_NODES,
    "max_cells": MAX_CELLS, "max_sheets": MAX_SHEETS,
    "max_text_bytes": MAX_BODY_BYTES, "redirects": 0,
    "formulas": "publisher-cached-values-only", "hidden_cells": "included",
    "numeric_formats": "retained-not-executed", "graphics": "excluded-from-text",
    "external_links": "never-followed", "macros": "rejected",
})
O = "{urn:oasis:names:tc:opendocument:xmlns:office:1.0}"
T = "{urn:oasis:names:tc:opendocument:xmlns:table:1.0}"
X = "{urn:oasis:names:tc:opendocument:xmlns:text:1.0}"
S = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
R = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}"
P = "{http://schemas.openxmlformats.org/package/2006/relationships}"


@dataclass(frozen=True, slots=True)
class GovUkSpreadsheetDeclaration:
    asset_url: str
    title: str
    filename: str
    content_type: str
    file_size: int
    publication: datetime
    updated: datetime
    organisations: tuple[str, ...]
    exclusion_signals: tuple[str, ...]


def is_spreadsheet_url(url: str) -> bool:
    return (type(url) is str and not url.startswith("/")
            and _safe_attachment_location(url)
            and posixpath.splitext(urlsplit(url).path)[1] in MIME_TYPES)


def declared_spreadsheet(parent_url, parent_raw, asset_url, *, retrieved_at):
    _api_url(parent_url)
    if not is_spreadsheet_url(asset_url) or not 0 < len(parent_raw) <= MAX_BODY_BYTES:
        raise ValueError("spreadsheet declaration route or size differs")
    try:
        parse_govuk_content_document(parent_url, parent_raw, retrieved_at=retrieved_at)
    except GovUkContentHold as exc:
        if (asset_url not in {url for url, _ in exc.unsupported_attachments}
                or exc.exclusion_signals):
            raise ValueError("spreadsheet parent inventory or rights differs") from None
    else:
        raise ValueError("spreadsheet parent inventory is absent")
    value = json.loads(parent_raw, object_pairs_hook=_unique_object)
    if value.get("schema_name") != "publication":
        raise ValueError("spreadsheet parent publication schema differs")
    matches = [a for a in value["details"].get("attachments", []) if a.get("url") == asset_url]
    if len(matches) != 1:
        raise ValueError("spreadsheet declaration is ambiguous")
    a = matches[0]
    suffix = posixpath.splitext(urlsplit(asset_url).path)[1]
    if (a.get("attachment_type") != "file" or a.get("content_type") != MIME_TYPES[suffix]
            or type(a.get("file_size")) is not int or not 0 < a["file_size"] <= MAX_BODY_BYTES
            or a.get("filename") != posixpath.basename(unquote(urlsplit(asset_url).path))
            or a.get("locale", "en") != "en"
            or type(a.get("id")) is not str or not a["id"].strip()):
        raise ValueError("spreadsheet declaration metadata differs")
    # Inspect the parent body and this attachment's rights statements as well as
    # ordinary page metadata. A file suffix is not an asset-specific licence.
    signals = tuple(sorted(set(
        _exclusion_signals(value, _html_text(value["details"]["body"])
                           if value["details"].get("body") else "")
        + _exclusion_signals({"details": a}, a["title"])
    )))
    if signals:
        raise ValueError("spreadsheet rights exclusion")
    return GovUkSpreadsheetDeclaration(
        asset_url, a["title"].strip(), a["filename"], a["content_type"], a["file_size"],
        _instant(value["first_published_at"]), _instant(value["public_updated_at"]),
        _organisation_names(value), signals,
    )


def _xml(archive, path):
    if archive.getinfo(path).file_size > MAX_XML_BYTES:
        raise ValueError("spreadsheet XML size exceeds bound")
    parser = etree.XMLPullParser(events=("start",), resolve_entities=False,
                                 load_dtd=False, no_network=True)
    nodes = 0
    with archive.open(path) as stream:
        while chunk := stream.read(65_536):
            parser.feed(chunk)
            nodes += sum(1 for _ in parser.read_events())
            if nodes > MAX_XML_NODES:
                raise ValueError("spreadsheet XML node count exceeds bound")
    root = parser.close()
    if root.getroottree().docinfo.doctype or any(isinstance(n, etree._Entity) for n in root.iter()):
        raise ValueError("spreadsheet XML declarations are unsupported")
    return root


def _bounded_int(text, maximum, *, minimum=1):
    if not re.fullmatch(r"[0-9]{1,8}", str(text)):
        raise ValueError("spreadsheet integer differs")
    value = int(text)
    if not minimum <= value <= maximum:
        raise ValueError("spreadsheet integer exceeds bound")
    return value


def _column(number):
    result = ""
    while number:
        number, digit = divmod(number - 1, 26)
        result = chr(65 + digit) + result
    return result


def _quoted(value):
    return json.dumps(value, ensure_ascii=False)


class _Text:
    def __init__(self):
        self.lines = []
        self.bytes = self.cells = 0

    def line(self, text):
        self.bytes += len(text.encode("utf-8")) + 1
        if self.bytes > MAX_BODY_BYTES:
            raise ValueError("spreadsheet text exceeds bound")
        self.lines.append(text)

    def row(self, number, cells):
        self.cells += len(cells)
        if self.cells > MAX_CELLS:
            raise ValueError("spreadsheet cell count exceeds bound")
        if cells:
            # Bound the joined row before allocation, including repeated values.
            if self.bytes + sum(len(v.encode()) + 20 for _, v in cells) > MAX_BODY_BYTES:
                raise ValueError("spreadsheet row text exceeds bound")
            self.line(f"Row {number}: " + "; ".join(f"{_column(col)}={val}" for col, val in cells))


def _ods_text(node):
    result = [node.text or ""]
    size = len(result[0])
    for child in node:
        if child.tag == X + "s":
            result.append(" " * _bounded_int(child.get(X + "c", "1"), MAX_BODY_BYTES))
        elif child.tag in {X + "tab", X + "line-break"}:
            result.append("\t" if child.tag == X + "tab" else "\n")
        else:
            result.append(_ods_text(child))
        result.append(child.tail or "")
        size += len(result[-1]) + len(result[-2])
        if size > MAX_BODY_BYTES:
            raise ValueError("spreadsheet cell text exceeds bound")
    text = "".join(result)
    if len(text.encode()) > MAX_BODY_BYTES:
        raise ValueError("spreadsheet cell text exceeds bound")
    return text


def _ods_cell(cell):
    paragraphs = cell.findall(X + "p")
    text = "\n".join(_ods_text(p) for p in paragraphs)
    kind = cell.get(O + "value-type")
    formula = cell.get(T + "formula")
    attr = {"float": "value", "percentage": "value", "currency": "value",
            "boolean": "boolean-value", "date": "date-value", "time": "time-value",
            "string": "string-value"}
    if kind is not None and kind not in attr:
        raise ValueError("spreadsheet cell type is unsupported")
    stored = cell.get(O + attr[kind]) if kind else None
    if formula and (not kind or stored is None and not text):
        raise ValueError("spreadsheet formula has no cached value")
    if kind not in {None, "string"} and stored is None:
        raise ValueError("spreadsheet typed cell value is absent")
    if stored is not None:
        if kind in {"float", "percentage", "currency"} and not re.fullmatch(
            r"[+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[Ee][+-]?[0-9]+)?", stored
        ):
            raise ValueError("spreadsheet numeric value differs")
        if kind == "boolean" and stored not in {"true", "false"}:
            raise ValueError("spreadsheet boolean value differs")
        if kind == "date":
            datetime.fromisoformat(stored.replace("Z", "+00:00"))
        if kind == "time" and not re.fullmatch(r"-?P(?=.+)(?:[0-9]+D)?(?:T(?=.+)(?:[0-9]+H)?(?:[0-9]+M)?(?:[0-9]+(?:\.[0-9]+)?S)?)?", stored):
            raise ValueError("spreadsheet duration value differs")
        if kind == "currency" and cell.get(O + "currency") is not None and not re.fullmatch(r"[A-Z]{3}", cell.get(O + "currency")):
            raise ValueError("spreadsheet currency code differs")
    if kind == "string" and not text:
        text = stored or ""
    result = _quoted(text) if text else ""
    if stored is not None and kind != "string":
        tags = {"float": "n", "percentage": "p", "currency": "c", "boolean": "b", "date": "d", "time": "t"}
        result = tags[kind] + ":" + stored
        if cell.get(O + "currency"):
            result += " " + cell.get(O + "currency")
        if text and text != stored:
            result += " (" + _quoted(text) + ")"
    if formula:
        if text in {"#REF!", "#VALUE!", "#DIV/0!", "#NAME?", "#N/A", "#NUM!"}:
            raise ValueError("spreadsheet formula cached an error")
        result += " [cached]"
    for axis in ("columns", "rows"):
        span = _bounded_int(cell.get(T + f"number-{axis}-spanned", "1"),
                            MAX_COLUMNS if axis == "columns" else MAX_ROWS)
        if span > 1:
            result += f" [spans {span} {axis}]"
    for child in cell:
        if child.tag in {X + "p", O + "annotation"}:
            continue
        # Images/logos are not licensed text. Other drawing content (e.g.
        # text-box-only notes or charts) requires a different extraction path.
        if etree.QName(child).localname != "frame" or any(
            etree.QName(n).localname not in {"frame", "image", "title", "desc"}
            for n in child.iter()
        ):
            raise ValueError("spreadsheet non-cell content is unsupported")
    annotations = cell.findall(O + "annotation")
    for annotation in annotations:
        result += " [note=" + _quoted("\n".join(_ods_text(p) for p in annotation.findall(X + "p"))) + "]"
    return result


def _ods(archive, output):
    if archive.read("mimetype") != MIME_TYPES[".ods"].encode():
        raise ValueError("spreadsheet ODS mimetype differs")
    root = _xml(archive, "content.xml")
    if root.tag != O + "document-content":
        raise ValueError("spreadsheet ODS root differs")
    if root.findall("./" + O + "scripts") or root.findall(".//" + T + "table-source"):
        raise ValueError("spreadsheet executable or linked table is unsupported")
    sheets = root.findall("./" + O + "body/" + O + "spreadsheet/" + T + "table")
    if not 0 < len(sheets) <= MAX_SHEETS:
        raise ValueError("spreadsheet sheet inventory differs")
    names = set()
    for sheet in sheets:
        name = sheet.get(T + "name")
        if not name or name in names:
            raise ValueError("spreadsheet sheet name differs")
        names.add(name)
        output.line("Sheet " + _quoted(name) + " (all stored rows and columns, including hidden)")
        row_number = 1
        for row in sheet.iter(T + "table-row"):
            repeat = _bounded_int(row.get(T + "number-rows-repeated", "1"), MAX_ROWS)
            if row_number + repeat - 1 > MAX_ROWS:
                raise ValueError("spreadsheet row coordinate exceeds bound")
            cells = []
            column = 1
            for cell in row:
                if cell.tag not in {T + "table-cell", T + "covered-table-cell"}:
                    raise ValueError("spreadsheet row content is unsupported")
                copies = _bounded_int(cell.get(T + "number-columns-repeated", "1"), MAX_COLUMNS)
                if column + copies - 1 > MAX_COLUMNS:
                    raise ValueError("spreadsheet column coordinate exceeds bound")
                value = _ods_cell(cell)
                if value:
                    if len(cells) + copies > MAX_CELLS:
                        raise ValueError("spreadsheet repeated cells exceed bound")
                    cells.extend((col, value) for col in range(column, column + copies))
                column += copies
            if len(cells) * repeat + output.cells > MAX_CELLS:
                raise ValueError("spreadsheet repeated rows exceed bound")
            # Empty repeated tails are positional, not millions of blank cells.
            if cells:
                for offset in range(repeat):
                    output.row(row_number + offset, cells)
            row_number += repeat


def _relationships(archive, path):
    root = _xml(archive, path)
    if root.tag != P + "Relationships":
        raise ValueError("spreadsheet relationship root differs")
    items = root.findall(P + "Relationship")
    result = {item.get("Id"): item for item in items}
    if None in result or len(result) != len(items):
        raise ValueError("spreadsheet relationship identity differs")
    return result


def _xlsx_text(node):
    # Phonetic guides are not a second copy of the displayed cell string.
    parts = node.findall(S + "t") + node.findall("./" + S + "r/" + S + "t")
    text = "".join(part.text or "" for part in parts)
    text = re.sub(r"_x([0-9A-Fa-f]{4})_", lambda m: chr(int(m[1], 16)), text)
    return text.encode("utf-16-le", "surrogatepass").decode("utf-16-le")


def _xlsx(archive, output):
    workbook = _xml(archive, "xl/workbook.xml")
    if workbook.tag != S + "workbook":
        raise ValueError("spreadsheet XLSX root differs")
    rels = _relationships(archive, "xl/_rels/workbook.xml.rels")
    strings = []
    if "xl/sharedStrings.xml" in archive.namelist():
        tree = _xml(archive, "xl/sharedStrings.xml")
        if tree.tag != S + "sst":
            raise ValueError("spreadsheet shared strings differ")
        strings = [_xlsx_text(item) for item in tree.findall(S + "si")]
        if len(strings) > MAX_CELLS:
            raise ValueError("spreadsheet shared strings exceed bound")
    styles = [0]
    formats = {}
    if "xl/styles.xml" in archive.namelist():
        tree = _xml(archive, "xl/styles.xml")
        if tree.tag != S + "styleSheet":
            raise ValueError("spreadsheet style root differs")
        declared_formats = tree.findall("./" + S + "numFmts/" + S + "numFmt")
        formats = {_bounded_int(n.get("numFmtId"), 65_535, minimum=0): n.get("formatCode")
                   for n in declared_formats}
        if len(formats) != len(declared_formats) or any(type(v) is not str or not v for v in formats.values()):
            raise ValueError("spreadsheet numeric format differs")
        styles = [_bounded_int(n.get("numFmtId", "0"), 65_535, minimum=0)
                  for n in tree.findall("./" + S + "cellXfs/" + S + "xf")]
        if not styles or any(i >= 164 and i not in formats for i in styles):
            raise ValueError("spreadsheet numeric format definition is absent")
    epoch = workbook.find(S + "workbookPr")
    date1904 = epoch is not None and epoch.get("date1904", "0") in {"1", "true"}
    output.line("XLSX stored numbers; date system=" + ("1904" if date1904 else "1900 (Excel leap-year convention)"))
    for index, format_id in enumerate(styles):
        if format_id:
            output.line(f"Number style {index}: " + _quoted(formats.get(format_id, f"ECMA-376 builtin {format_id}")))
    sheets = workbook.findall("./" + S + "sheets/" + S + "sheet")
    if not 0 < len(sheets) <= MAX_SHEETS:
        raise ValueError("spreadsheet sheet inventory differs")
    names, targets = set(), set()
    for sheet in sheets:
        name, rel = sheet.get("name"), rels.get(sheet.get(R + "id"))
        if not name or name in names or rel is None or rel.get("TargetMode", "Internal") != "Internal" or rel.get("Type") != R[1:-1] + "/worksheet":
            raise ValueError("spreadsheet sheet binding differs")
        target = rel.get("Target", "")
        target = target.lstrip("/") if target.startswith("/xl/") else posixpath.normpath("xl/" + target)
        if not target.startswith("xl/worksheets/") or target in targets:
            raise ValueError("spreadsheet worksheet target differs")
        names.add(name); targets.add(target)
        tree = _xml(archive, target)
        if tree.tag != S + "worksheet":
            raise ValueError("spreadsheet worksheet root differs")
        # Comments can qualify a value; never silently drop an unparsed note.
        if any(x.tag in {S + "legacyDrawing", S + "oleObjects", S + "controls"} for x in tree):
            raise ValueError("spreadsheet annotations or embedded controls unsupported")
        output.line("Sheet " + _quoted(name) + " (" + sheet.get("state", "visible") + "; hidden rows/columns included)")
        for merge in tree.findall("./" + S + "mergeCells/" + S + "mergeCell"):
            area = merge.get("ref", "")
            if not re.fullmatch(r"[A-Z]{1,3}[1-9][0-9]{0,6}:[A-Z]{1,3}[1-9][0-9]{0,6}", area):
                raise ValueError("spreadsheet merged range differs")
            output.line("Merged cells " + area + " (value at anchor)")
        previous_row = 0
        for row in tree.findall("./" + S + "sheetData/" + S + "row"):
            row_number = _bounded_int(row.get("r"), MAX_ROWS)
            if row_number <= previous_row:
                raise ValueError("spreadsheet row order differs")
            previous_row = row_number
            cells, previous_col = [], 0
            for cell in row:
                coordinate = cell.get("r", "")
                match = re.fullmatch(r"([A-Z]{1,3})([1-9][0-9]{0,6})", coordinate)
                if cell.tag != S + "c" or not match or int(match[2]) != row_number:
                    raise ValueError("spreadsheet cell coordinate differs")
                column = 0
                for letter in match[1]: column = column * 26 + ord(letter) - 64
                if not previous_col < column <= MAX_COLUMNS:
                    raise ValueError("spreadsheet column order differs")
                previous_col = column
                kind = cell.get("t", "n")
                value = cell.findtext(S + "v")
                formula = cell.find(S + "f")
                if formula is not None and value is None:
                    raise ValueError("spreadsheet formula has no cached value")
                if kind == "inlineStr":
                    inline = cell.find(S + "is")
                    if inline is None:
                        raise ValueError("spreadsheet inline string is absent")
                    value = _quoted(_xlsx_text(inline))
                elif kind == "s":
                    index = _bounded_int(value, len(strings) - 1, minimum=0)
                    value = _quoted(strings[index])
                elif kind in {"str", "d", "b"}:
                    if kind == "b" and value not in {"0", "1"}:
                        raise ValueError("spreadsheet boolean differs")
                    if kind == "d":
                        datetime.fromisoformat(value.replace("Z", "+00:00"))
                    value = f"{kind}:" + _quoted(value) if value is not None else None
                elif kind == "n":
                    if value is not None and not re.fullmatch(r"[+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[Ee][+-]?[0-9]+)?", value):
                        raise ValueError("spreadsheet numeric value differs")
                    if value is not None:
                        style = _bounded_int(cell.get("s", "0"), len(styles) - 1, minimum=0)
                        if styles[style]: value += f"@s{style}"
                else:
                    raise ValueError("spreadsheet error or cell type unsupported")
                if value is not None:
                    cells.append((column, value + (" [cached]" if formula is not None else "")))
            output.row(row_number, cells)


def parse_govuk_spreadsheet(parent_url, parent_raw, asset_url, raw, *, retrieved_at):
    declaration = declared_spreadsheet(parent_url, parent_raw, asset_url, retrieved_at=retrieved_at)
    if type(raw) is not bytes or len(raw) != declaration.file_size:
        raise ValueError("spreadsheet observed size differs from declaration")
    output = _Text()
    output.line("Attachment: " + asset_url)
    output.line("Published cells: Sheet + Row + column identify each cell. ODS types n=number, p=percentage, c=currency, b=boolean, d=date, t=duration; quoted parentheses retain display text. XLSX @sN refers to number style N. [cached] is a publisher-stored formula result, never recalculated. Hidden cells included; graphics excluded.")
    try:
        with zipfile.ZipFile(io.BytesIO(raw)) as archive:
            infos = archive.infolist()
            names = [i.filename for i in infos]
            if (not 0 < len(infos) <= MAX_PARTS or len(set(names)) != len(names)
                    or sum(i.file_size for i in infos) > MAX_EXPANDED_BYTES
                    or any(i.flag_bits & 1 or i.compress_type not in {0, 8}
                           or i.filename.startswith("/") or "\\" in i.filename
                           or ".." in i.filename.split("/") for i in infos)
                    or any(any(s in n.lower() for s in ("vbaproject", "activex", "embeddings/", "scripts/")) for n in names)):
                raise ValueError("spreadsheet archive exceeds supported bounds")
            (_ods if declaration.filename.endswith(".ods") else _xlsx)(archive, output)
    except (zipfile.BadZipFile, KeyError, etree.XMLSyntaxError, RuntimeError, OverflowError) as exc:
        raise ValueError("spreadsheet archive or XML differs") from exc
    if not output.cells:
        raise ValueError("spreadsheet has no published cells")
    body = "\n".join(output.lines)
    signals = _exclusion_signals({"details": {}}, body)
    if signals:
        raise ValueError("spreadsheet cell rights exclusion")
    return GovUkContentDocument("spreadsheet", declaration.title, body,
                              declaration.publication, declaration.updated,
                              declaration.organisations, signals)
