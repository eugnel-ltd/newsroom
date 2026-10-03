"""Complete declared GOV.UK PDF text; unsupported content retains a typed HOLD.

Only published text and publisher-marked decorative Artifact graphics are in
scope. The latter require a tagged structure covering every text span. No OCR,
active content, font guessing, hidden truncation or source fetch occurs here.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import io
import json
import logging
import posixpath
import resource
import subprocess
import sys
import tempfile
import time
from urllib.parse import unquote, urlsplit

from newsroom.authority.canonical import digest_bytes, digest_canonical
from .govuk_evidence import (GovUkContentDocument, GovUkContentHold, MAX_BODY_BYTES,
    _api_url, _exclusion_signals, _html_text, _instant, _organisation_names,
    _safe_attachment_location, _unique_object, parse_govuk_content_document)

VERSION = 'hermes-govuk-pdf-text-v1'
MAX_RAW_BYTES = 8 * 1_048_576
MAX_TEXT_BYTES = MAX_BODY_BYTES
MAX_PAGES = 256
MAX_OPERATORS = 500_000
MAX_OBJECTS = 100_000
MAX_DEPTH = 64
CPU_SECONDS = 10
TIMEOUT_SECONDS = 15
MEMORY_BYTES = 256 * 1_048_576
MAX_WORKER_OUTPUT = MAX_TEXT_BYTES
POLICY_DIGEST = digest_canonical({'version': VERSION, 'parser': 'pypdf',
    'raw_bytes': MAX_RAW_BYTES, 'text_bytes': MAX_TEXT_BYTES, 'pages': MAX_PAGES,
    'operators': MAX_OPERATORS, 'objects': MAX_OBJECTS, 'depth': MAX_DEPTH,
    'cpu_seconds': CPU_SECONDS, 'wall_seconds': TIMEOUT_SECONDS, 'memory_bytes': MEMORY_BYTES,
    'worker_output_bytes': MAX_WORKER_OUTPUT, 'memory_enforcement': 'own-child-RSS-supervision',
    'graphics': 'tagged-publisher-Artifact-only-with-complete-structured-text',
    'forms': 'explicit-empty-Fields-with-declarative-DA-DR-only',
    'active_content': 'rejected', 'unmapped_glyphs': 'rejected', 'ocr': False})


class GovUkPdfHold(GovUkContentHold):
    pass


def _hold(code):
    raise GovUkPdfHold('SOURCE_PDF_' + code + '_HOLD')


@dataclass(frozen=True, slots=True)
class GovUkPdfDeclaration:
    asset_url: str
    title: str
    filename: str
    file_size: int
    page_count: int | None
    publication: datetime
    updated: datetime
    organisations: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class GovUkPdfDocument(GovUkContentDocument):
    raw_digest: str
    page_inventory: tuple[dict, ...]
    parser_version: str


def is_pdf_url(url):
    return (isinstance(url, str) and not url.startswith('/') and _safe_attachment_location(url)
            and posixpath.splitext(urlsplit(url).path)[1].lower() == '.pdf')


def declared_pdf(parent_url, parent_raw, asset_url, *, retrieved_at):
    _api_url(parent_url)
    if not is_pdf_url(asset_url) or not 0 < len(parent_raw) <= MAX_BODY_BYTES:
        _hold('DECLARATION')
    try:
        parse_govuk_content_document(parent_url, parent_raw, retrieved_at=retrieved_at)
    except GovUkContentHold as exc:
        if asset_url not in dict(exc.unsupported_attachments) or exc.exclusion_signals:
            _hold('DECLARATION_RIGHTS')
    else:
        _hold('DECLARATION')
    value = json.loads(parent_raw, object_pairs_hook=_unique_object)
    attachments = [entry for entry in value.get('details', {}).get('attachments', []) if entry.get('url') == asset_url]
    if value.get('schema_name') != 'publication' or len(attachments) != 1:
        _hold('DECLARATION')
    asset = attachments[0]
    if (asset.get('attachment_type') != 'file' or asset.get('content_type') != 'application/pdf'
        or type(asset.get('file_size')) is not int or not 0 < asset['file_size'] <= MAX_RAW_BYTES
        or asset.get('number_of_pages') is not None
        and (type(asset['number_of_pages']) is not int or not 0 < asset['number_of_pages'] <= MAX_PAGES)
        or asset.get('filename') != posixpath.basename(unquote(urlsplit(asset_url).path))
        or asset.get('locale', 'en') != 'en' or not isinstance(asset.get('id'), str) or not asset['id'].strip()):
        _hold('DECLARATION')
    body = value['details'].get('body')
    if (_exclusion_signals(value, _html_text(body) if body else '')
            or _exclusion_signals({'details': asset}, asset['title'])):
        _hold('RIGHTS_EXCLUSION')
    return GovUkPdfDeclaration(asset_url, asset['title'].strip(), asset['filename'], asset['file_size'],
        asset.get('number_of_pages'), _instant(value['first_published_at']), _instant(value['public_updated_at']),
        _organisation_names(value))


class _ParserWarnings(logging.Handler):
    def emit(self, record):
        _hold('PARSER_WARNING')


def _extract(raw):
    import pypdf
    from pypdf._font import Font
    from pypdf.generic import ArrayObject, ContentStream, DictionaryObject, IndirectObject, StreamObject
    pypdf.overwrite_configuration(disable_legacy_handling=True,
        maximum_declared_stream_length=MAX_RAW_BYTES,
        array_based_stream_maximum_output_length=MAX_RAW_BYTES,
        zlib_maximum_output_length=MAX_RAW_BYTES, lzw_maximum_output_length=MAX_RAW_BYTES,
        run_length_maximum_output_length=MAX_RAW_BYTES, image_maximum_buffer_size=MAX_RAW_BYTES,
        page_tree_maximum_entries=MAX_OBJECTS, page_tree_maximum_depth=MAX_DEPTH,
        jbig2dec_binary=None)
    reader = pypdf.PdfReader(io.BytesIO(raw), strict=True)
    if reader.is_encrypted:
        _hold('ENCRYPTED')
    visited, mcids = set(), set()
    count = 0

    def inspect(value, depth=0):
        nonlocal count
        if depth > MAX_DEPTH:
            _hold('STRUCTURE_BOUND')
        if isinstance(value, IndirectObject):
            key = (value.idnum, value.generation)
            if key in visited:
                return
            visited.add(key)
            value = value.get_object()
        count += 1
        if count > MAX_OBJECTS:
            _hold('STRUCTURE_BOUND')
        if isinstance(value, DictionaryObject):
            if any(key in value for key in ('/JS', '/JavaScript', '/EmbeddedFiles', '/EF', '/XFA', '/RichMediaContent')):
                _hold('ACTIVE_CONTENT')
            if value.get('/S') in ('/JavaScript', '/Launch', '/SubmitForm', '/ImportData', '/GoToR', '/GoToE', '/Rendition', '/Sound', '/Movie'):
                _hold('ACTIVE_CONTENT')
            if value.get('/Type') == '/Action' and value.get('/S') not in ('/GoTo', '/URI'):
                _hold('ACTIVE_CONTENT')
            if value.get('/S') == '/Figure' or '/ActualText' in value:
                _hold('UNSUPPORTED_TEXT_GRAPHIC')
            if '/AcroForm' in value and value['/AcroForm'].get_object():
                form = value['/AcroForm'].get_object()
                if '/XFA' in form:
                    _hold('ACTIVE_CONTENT')
                # An explicit empty publisher shell has no interactive text.
                # DA/DR resources still pass through the recursive inspection.
                if (not isinstance(form, DictionaryObject)
                        or set(form) - {'/Fields', '/DA', '/DR'}
                        or '/Fields' not in form
                        or not isinstance(form['/Fields'], ArrayObject) or form['/Fields']
                        or '/DA' in form and not isinstance(form['/DA'], str)
                        or '/DR' in form and not isinstance(form['/DR'], DictionaryObject)):
                    _hold('FORM_CONTENT')
            if '/Annots' in value and any(annotation.get_object().get('/Subtype') != '/Link'
                    for annotation in value['/Annots']):
                _hold('ANNOTATION_TEXT')
            if value.get('/Type') == '/MCR' and isinstance(value.get('/MCID'), int) and '/Pg' in value:
                mcids.add((value.raw_get('/Pg').idnum, int(value['/MCID'])))
            if value.get('/Type') == '/StructElem' and '/Pg' in value:
                kids = value.get('/K', [])
                kids = kids if isinstance(kids, ArrayObject) else [kids]
                for kid in kids:
                    if isinstance(kid, int):
                        mcids.add((value.raw_get('/Pg').idnum, int(kid)))
            for item in value.values():
                inspect(item, depth + 1)
        elif isinstance(value, ArrayObject):
            for item in value:
                inspect(item, depth + 1)
    inspect(reader.trailer['/Root'])
    root = reader.trailer['/Root']
    tagged = bool(root.get('/MarkInfo', {}).get('/Marked')) and '/StructTreeRoot' in root
    pages = list(reader.pages)
    if not 0 < len(pages) <= MAX_PAGES:
        _hold('PAGE_BOUND')
    result, text_bytes, operators = [], 0, 0
    for number, page in enumerate(pages, 1):
        resources = page.get('/Resources', {}).get_object()
        fonts = resources.get('/Font', {}).get_object()
        font, stack, marks, glyph_text, glyphs, images = None, [], [], [], 0, 0
        contents = page.get_contents()
        if contents is None:
            result.append({'page': number, 'text': '', 'glyphs': 0, 'decorative_images': 0})
            continue
        if len(contents.get_data()) > MAX_RAW_BYTES:
            _hold('STREAM_BOUND')
        stream = ContentStream(contents, reader, forced_encoding='bytes')
        for args, operation in stream.operations:
            operators += 1
            if operators > MAX_OPERATORS:
                _hold('OPERATOR_BOUND')
            if operation == b'Tf':
                definition = fonts.get(args[0])
                if definition is None:
                    _hold('FONT_MAPPING')
                definition = definition.get_object()
                if definition.get('/Subtype') == '/Type3':
                    _hold('FONT_MAPPING')
                if '/ToUnicode' not in definition and not (definition.get('/Subtype') == '/Type1'
                        and definition.get('/BaseFont') in ('/Helvetica', '/Helvetica-Bold', '/Helvetica-Oblique', '/Helvetica-BoldOblique',
                            '/Times-Roman', '/Times-Bold', '/Times-Italic', '/Times-BoldItalic', '/Courier', '/Courier-Bold', '/Courier-Oblique', '/Courier-BoldOblique')
                        and definition.get('/Encoding', '/StandardEncoding') in ('/StandardEncoding', '/WinAnsiEncoding', '/MacRomanEncoding')):
                    _hold('FONT_MAPPING')
                font = Font.from_font_resource(definition)
                if not font.interpretable:
                    _hold('FONT_MAPPING')
            elif operation == b'q':
                stack.append(font)
            elif operation == b'Q':
                if not stack:
                    _hold('CONTENT_STRUCTURE')
                font = stack.pop()
            elif operation in (b'BMC', b'BDC'):
                properties = args[1] if operation == b'BDC' else {}
                if not isinstance(properties, dict):
                    properties = resources.get('/Properties', {}).get(properties, {}).get_object()
                marks.append((args[0] == '/Artifact', properties.get('/MCID')))
            elif operation == b'EMC':
                if not marks:
                    _hold('CONTENT_STRUCTURE')
                marks.pop()
            elif operation in (b'Tj', b'TJ', b"'", b'"'):
                if font is None:
                    _hold('FONT_MAPPING')
                values = args[0] if operation == b'TJ' else [args[-1]]
                for value in values:
                    if not isinstance(value, bytes):
                        if isinstance(value, (int, float)):
                            continue
                        _hold('FONT_MAPPING')
                    if isinstance(font.encoding, dict):
                        if any(code not in font.encoding for code in value):
                            _hold('FONT_MAPPING')
                        decoded = ''.join(font.encoding[code] for code in value)
                    elif font.encoding in ('utf-16-be', 'utf-16-le', 'charmap'):
                        decoded = value.decode(font.encoding, errors='strict')
                    else:
                        _hold('FONT_MAPPING')
                    if font.character_map and any(character not in font.character_map for character in decoded):
                        _hold('FONT_MAPPING')
                    decoded = ''.join(font.character_map.get(character, character) for character in decoded)
                    if any(character == '\ufffd' or 0xd800 <= ord(character) <= 0xdfff
                           or 127 <= ord(character) <= 159
                           or ord(character) < 32 and character not in '\t\r\n' for character in decoded):
                        _hold('FONT_MAPPING')
                    glyph_text.append(decoded)
                    glyphs += len(value) if isinstance(font.encoding, dict) or font.encoding == 'charmap' else len(value) // 2
                    if tagged and not any((page.indirect_reference.idnum, mcid) in mcids for artifact, mcid in marks if not artifact):
                        _hold('STRUCTURED_TEXT_COVERAGE')
            elif operation == b'Do':
                value = resources.get('/XObject', {}).get(args[0])
                if value is None or value.get_object().get('/Subtype') != '/Image':
                    _hold('UNSUPPORTED_CONTENT')
                if not tagged or not any(artifact for artifact, _ in marks):
                    _hold('UNSUPPORTED_TEXT_GRAPHIC')
                images += 1
            elif operation in (b'INLINE IMAGE', b'sh'):
                _hold('UNSUPPORTED_TEXT_GRAPHIC')
            elif operation in (b'S', b's', b'f', b'F', b'f*', b'B', b'B*', b'b', b'b*'):
                if not tagged or not any(artifact for artifact, _ in marks):
                    _hold('UNSUPPORTED_TEXT_GRAPHIC')
            elif operation not in {b'BT', b'ET', b'Td', b'TD', b'Tm', b'T*', b'Tc', b'Tw', b'Tz', b'TL', b'Ts', b'Tr',
                    b'cm', b'w', b'J', b'j', b'M', b'd', b'ri', b'i', b'gs', b'G', b'g', b'RG', b'rg', b'K', b'k',
                    b'CS', b'cs', b'SC', b'sc', b'SCN', b'scn', b'm', b'l', b'c', b'v', b'y', b'h', b're', b'n', b'W', b'W*'}:
                _hold('UNSUPPORTED_CONTENT')
        if stack or marks:
            _hold('CONTENT_STRUCTURE')
        if images and not glyphs:
            _hold('SCANNED_CONTENT')
        text = page.extract_text(orientations=(0, 90, 180, 270))
        if ''.join(text.split()) != ''.join(''.join(glyph_text).split()):
            _hold('TEXT_COVERAGE')
        text_bytes += len(text.encode('utf-8'))
        if text_bytes > MAX_TEXT_BYTES:
            _hold('TEXT_BOUND')
        result.append({'page': number, 'text': text, 'glyphs': glyphs, 'decorative_images': images})
    if not any(page['glyphs'] for page in result):
        _hold('SCANNED_CONTENT')
    return {'pages': result, 'metadata': str(reader.metadata or {}), 'parser_version': pypdf.__version__}


def _worker():
    resource.setrlimit(resource.RLIMIT_CPU, (CPU_SECONDS, CPU_SECONDS + 1))
    resource.setrlimit(resource.RLIMIT_FSIZE, (MAX_WORKER_OUTPUT, MAX_WORKER_OUTPUT))
    logger = logging.getLogger('pypdf')
    logger.handlers = [_ParserWarnings()]; logger.propagate = False
    try:
        raw = sys.stdin.buffer.read(MAX_RAW_BYTES + 1)
        if len(raw) > MAX_RAW_BYTES:
            _hold('RAW_BOUND')
        value = _extract(raw)
    except GovUkPdfHold as exc:
        value = {'hold': exc.reason_code}
    except Exception:
        value = {'hold': 'SOURCE_PDF_STRUCTURE_HOLD'}
    encoded = json.dumps(value, ensure_ascii=False).encode()
    if len(encoded) > MAX_WORKER_OUTPUT:
        encoded = b'{"hold":"SOURCE_PDF_OUTPUT_BOUND_HOLD"}'
    sys.stdout.buffer.write(encoded)


def parse_govuk_pdf(parent_url, parent_raw, asset_url, raw, *, retrieved_at):
    declaration = declared_pdf(parent_url, parent_raw, asset_url, retrieved_at=retrieved_at)
    if type(raw) is not bytes or len(raw) != declaration.file_size or not raw.startswith(b'%PDF-') or not raw.rstrip().endswith(b'%%EOF'):
        _hold('RAW_IDENTITY')
    try:
        # Anonymous bounded input avoids Python 3.12's timed pipe-resumption
        # trap without a writer thread, version branch or retained raw archive.
        with tempfile.TemporaryFile() as source:
            source.write(raw)
            source.seek(0)
            process = subprocess.Popen([sys.executable, '-I', '-c',
                'from newsroom.control_plane.govuk_pdf import _worker; _worker()'],
                stdin=source, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            deadline = time.monotonic() + TIMEOUT_SECONDS
            try:
                while True:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        _hold('WORKER_BOUND')
                    try:
                        stdout, stderr = process.communicate(timeout=min(0.05, remaining))
                        break
                    except subprocess.TimeoutExpired:
                        if process.poll() is not None:
                            continue
                        # Darwin rejects address-space/data rlimits. Observe only
                        # our own child; a live child needs numeric memory evidence.
                        try:
                            memory = subprocess.run(['/bin/ps', '-o', 'rss=', '-p', str(process.pid)],
                                capture_output=True, timeout=min(1, remaining), check=True).stdout.strip()
                        except (subprocess.SubprocessError, OSError):
                            if process.poll() is not None:
                                continue
                            raise
                        if not memory.isdigit():
                            # Our child may exit between poll() and ps. Completion
                            # still passes through the exact output checks below.
                            if process.poll() is not None:
                                continue
                            _hold('MEMORY_BOUND')
                        if int(memory) * 1024 > MEMORY_BYTES:
                            _hold('MEMORY_BOUND')
            finally:
                if process.poll() is None:
                    process.kill()
                process.communicate()
            if process.returncode or len(stdout) > MAX_WORKER_OUTPUT or stderr:
                _hold('WORKER_OUTPUT')
            value = json.loads(stdout)
    except GovUkPdfHold:
        raise
    except (subprocess.SubprocessError, ValueError, OSError):
        _hold('WORKER_BOUND')
    if not isinstance(value, dict):
        _hold('WORKER_OUTPUT')
    if value.get('hold'):
        if not isinstance(value['hold'], str) or not value['hold'].startswith('SOURCE_PDF_'):
            _hold('WORKER_OUTPUT')
        raise GovUkPdfHold(value['hold'])
    pages = value.get('pages')
    if (not isinstance(pages, list) or not 0 < len(pages) <= MAX_PAGES
            or not isinstance(value.get('metadata'), str) or not isinstance(value.get('parser_version'), str)
            or any(not isinstance(page, dict) or page.get('page') != index
                   or not isinstance(page.get('text'), str)
                   or type(page.get('glyphs')) is not int or page['glyphs'] < 0
                   or type(page.get('decorative_images')) is not int or page['decorative_images'] < 0
                   for index, page in enumerate(pages, 1))):
        _hold('WORKER_OUTPUT')
    if declaration.page_count is not None and len(pages) != declaration.page_count:
        _hold('PAGE_COVERAGE')
    body = 'Attachment: ' + asset_url + '\n' + '\n\n'.join(f"Page {page['page']}\n{page['text']}" for page in pages)
    if len((declaration.title + '\n\n' + body).encode()) > MAX_TEXT_BYTES:
        _hold('TEXT_BOUND')
    signals = _exclusion_signals({'details': {'body': value['metadata']}}, body)
    if signals:
        _hold('RIGHTS_EXCLUSION')
    inventory = tuple({'page': page['page'], 'glyphs': page['glyphs'], 'text_digest': digest_bytes(page['text'].encode()),
        'decorative_images': page['decorative_images']} for page in pages)
    return GovUkPdfDocument('pdf', declaration.title, body, declaration.publication, declaration.updated,
        declaration.organisations, (), digest_bytes(raw), inventory, value['parser_version'])
