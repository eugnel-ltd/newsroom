"""Complete born-digital text fixtures; no source fetch or provider effects."""
from datetime import UTC, datetime
import json
import pytest

NOW = datetime(2026, 10, 3, tzinfo=UTC)
PARENT = 'https://www.gov.uk/government/publications/pdf-guidance'
ASSET = 'https://assets.publishing.service.gov.uk/media/fixture/guidance.pdf'


def pdf_bytes(*pages, padding=0, font=b'<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica /Encoding /WinAnsiEncoding >>', catalog=b''):
    """A small independent PDF fixture, not an extraction implementation."""
    objects = [b'<< /Type /Catalog /Pages 2 0 R ' + catalog + b' >>', b'', font]
    kids = []
    for text in pages:
        page_id, stream_id = len(objects) + 1, len(objects) + 2
        kids.append(f'{page_id} 0 R')
        escaped = text.encode('cp1252').replace(b'\\', b'\\\\').replace(b'(', b'\\(').replace(b')', b'\\)')
        stream = b'BT /F1 12 Tf 72 720 Td (' + escaped + b') Tj ET'
        objects.extend([f'<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Resources << /Font << /F1 3 0 R >> >> /Contents {stream_id} 0 R >>'.encode(),
                        b'<< /Length ' + str(len(stream)).encode() + b' >>\nstream\n' + stream + b'\nendstream'])
    objects[1] = f'<< /Type /Pages /Kids [{" ".join(kids)}] /Count {len(kids)} >>'.encode()
    raw, offsets = bytearray(b'%PDF-1.7\n'), [0]
    for number, value in enumerate(objects, 1):
        offsets.append(len(raw)); raw.extend(f'{number} 0 obj\n'.encode() + value + b'\nendobj\n')
    if padding:
        raw.extend(b"%" + b"x" * padding + b"\n")
    start = len(raw)
    raw.extend(f'xref\n0 {len(offsets)}\n0000000000 65535 f \n'.encode())
    raw.extend(b''.join(f'{offset:010} 00000 n \n'.encode() for offset in offsets[1:]))
    raw.extend(f'trailer\n<< /Size {len(offsets)} /Root 1 0 R >>\nstartxref\n{start}\n%%EOF\n'.encode())
    return bytes(raw)


def parent_bytes(raw, pages=2):
    return json.dumps({'base_path': '/government/publications/pdf-guidance', 'locale': 'en',
        'schema_name': 'publication', 'document_type': 'guidance', 'title': 'Complete PDF guidance',
        'first_published_at': '2026-10-01T09:00:00Z', 'public_updated_at': '2026-10-02T09:00:00Z',
        'details': {'body': '<p>Publisher guidance.</p>', 'attachments': [{
            'id': 'pdf-guidance', 'title': 'Complete PDF guidance', 'attachment_type': 'file',
            'url': ASSET, 'filename': 'guidance.pdf', 'content_type': 'application/pdf',
            'file_size': len(raw), 'number_of_pages': pages, 'locale': 'en'}]},
        'links': {'organisations': [{'title': 'Home Office'}]}}).encode()


def test_complete_pdf_retains_every_page_text_and_exact_raw_inventory():
    from newsroom.control_plane.govuk_pdf import parse_govuk_pdf
    from newsroom.authority.canonical import digest_bytes
    raw = pdf_bytes('First page: 12 applicants may apply.', 'Second page: fees are £45. Footnote 1 applies.')
    document = parse_govuk_pdf(PARENT, parent_bytes(raw), ASSET, raw, retrieved_at=NOW)
    assert document.document_type == 'pdf'
    assert document.raw_digest == digest_bytes(raw)
    assert len(document.page_inventory) == 2
    assert [page['page'] for page in document.page_inventory] == [1, 2]
    assert 'First page: 12 applicants may apply.' in document.body_text
    assert 'Second page: fees are £45. Footnote 1 applies.' in document.body_text
    assert sum(page['glyphs'] for page in document.page_inventory) == 82


@pytest.mark.parametrize('change,reason', [
    ('missing-font-map', 'SOURCE_PDF_FONT_MAPPING_HOLD'),
    ('type3', 'SOURCE_PDF_FONT_MAPPING_HOLD'),
    ('empty', 'SOURCE_PDF_SCANNED_CONTENT_HOLD'),
    ('wrong-pages', 'SOURCE_PDF_PAGE_COVERAGE_HOLD'),
    ('js', 'SOURCE_PDF_ACTIVE_CONTENT_HOLD'),
    ('launch', 'SOURCE_PDF_ACTIVE_CONTENT_HOLD'),
    ('embedded', 'SOURCE_PDF_ACTIVE_CONTENT_HOLD'),
    ('xfa', 'SOURCE_PDF_ACTIVE_CONTENT_HOLD'),
])
def test_incomplete_or_active_pdf_is_a_typed_hold_not_partial_text(change, reason):
    from newsroom.control_plane.govuk_pdf import parse_govuk_pdf, GovUkPdfHold
    options, pages = {}, 2
    if change == 'missing-font-map':
        options['font'] = b'<< /Type /Font /Subtype /TrueType /BaseFont /Unknown >>'
    elif change == 'type3':
        options['font'] = b'<< /Type /Font /Subtype /Type3 /BaseFont /Unknown >>'
    elif change == 'js':
        options['catalog'] = b'/OpenAction << /S /JavaScript /JS (app.alert\\(1\\)) >>'
    elif change == 'launch':
        options['catalog'] = b'/OpenAction << /Type /Action /S /Launch /F (program) >>'
    elif change == 'embedded':
        options['catalog'] = b'/Names << /EmbeddedFiles << /Names [] >> >>'
    elif change == 'xfa':
        options['catalog'] = b'/AcroForm << /XFA (dynamic form) >>'
    elif change == 'wrong-pages':
        pages = 1
    raw = pdf_bytes(*(['', ''] if change == 'empty' else ['Complete first page.', 'Complete second page.']), **options)
    with pytest.raises(GovUkPdfHold, match=reason):
        parse_govuk_pdf(PARENT, parent_bytes(raw, pages), ASSET, raw, retrieved_at=NOW)


def tagged_image_pdf(*, artifact=True, covered=True, image_only=False):
    """Add one explicit publisher Artifact and a real MCID structure to the tiny PDF."""
    import io
    from pypdf import PdfReader, PdfWriter
    from pypdf.generic import (ArrayObject, BooleanObject, DecodedStreamObject, DictionaryObject,
                              NameObject, NumberObject)
    writer = PdfWriter()
    writer.append_pages_from_reader(PdfReader(io.BytesIO(pdf_bytes('Published text: 12 applicants may apply.'))))
    page = writer.pages[0]
    image = DecodedStreamObject()
    image.update({NameObject('/Type'): NameObject('/XObject'), NameObject('/Subtype'): NameObject('/Image'),
                  NameObject('/Width'): NumberObject(1), NameObject('/Height'): NumberObject(1),
                  NameObject('/ColorSpace'): NameObject('/DeviceGray'), NameObject('/BitsPerComponent'): NumberObject(8)})
    image.set_data(b'\xff')
    page['/Resources'][NameObject('/XObject')] = DictionaryObject({NameObject('/Logo'): writer._add_object(image)})
    content = DecodedStreamObject()
    graphic = b'/Artifact BMC /Logo Do EMC\n' if artifact else b'/Logo Do\n'
    text = b'' if image_only else b'/P << /MCID 0 >> BDC\n' + page.get_contents().get_data() + b'\nEMC'
    content.set_data(graphic + text)
    page[NameObject('/Contents')] = writer._add_object(content)
    structure = DictionaryObject({NameObject('/Type'): NameObject('/StructTreeRoot')})
    element = DictionaryObject({NameObject('/Type'): NameObject('/StructElem'), NameObject('/S'): NameObject('/P'),
                                NameObject('/Pg'): page.indirect_reference, NameObject('/K'): NumberObject(0 if covered else 1)})
    structure[NameObject('/K')] = ArrayObject([writer._add_object(element)])
    writer.root_object[NameObject('/StructTreeRoot')] = writer._add_object(structure)
    writer.root_object[NameObject('/MarkInfo')] = DictionaryObject({NameObject('/Marked'): BooleanObject(True)})
    output = io.BytesIO(); writer.write(output)
    return output.getvalue()


def test_publisher_tagged_artifact_logo_with_complete_text_structure_is_allowed():
    from newsroom.control_plane.govuk_pdf import parse_govuk_pdf
    raw = tagged_image_pdf()
    document = parse_govuk_pdf(PARENT, parent_bytes(raw, 1), ASSET, raw, retrieved_at=NOW)
    assert 'Published text: 12 applicants may apply.' in document.body_text
    assert document.page_inventory[0]['decorative_images'] == 1
    assert document.page_inventory[0]['glyphs'] == 40


@pytest.mark.parametrize('options,reason', [
    ({'artifact': False}, 'SOURCE_PDF_UNSUPPORTED_TEXT_GRAPHIC_HOLD'),
    ({'covered': False}, 'SOURCE_PDF_STRUCTURED_TEXT_COVERAGE_HOLD'),
    ({'image_only': True}, 'SOURCE_PDF_SCANNED_CONTENT_HOLD'),
])
def test_text_bearing_or_uncovered_graphics_never_receive_empty_success(options, reason):
    from newsroom.control_plane.govuk_pdf import parse_govuk_pdf, GovUkPdfHold
    raw = tagged_image_pdf(**options)
    with pytest.raises(GovUkPdfHold, match=reason):
        parse_govuk_pdf(PARENT, parent_bytes(raw, 1), ASSET, raw, retrieved_at=NOW)


def test_internal_view_destination_is_not_an_executable_action():
    from newsroom.control_plane.govuk_pdf import parse_govuk_pdf
    raw = pdf_bytes('First page text.', 'Second page text.', catalog=b'/OpenAction [4 0 R /Fit]')
    assert len(parse_govuk_pdf(PARENT, parent_bytes(raw), ASSET, raw, retrieved_at=NOW).page_inventory) == 2


def unicode_font_pdf(*, omitted=None):
    import io
    from pypdf import PdfReader, PdfWriter
    from pypdf.generic import DecodedStreamObject, NameObject
    writer = PdfWriter()
    writer.append_pages_from_reader(PdfReader(io.BytesIO(pdf_bytes('Fees are £45; applicants may apply.'))))
    font = writer.pages[0]['/Resources']['/Font']['/F1']
    font[NameObject('/Subtype')] = NameObject('/TrueType')
    font[NameObject('/BaseFont')] = NameObject('/ABCDEF+Arial')
    codes = [code for code in list(range(32, 127)) + [163] if code != omitted]
    mapping = DecodedStreamObject()
    mapping.set_data(b'/CIDInit /ProcSet findresource begin 12 dict begin begincmap\n'
        b'1 begincodespacerange <00> <FF> endcodespacerange\n' + str(len(codes)).encode() + b' beginbfchar\n' +
        b'\n'.join(f'<{code:02x}> <{code:04x}>'.encode() for code in codes) + b'\nendbfchar endcmap end end')
    font[NameObject('/ToUnicode')] = writer._add_object(mapping)
    result = io.BytesIO(); writer.write(result)
    return result.getvalue()


def test_publisher_unicode_mapping_covers_every_used_glyph_in_a_subset_font():
    from newsroom.control_plane.govuk_pdf import parse_govuk_pdf
    raw = unicode_font_pdf()
    document = parse_govuk_pdf(PARENT, parent_bytes(raw, 1), ASSET, raw, retrieved_at=NOW)
    assert 'Fees are £45; applicants may apply.' in document.body_text


def test_partial_unicode_mapping_does_not_guess_a_missing_pound_sign():
    from newsroom.control_plane.govuk_pdf import parse_govuk_pdf, GovUkPdfHold
    raw = unicode_font_pdf(omitted=163)
    with pytest.raises(GovUkPdfHold, match='SOURCE_PDF_FONT_MAPPING_HOLD'):
        parse_govuk_pdf(PARENT, parent_bytes(raw, 1), ASSET, raw, retrieved_at=NOW)


@pytest.mark.parametrize('rss', [b'', b'UNKNOWN', b'999999999'])
def test_unknown_or_excessive_own_child_rss_holds_and_reaps_child(monkeypatch, rss):
    from types import SimpleNamespace
    import subprocess
    import newsroom.control_plane.govuk_pdf as module
    class Child:
        pid = 123
        returncode = None
        killed = False
        def poll(self): return self.returncode
        def communicate(self, **values):
            if 'timeout' in values: raise subprocess.TimeoutExpired('pdf-worker', 0.05)
            return b'', b''
        def kill(self): self.killed = True; self.returncode = -9
    child = Child()
    monkeypatch.setattr(module.subprocess, 'Popen', lambda *_args, **_values: child)
    monkeypatch.setattr(module.subprocess, 'run', lambda *_args, **_values: SimpleNamespace(stdout=rss))
    raw = pdf_bytes('First page.', 'Second page.')
    with pytest.raises(module.GovUkPdfHold, match='SOURCE_PDF_MEMORY_BOUND_HOLD'):
        module.parse_govuk_pdf(PARENT, parent_bytes(raw), ASSET, raw, retrieved_at=NOW)
    assert child.killed


def test_worker_wall_deadline_holds_and_reaps_only_its_child(monkeypatch):
    import newsroom.control_plane.govuk_pdf as module
    class Child:
        returncode = None
        killed = False
        def poll(self): return self.returncode
        def communicate(self, **_values): return b'', b''
        def kill(self): self.killed = True; self.returncode = -9
    child = Child()
    clock = iter([0.0, 20.0])
    monkeypatch.setattr(module.subprocess, 'Popen', lambda *_args, **_values: child)
    monkeypatch.setattr(module.time, 'monotonic', lambda: next(clock))
    raw = pdf_bytes('First page.', 'Second page.')
    with pytest.raises(module.GovUkPdfHold, match='SOURCE_PDF_WORKER_BOUND_HOLD'):
        module.parse_govuk_pdf(PARENT, parent_bytes(raw), ASSET, raw, retrieved_at=NOW)
    assert child.killed


def test_missing_declared_page_count_does_not_invent_an_extra_publication_gate():
    from newsroom.control_plane.govuk_pdf import parse_govuk_pdf
    raw = pdf_bytes('First page.', 'Second page.')
    parent = json.loads(parent_bytes(raw)); parent['details']['attachments'][0].pop('number_of_pages')
    document = parse_govuk_pdf(PARENT, json.dumps(parent).encode(), ASSET, raw, retrieved_at=NOW)
    assert len(document.page_inventory) == 2


@pytest.mark.parametrize('change', ['mime', 'size', 'filename', 'rights', 'undeclared'])
def test_declared_pdf_metadata_and_rights_stay_exact(change):
    from newsroom.control_plane.govuk_pdf import declared_pdf, GovUkPdfHold
    raw = pdf_bytes('First page.', 'Second page.')
    parent = json.loads(parent_bytes(raw)); asset = parent['details']['attachments'][0]
    if change == 'mime': asset['content_type'] = 'text/plain'
    elif change == 'size': asset['file_size'] = 8 * 1_048_576 + 1
    elif change == 'filename': asset['filename'] = 'other.pdf'
    elif change == 'rights': parent['details']['body'] = '<p>Third-party copyright requires permission from the copyright holder.</p>'
    elif change == 'undeclared': asset['url'] = ASSET + '.unknown'
    with pytest.raises(GovUkPdfHold):
        declared_pdf(PARENT, json.dumps(parent).encode(), ASSET, retrieved_at=NOW)


@pytest.mark.parametrize('output', [b'{', b'null', b'{}', b'{"pages":null}', b'{"hold":"PASS"}'])
def test_truncated_or_unknown_worker_output_never_becomes_complete(monkeypatch, output):
    import newsroom.control_plane.govuk_pdf as module
    class Child:
        returncode = 0
        def poll(self): return self.returncode
        def communicate(self, **_values): return output, b''
    monkeypatch.setattr(module.subprocess, 'Popen', lambda *_args, **_values: Child())
    raw = pdf_bytes('First page.', 'Second page.')
    with pytest.raises(module.GovUkPdfHold, match='SOURCE_PDF_WORKER_(BOUND|OUTPUT)_HOLD'):
        module.parse_govuk_pdf(PARENT, parent_bytes(raw), ASSET, raw, retrieved_at=NOW)


def test_extracted_text_bound_holds_without_truncation(monkeypatch):
    import newsroom.control_plane.govuk_pdf as module
    raw = pdf_bytes('First full page.', 'Second full page.')
    monkeypatch.setattr(module, 'MAX_TEXT_BYTES', 10)
    with pytest.raises(module.GovUkPdfHold, match='SOURCE_PDF_TEXT_BOUND_HOLD'):
        module.parse_govuk_pdf(PARENT, parent_bytes(raw), ASSET, raw, retrieved_at=NOW)


def test_encrypted_pdf_is_not_decrypted_or_counted_complete():
    import io
    from pypdf import PdfReader, PdfWriter
    from newsroom.control_plane.govuk_pdf import parse_govuk_pdf, GovUkPdfHold
    writer = PdfWriter()
    writer.append_pages_from_reader(PdfReader(io.BytesIO(pdf_bytes('First page.', 'Second page.'))))
    writer.encrypt('private')
    output = io.BytesIO(); writer.write(output)
    raw = output.getvalue()
    with pytest.raises(GovUkPdfHold, match='SOURCE_PDF_ENCRYPTED_HOLD'):
        parse_govuk_pdf(PARENT, parent_bytes(raw), ASSET, raw, retrieved_at=NOW)
