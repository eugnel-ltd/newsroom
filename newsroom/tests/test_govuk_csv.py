"""Literal, bounded CSV cells on the existing declared-attachment route."""
import csv
import io
import json

import pytest

from newsroom.control_plane import govuk_spreadsheet as m
from newsroom.tests.test_govuk_spreadsheet import ASSET, parent, parse

URL = ASSET.replace('.ods', '.csv')


def test_csv_preserves_source_cells_without_type_or_header_inference():
    raw = '\ufeff Name ,Amount,Note,Empty\r\n Alice ,001.20,"line1\r\nline2",\r\nBob,=1+1,"a ""quote""", \r\n\r\n'.encode()
    doc = parse(raw, url=URL)
    assert doc == parse(raw, url=URL)
    assert 'Row 1: A=" Name "; B="Amount"' in doc.body_text
    assert 'Row 2: A=" Alice "; B="001.20"; C="line1\\r\\nline2"; D=""' in doc.body_text
    assert 'Row 3: A="Bob"; B="=1+1"; C="a \\"quote\\""; D=" "' in doc.body_text
    assert 'Row 4: [empty record]' in doc.body_text
    assert doc.document_type == 'spreadsheet'


def test_csv_preserves_minimised_retained_cp1252_publisher_and_name_cells():
    # Independent cells from the retained DfE Q1 2026-27 expenses CSV; this
    # minimisation does not associate the publisher's instruction with a person.
    raw = (
        b'Cell,Value\r\n'
        b'Publisher note,"avoid generic descriptions e.g., \x93Site Visit"" wherever possible."\r\n'
        b'Official name,Julia Kinniburgh\r\n'
    )
    text = parse(raw, url=URL).body_text
    assert 'B="avoid generic descriptions e.g., “Site Visit\\" wherever possible."' in text
    assert 'Row 3: A="Official name"; B="Julia Kinniburgh"' in text


def test_csv_cp1252_letter_ff_is_preserved_not_treated_as_bad_utf8():
    assert 'Row 1: A="ÿ"; B="a"' in parse(b'\xff,a\n', url=URL).body_text


@pytest.mark.parametrize('bom', [False, True])
def test_csv_prefers_utf8_without_changing_existing_non_ascii_cells(bom):
    raw = 'Name,Note\r\nÉlodie,"£12, 香港 “quoted”"\r\n'.encode()
    if bom:
        raw = b'\xef\xbb\xbf' + raw
    text = parse(raw, url=URL).body_text
    assert 'A="Élodie"; B="£12, 香港 “quoted”"' in text


@pytest.mark.parametrize('raw', [b'a,b\n1,2\n', b'\xef\xbb\xbfa,b\r\n1,2\r\n', b'Only\nvalue\n', b'a,b\n1\n2,3,4\n'])
def test_csv_preserves_single_column_and_ragged_records_as_published(raw):
    assert 'Row 1:' in parse(raw, url=URL).body_text


@pytest.mark.parametrize('raw', [b'a,\x00b\n', b'a,"unclosed\n', b'a,"closed"extra\n', b'\xef\xbb\xbf'])
def test_invalid_csv_does_not_become_complete_text(raw):
    with pytest.raises(ValueError):
        parse(raw, url=URL)


@pytest.mark.parametrize('byte', [0x81, 0x8d, 0x8f, 0x90, 0x9d])
def test_csv_undefined_cp1252_bytes_remain_errors(byte):
    with pytest.raises(UnicodeDecodeError):
        parse(b'Name,' + bytes([byte]) + b'\n', url=URL)


@pytest.mark.parametrize('raw', [b'\x93quote', b'\xff', b'\xc3('])
def test_csv_malformed_utf8_bom_never_falls_back_to_cp1252(raw):
    with pytest.raises(UnicodeDecodeError):
        parse(b'\xef\xbb\xbfName,' + raw + b'\n', url=URL)


@pytest.mark.parametrize('raw', [b'\xfe\xff\x4e\x2d', b'\xff\xfe\x2d\x4e',
                               b'\x00\x00\xfe\xff\x00\x00\x4e\x2d',
                               b'\xff\xfe\x00\x00\x2d\x4e\x00\x00'])
def test_csv_other_encoding_boms_are_not_cp1252_text(raw):
    with pytest.raises(ValueError):
        parse(raw, url=URL)


@pytest.mark.parametrize('control', ['\x00', '\x01', '\x08', '\x0b', '\x0c',
                                    '\x0e', '\x1f', '\x7f', '\x80', '\x85', '\x9f'])
def test_csv_rejects_controls_other_than_tab_cr_and_lf(control):
    with pytest.raises(ValueError, match='control'):
        parse(('Name,' + control + '\n').encode(), url=URL)


def test_csv_parser_policy_binds_the_new_decoder_contract():
    assert m.VERSION == 'hermes-govuk-spreadsheet-text-v3'
    assert m.POLICY_DIGEST != 'sha256:386f571df74bc50ed66c28de33447e1f5955abdd77db3dc3b40f5628fe6d3e49'


@pytest.mark.parametrize('key,value', [('content_type','application/vnd.ms-excel'), ('filename','other.csv'), ('file_size',1)])
def test_csv_requires_exact_declaration_not_only_a_file_suffix(key, value):
    raw = b'a,b\n1,2\n'; metadata = json.loads(parent(raw, URL))
    metadata['details']['attachments'][0][key] = value
    with pytest.raises(ValueError):
        parse(raw, url=URL, metadata=json.dumps(metadata).encode())


@pytest.mark.parametrize('bound', ['MAX_COLUMNS', 'MAX_ROWS', 'MAX_CELLS', 'MAX_BODY_BYTES'])
@pytest.mark.parametrize('encoding', ['utf8', 'cp1252'])
def test_csv_reuses_finite_row_column_cell_and_output_bounds(monkeypatch, bound, encoding):
    raw = b'a,b\n1,2\n' if bound != 'MAX_BODY_BYTES' else (b'a,' * 10 + b'a\n') * 100
    if encoding == 'cp1252':
        raw = b'\x93' + raw
    monkeypatch.setattr(m, bound, 1 if bound != 'MAX_BODY_BYTES' else 3000)
    with pytest.raises(ValueError):
        parse(raw, url=URL)


@pytest.mark.parametrize('notice', [
    'All rights reserved', 'All rights\nreserved', 'All  rights reserved',
    'Third-party\ncopyright', 'Permission required from the\ncopyright holder',
    'Not covered by the\r\nOpen Government Licence',
    'All\trights\u00a0reserved',
])
@pytest.mark.parametrize('encoding', ['utf8', 'cp1252'])
def test_csv_cell_rights_exclusion_stays_a_hold(notice, encoding):
    stream = io.StringIO(newline='')
    csv.writer(stream).writerows([['heading', 'note'], ['£value', notice]])
    with pytest.raises(ValueError, match='rights exclusion'):
        parse(stream.getvalue().encode(encoding), url=URL)
