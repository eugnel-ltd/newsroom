"""Literal, bounded CSV cells on the existing declared-attachment route."""
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


@pytest.mark.parametrize('raw', [b'a,b\n1,2\n', b'\xef\xbb\xbfa,b\r\n1,2\r\n', b'Only\nvalue\n', b'a,b\n1\n2,3,4\n'])
def test_csv_preserves_single_column_and_ragged_records_as_published(raw):
    assert 'Row 1:' in parse(raw, url=URL).body_text


@pytest.mark.parametrize('raw', [b'\xff,a\n', b'a,\x00b\n', b'a,"unclosed\n', b'a,"closed"extra\n', b'\xef\xbb\xbf'])
def test_invalid_csv_does_not_become_complete_text(raw):
    with pytest.raises(ValueError):
        parse(raw, url=URL)


@pytest.mark.parametrize('key,value', [('content_type','application/vnd.ms-excel'), ('filename','other.csv'), ('file_size',1)])
def test_csv_requires_exact_declaration_not_only_a_file_suffix(key, value):
    raw = b'a,b\n1,2\n'; metadata = json.loads(parent(raw, URL))
    metadata['details']['attachments'][0][key] = value
    with pytest.raises(ValueError):
        parse(raw, url=URL, metadata=json.dumps(metadata).encode())


@pytest.mark.parametrize('bound', ['MAX_COLUMNS', 'MAX_ROWS', 'MAX_CELLS', 'MAX_BODY_BYTES'])
def test_csv_reuses_finite_row_column_cell_and_output_bounds(monkeypatch, bound):
    raw = b'a,b\n1,2\n' if bound != 'MAX_BODY_BYTES' else (b'a,' * 10 + b'a\n') * 100
    monkeypatch.setattr(m, bound, 1 if bound != 'MAX_BODY_BYTES' else 3000)
    with pytest.raises(ValueError):
        parse(raw, url=URL)


def test_csv_cell_rights_exclusion_stays_a_hold():
    with pytest.raises(ValueError, match='rights exclusion'):
        parse(b'heading,note\nvalue,All rights reserved\n', url=URL)
