"""Real bounded ZIP/XML fixtures; no network, spreadsheet application or model."""
import io
import json
import zipfile
from datetime import UTC, datetime

import pytest

from newsroom.control_plane import govuk_spreadsheet as m

NOW = datetime(2026, 9, 27, tzinfo=UTC)
PARENT = "https://www.gov.uk/government/publications/data"
ASSET = "https://assets.publishing.service.gov.uk/media/abc/data.ods"


def archive(parts):
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as z:
        for path, data in parts.items():
            z.writestr(path, data)
    return output.getvalue()


def parent(raw, url=ASSET):
    suffix = '.' + url.rsplit('.', 1)[1]
    return json.dumps({
        "base_path": "/government/publications/data", "locale": "en",
        "document_type": "transparency", "schema_name": "publication",
        "title": "Published data", "first_published_at": "2025-01-01T00:00:00Z",
        "public_updated_at": "2026-09-26T00:00:00Z",
        "links": {"organisations": [{"title": "Department for Education"}]},
        "details": {"body": "Published tables", "attachments": [{
            "url": url, "filename": url.rsplit('/', 1)[1], "attachment_type": "file",
            "id": "123", "title": "Allocations", "file_size": len(raw),
            "content_type": m.MIME_TYPES[suffix],
        }]},
    }).encode()


def ods(cells=None, *, rows=None):
    cells = cells or '<table:table-cell office:value-type="string"><text:p>Funding <text:span>values</text:span></text:p></table:table-cell>'
    rows = rows or '<table:table-row>' + cells + '</table:table-row>'
    return archive({
        "mimetype": m.MIME_TYPES['.ods'],
        "content.xml": f'''<office:document-content xmlns:office="{m.O[1:-1]}" xmlns:table="{m.T[1:-1]}" xmlns:text="{m.X[1:-1]}">
        <office:body><office:spreadsheet><table:table table:name="Notes">{rows}</table:table></office:spreadsheet></office:body></office:document-content>''',
    })


def xlsx(cells='<c r="A1" t="s"><v>0</v></c>', **parts):
    return archive({
        "xl/workbook.xml": f'<workbook xmlns="{m.S[1:-1]}" xmlns:r="{m.R[1:-1]}"><sheets><sheet name="Funding" state="hidden" r:id="rId1"/></sheets></workbook>',
        "xl/_rels/workbook.xml.rels": f'<Relationships xmlns="{m.P[1:-1]}"><Relationship Id="rId1" Type="{m.R[1:-1]}/worksheet" Target="worksheets/sheet1.xml"/></Relationships>',
        "xl/sharedStrings.xml": f'<sst xmlns="{m.S[1:-1]}"><si><r><t>Funding</t></r><r><t> values</t></r></si></sst>',
        "xl/worksheets/sheet1.xml": f'<worksheet xmlns="{m.S[1:-1]}"><sheetData><row r="1" hidden="1">{cells}</row></sheetData></worksheet>',
        **parts,
    })


def parse(raw, *, url=ASSET, metadata=None):
    return m.parse_govuk_spreadsheet(PARENT, metadata or parent(raw, url), url, raw, retrieved_at=NOW)


@pytest.mark.parametrize('suffix,factory', [('.ods', ods), ('.xlsx', xlsx)])
def test_published_cells_and_metadata_are_deterministic(suffix, factory):
    raw = factory(); url = ASSET.replace('.ods', suffix)
    a = parse(raw, url=url)
    assert a == parse(raw, url=url)
    assert a.title == 'Allocations' and a.document_type == 'spreadsheet'
    assert 'A="Funding values"' in a.body_text
    assert a.organisations == ('Department for Education',)
    assert a.updated == datetime(2026, 9, 26, tzinfo=UTC)
    assert a.exclusion_signals == ()


def test_ods_repeated_blank_tails_notes_currency_dates_and_cached_formula():
    raw = ods(rows='''<table:table-row>
      <table:table-cell office:value-type="currency" office:value="1234" office:currency="GBP"><text:p>£1,234</text:p><office:annotation><text:p>Provisional</text:p></office:annotation></table:table-cell>
      <table:table-cell office:value-type="date" office:date-value="2026-09-25"><text:p>25/09/2026</text:p></table:table-cell>
      <table:table-cell office:value-type="float" office:value="2" table:formula="of:=1+1"><text:p>2</text:p></table:table-cell>
      <table:table-cell table:number-columns-repeated="16381"/>
      </table:table-row><table:table-row table:number-rows-repeated="1048575"><table:table-cell table:number-columns-repeated="16384"/></table:table-row>''')
    text = parse(raw).body_text
    assert 'c:1234 GBP ("£1,234")' in text
    assert 'd:2026-09-25 ("25/09/2026")' in text
    assert '[cached]' in text and 'Provisional' in text
    assert len(text) < 1000 and 'Row 2' not in text


def test_ods_repeated_nonblank_coordinates_and_text_spacing():
    raw = ods(rows='''<table:table-row table:number-rows-repeated="2"><table:table-cell table:number-columns-repeated="2" office:value-type="string"><text:p>a<text:s text:c="2"/>b<text:tab/>c<text:line-break/>d</text:p></table:table-cell></table:table-row>''')
    text = parse(raw).body_text
    assert 'B="a  b\\tc\\nd"' in text


def test_xlsx_styles_and_cached_formula_without_running_external_links():
    raw = xlsx('<c r="B1" s="1"><f>1+1</f><v>2</v></c>', **{
        'xl/styles.xml': f'<styleSheet xmlns="{m.S[1:-1]}"><numFmts><numFmt numFmtId="164" formatCode="£#,##0"/></numFmts><cellXfs><xf numFmtId="0"/><xf numFmtId="164"/></cellXfs></styleSheet>',
        'xl/externalLinks/externalLink1.xml': '<never-requested/>',
    })
    text = parse(raw, url=ASSET.replace('.ods', '.xlsx')).body_text
    assert 'Number style 1: "£#,##0"' in text
    assert 'B=2@s1 [cached]' in text
    assert '(hidden; hidden rows/columns included)' in text


@pytest.mark.parametrize('change', ['foreign_host','undeclared','mime','size','duplicate','parent_rights','asset_rights','future','withdrawn'])
def test_declaration_rejects_incomplete_or_excluded_inventory(change):
    raw = ods(); url = ASSET; value = json.loads(parent(raw)); a = value['details']['attachments'][0]
    if change == 'foreign_host': url = ASSET.replace('assets.publishing.service.gov.uk', 'elsewhere.test')
    elif change == 'undeclared': a['url'] += 'x'
    elif change == 'mime': a['content_type'] = 'application/pdf'
    elif change == 'size': a['file_size'] = m.MAX_BODY_BYTES + 1
    elif change == 'duplicate': value['details']['attachments'].append(dict(a))
    elif change == 'parent_rights': value['details']['body'] = 'All rights reserved'
    elif change == 'asset_rights': a['copyright_notice'] = 'Third-party copyright'
    elif change == 'future': value['public_updated_at'] = '2027-01-01T00:00:00Z'
    else: value['withdrawn_notice'] = {'explanation': 'Withdrawn'}
    with pytest.raises(ValueError): parse(raw, url=url, metadata=json.dumps(value).encode())


@pytest.mark.parametrize('cells', [
    '<c r="A1"><f>NOW()</f></c>', '<c r="A1" t="s"><v>99</v></c>',
    '<c r="A2"><v>2</v></c>', '<c r="A1"><v>2</v></c><c r="A1"><v>3</v></c>',
    '<c r="A1" t="e"><v>#REF!</v></c>', '<c r="XFE1"><v>2</v></c>',
    '<c r="A1"><v>NaN</v></c>',
])
def test_xlsx_invalid_cells_hold(cells):
    with pytest.raises(ValueError): parse(xlsx(cells), url=ASSET.replace('.ods', '.xlsx'))


@pytest.mark.parametrize('cells', [
    '<table:table-cell table:formula="of:=1+1"/>',
    '<table:table-cell office:value-type="currency"><text:p>£2</text:p></table:table-cell>',
    '<table:table-cell table:number-columns-repeated="16385"/>',
    '<table:table-cell office:value-type="unknown"/>',
])
def test_ods_incomplete_cells_hold(cells):
    with pytest.raises(ValueError): parse(ods(cells))


def test_archive_size_and_xml_entity_budgets(monkeypatch):
    raw = ods()
    with pytest.raises(ValueError): parse(raw[:-1], metadata=parent(raw))
    monkeypatch.setattr(m, 'MAX_EXPANDED_BYTES', 8)
    with pytest.raises(ValueError): parse(raw)
    monkeypatch.undo()
    hostile = archive({'mimetype': m.MIME_TYPES['.ods'], 'content.xml': '<!DOCTYPE root [<!ENTITY x SYSTEM "file:///etc/passwd">]><root>&x;</root>'})
    with pytest.raises(ValueError): parse(hostile)
    for name in ['../content.xml', 'Scripts/run.py', 'xl/vbaProject.bin']:
        with pytest.raises(ValueError): parse(archive({name: 'contents'}))


def test_repeated_nonblank_and_output_budgets(monkeypatch):
    raw = ods(rows='<table:table-row table:number-rows-repeated="1048576"><table:table-cell office:value-type="string"><text:p>x</text:p></table:table-cell></table:table-row>')
    with pytest.raises(ValueError): parse(raw)
    monkeypatch.setattr(m, 'MAX_BODY_BYTES', 500)
    with pytest.raises(ValueError): parse(ods())


def test_xml_node_and_part_budgets(monkeypatch):
    raw = ods()
    monkeypatch.setattr(m, 'MAX_XML_NODES', 2)
    with pytest.raises(ValueError, match='node count'): parse(raw)
    monkeypatch.undo()
    monkeypatch.setattr(m, 'MAX_XML_BYTES', 2)
    with pytest.raises(ValueError, match='XML size'): parse(raw)
    monkeypatch.undo()
    raw = archive({str(n): '' for n in range(m.MAX_PARTS + 1)})
    with pytest.raises(ValueError, match='archive'): parse(raw)


def test_repeated_large_cell_is_bounded_before_row_join():
    raw = ods('<table:table-cell table:number-columns-repeated="16384" office:value-type="string"><text:p>' + 'a' * 10000 + '</text:p></table:table-cell>')
    with pytest.raises(ValueError, match='row text'): parse(raw)


@pytest.mark.parametrize('attributes', [
    'office:value-type="float" office:value="NaN"',
    'office:value-type="date" office:date-value="2026-99-99"',
    'office:value-type="boolean" office:boolean-value="maybe"',
])
def test_ods_typed_values_are_not_mislabelled(attributes):
    with pytest.raises(ValueError): parse(ods(f'<table:table-cell {attributes}/>'))


def test_text_box_content_is_not_silently_omitted():
    raw = ods('<table:table-cell><frame xmlns="urn:oasis:names:tc:opendocument:xmlns:drawing:1.0"><text-box><text:p>A qualification</text:p></text-box></frame></table:table-cell>')
    with pytest.raises(ValueError, match='non-cell'): parse(raw)


def test_ods_merged_anchor_records_its_extent():
    raw = ods('<table:table-cell table:number-columns-spanned="2" office:value-type="string"><text:p>Header</text:p></table:table-cell><table:covered-table-cell/>')
    assert 'A="Header" [spans 2 columns]' in parse(raw).body_text


def test_duplicate_archive_part_is_rejected():
    raw = ods(); output = io.BytesIO(raw)
    with zipfile.ZipFile(output, 'a') as z:
        with pytest.warns(UserWarning): z.writestr('mimetype', m.MIME_TYPES['.ods'])
    with pytest.raises(ValueError, match='archive'): parse(output.getvalue())


def test_xlsx_escaped_text_and_phonetic_guides_preserve_display_string():
    raw = xlsx(**{'xl/sharedStrings.xml': f'<sst xmlns="{m.S[1:-1]}"><si><t>A_x000A_B_x005F_x0041_ _xD83D__xDE00_</t><rPh sb="0" eb="1"><t>pronunciation</t></rPh></si></sst>'})
    text = parse(raw, url=ASSET.replace('.ods', '.xlsx')).body_text
    assert 'A="A\\nB_x0041_ 😀"' in text and 'pronunciation' not in text


def test_rights_notice_markup_is_not_hidden_from_exclusion_check():
    raw = ods(); value = json.loads(parent(raw))
    value['details']['body'] = '<p>All rights <em>reserved</em></p>'
    with pytest.raises(ValueError, match='rights exclusion'):
        parse(raw, metadata=json.dumps(value).encode())
