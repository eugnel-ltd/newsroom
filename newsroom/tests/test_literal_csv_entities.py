"""Only declared, exact CSV name cells contribute source-bound entities."""

import json

import pytest

from newsroom.control_plane.evidence import (
    _has_bounded_named_entity_shape, bounded_named_entities, rendered_named_entities,
)


PREFIX = (
    "Attachment: https://assets.publishing.service.gov.uk/media/fixture/officials.csv\n"
    "Published CSV cells: Row and column identify each literal text cell. Whitespace, empty fields, quoted newlines and formula-like text are preserved; nothing is executed. No header or numeric types are inferred.\n"
    'Sheet "CSV"\n'
)
SENIOR_HEADER = 'Row 1: A="Senior Official\'s Name "; B="Date"'
PARTY_HEADER = 'Row 1: A="Date"; B="Purpose"; C="Individual or organisation that provided hospitality "'
COMBINED_HEADER = (
    'Row 1: A="Senior Official\'s Name "; B="Date"; '
    'C="Individual or organisation that provided hospitality "'
)


def _row(number, cells):
    return f"Row {number}: " + "; ".join(
        f"{column}={json.dumps(value, ensure_ascii=False)}" for column, value in cells
    )


def test_literal_csv_person_cell_is_exact_and_survives_rendering():
    row = _row(2, [('A', 'Marianthi Leontaridi'), ('B', '2026-01-14')])
    source = PREFIX + SENIOR_HEADER + '\n' + row
    names = bounded_named_entities(row, source_context=source)
    assert names == frozenset({('Marianthi Leontaridi', 'PERSON')})
    assert rendered_named_entities('Marianthi Leontaridi 出席會議。', names) == names
    assert bounded_named_entities('Marianthi Leontaridi', source_context=source) == frozenset()


def test_literal_csv_party_cell_requires_existing_organisation_shape():
    row = _row(2, [('A', '2026-01-14'), ('B', 'Meeting; said "hello"\nNext line'),
                   ('C', 'Department for Education')])
    source = PREFIX + PARTY_HEADER + '\n' + row
    assert bounded_named_entities(row, source_context=source) == frozenset({
        ('Department for Education', 'ORGANISATION'),
    })


def test_literal_csv_mixed_party_group_name_is_table_bound_not_generic_prose():
    row = _row(2, [('A', '2026-01-14'), ('B', 'Hospitality; "dinner"'),
                   ('C', 'Boston Consulting Group')])
    source = PREFIX + PARTY_HEADER + '\n' + row
    assert bounded_named_entities(row, source_context=source) == frozenset({
        ('Boston Consulting Group', 'ORGANISATION'),
    })
    assert _has_bounded_named_entity_shape(
        'Boston Consulting Group', 'ORGANISATION', source_context=row,
    )
    assert not _has_bounded_named_entity_shape(
        'Boston Consulting Group', 'ORGANISATION', source_context='ordinary prose',
    )
    assert bounded_named_entities('Boston Consulting Group', source_context=source) == frozenset()
    assert bounded_named_entities(row, source_context=source.replace('Boston Consulting Group', 'Other Group')) == frozenset()
    other_column = _row(2, [('A', '2026-01-14'), ('B', 'Boston Consulting Group'),
                            ('C', 'N/A')])
    assert bounded_named_entities(other_column, source_context=PREFIX + PARTY_HEADER + '\n' + other_column) == frozenset()


def test_literal_csv_combined_name_columns_with_declared_title_prefix():
    row = _row(2, [('A', 'Marianthi Leontaridi'), ('B', '2026-01-14'),
                   ('C', 'Boston Consulting Group')])
    canonical = PREFIX + COMBINED_HEADER + '\n' + row
    expected = frozenset({
        ('Marianthi Leontaridi', 'PERSON'),
        ('Boston Consulting Group', 'ORGANISATION'),
    })
    assert bounded_named_entities(row, source_context=canonical) == expected
    assert bounded_named_entities(
        row, source_context='Senior officials hospitality\n\n' + canonical,
    ) == expected
    assert bounded_named_entities(
        row, source_context='Unrelated prose\nExtra prose\n\n' + canonical,
    ) == frozenset()


@pytest.mark.parametrize('change', [
    lambda source, row: source.replace(row, row.replace('2026-01-14', '2026-01-15')),
    lambda source, row: source.replace('Attachment: https://assets.publishing.service.gov.uk/',
                                        'Attachment: https://example.org/'),
    lambda source, row: source.replace('Sheet "CSV"', 'Sheet "Sheet1"'),
    lambda source, row: source.replace('Published CSV cells:', 'Other cells:'),
    lambda source, row: source.replace('Senior Official\'s Name ', 'Notes'),
    lambda source, row: source.replace('B="Date"', 'B="Senior Official\'s Name "'),
    lambda source, row: source.replace('Row 1:', 'Row 3:'),
    lambda source, row: source.replace('B="2026-01-14"', 'A="2026-01-14"'),
    lambda source, row: source.replace('A="Marianthi Leontaridi"',
                                        'A="Marianthi Leontaridi"; A="Other"'),
])
def test_literal_csv_person_rejects_changed_or_malformed_context(change):
    row = _row(2, [('A', 'Marianthi Leontaridi'), ('B', '2026-01-14')])
    source = PREFIX + SENIOR_HEADER + '\n' + row
    assert bounded_named_entities(row, source_context=change(source, row)) == frozenset()


@pytest.mark.parametrize('value', [
    'Nil Return', 'N/A', 'None', 'Not Applicable',
    'Marianthi "Leontaridi"', 'Marianthi; Leontaridi',
])
def test_literal_csv_nil_values_are_not_entities(value):
    row = _row(2, [('A', value), ('B', '2026-01-14')])
    assert bounded_named_entities(row, source_context=PREFIX + SENIOR_HEADER + '\n' + row) == frozenset()


def test_literal_csv_never_types_ordinary_prose_or_untyped_party_cell():
    row = _row(2, [('A', 'The policy changed'), ('B', '2026-01-14')])
    assert bounded_named_entities(row, source_context=PREFIX + SENIOR_HEADER + '\n' + row) == frozenset()
    party = _row(2, [('A', '2026-01-14'), ('B', 'Marianthi Leontaridi'),
                     ('C', 'Marianthi Leontaridi')])
    assert bounded_named_entities(party, source_context=PREFIX + PARTY_HEADER + '\n' + party) == frozenset()
    ambiguous = PARTY_HEADER.replace('Individual or organisation that provided hospitality ', 'Name')
    organisation = _row(2, [('A', '2026-01-14'), ('B', 'Meeting'), ('C', 'Boston Consulting Group')])
    assert bounded_named_entities(organisation, source_context=PREFIX + ambiguous + '\n' + organisation) == frozenset()


def test_literal_csv_generic_prose_does_not_become_a_declaration():
    row = _row(2, [('A', 'Marianthi Leontaridi'), ('B', '2026-01-14')])
    assert bounded_named_entities(row, source_context='A news article quotes ' + row) == frozenset()
