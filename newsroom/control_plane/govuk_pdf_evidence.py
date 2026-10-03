"""Exact current PDF reacquisition through the existing declared-asset contract."""
from newsroom.authority.canonical import digest_canonical
from .govuk_pdf import (MAX_RAW_BYTES, MAX_BODY_BYTES, POLICY_DIGEST as PARSER_POLICY_DIGEST,
                        declared_pdf, parse_govuk_pdf)
from .govuk_spreadsheet_evidence import GovUkSpreadsheetEvidenceAcquisition
from .native_source_intake import pdf_asset_url

VERSION = 'hermes-govuk-pdf-evidence-v1'
POLICY_DIGEST = digest_canonical({'version': VERSION, 'parser_policy': PARSER_POLICY_DIGEST,
    'method': 'GET', 'redirects': 0, 'max_parent_bytes': MAX_BODY_BYTES,
    'max_asset_bytes': MAX_RAW_BYTES, 'credentials': False, 'parent_and_asset_required': True})


class GovUkPdfEvidenceAcquisition(GovUkSpreadsheetEvidenceAcquisition):
    """PDF-specific decoder/byte scope; binding, fences and currentness are shared."""
    asset_url_for = staticmethod(pdf_asset_url)
    declare = staticmethod(declared_pdf)
    parse = staticmethod(parse_govuk_pdf)
    asset_byte_limit = MAX_RAW_BYTES
    parser_policy_digest = PARSER_POLICY_DIGEST
    version = VERSION
    reason_prefix = 'GOVUK_PDF'
    require_raw_identity = True
