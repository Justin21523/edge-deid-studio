from __future__ import annotations

from deid_pipeline.training.pii_markup import extract_pii_blocks, mask_pii_blocks


def test_extract_pii_blocks_xml() -> None:
    text = 'Call <PII type="PHONE">0912-345-678</PII> now.'
    blocks = extract_pii_blocks(text)
    assert len(blocks) == 1
    assert blocks[0].pii_type == "PHONE"
    assert blocks[0].value == "0912-345-678"


def test_extract_pii_blocks_bracket() -> None:
    text = "Email [EMAIL]bob@example.com[/EMAIL]."
    blocks = extract_pii_blocks(text)
    assert len(blocks) == 1
    assert blocks[0].pii_type == "EMAIL"
    assert blocks[0].value == "bob@example.com"


def test_mask_pii_blocks_removes_wrappers() -> None:
    text = "Hi [NAME]王小明[/NAME]!"
    blocks = extract_pii_blocks(text)
    masked = mask_pii_blocks(text, blocks, placeholder="")
    assert "王小明" not in masked
    assert "[NAME]" not in masked
    assert "[/NAME]" not in masked

