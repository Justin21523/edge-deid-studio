def test_composite():
    from pii_models import get_detector
    d = get_detector("zh")
    txt = "王小明的身分證號是A123456789，手機 0912-345678。"
    ents = d.detect(txt)
    types = {e["type"] for e in ents}
    assert {"NAME", "ID", "PHONE"} <= types
