from deid_pipeline.pii.detectors import get_detector
from deid_pipeline.pii.utils.replacer import Replacer

def test_replace_consistency():
    txt = "王小明借給王小明的妹妹0912345678。ID A123456789。"
    ents = get_detector("zh").detect(txt)
    new_txt, _ = Replacer().replace(txt, ents)
    assert new_txt.count("王小明") == 0
    # 同名兩處 → 新名字亦應一致
    names = [w for w in new_txt.split() if len(w) >= 3]
    assert len(set(names)) == 1
