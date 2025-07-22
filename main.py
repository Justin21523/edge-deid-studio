#from pii_models.detector import detect_pii
from deid_pipeline.pii import get_detector
from deid_pipeline.parser.text_extractor import extract_text

text = "Jennifer Lawrence 的電話是 +912345578，email 是 lucas@mail.com。"
#entities = detect_pii(text)

#for ent in entities:
#    print(f"{ent['label']}: {ent['text']}")
