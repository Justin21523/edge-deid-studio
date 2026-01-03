---
language:
  - zh
thumbnail: https://ckip.iis.sinica.edu.tw/files/ckip_logo.png
tags:
  - pytorch
  - token-classification
  - bert
  - zh
license: gpl-3.0
---

# CKIP BERT Base Chinese

This project provides traditional Chinese transformers models (including ALBERT, BERT, GPT2) and NLP tools (including word segmentation, part-of-speech tagging, named entity recognition).

## Homepage

- https://github.com/ckiplab/ckip-transformers

## Contributers

- [Mu Yang](https://muyang.pro) at [CKIP](https://ckip.iis.sinica.edu.tw) (Author & Maintainer)

## Usage

Please use BertTokenizerFast as tokenizer instead of AutoTokenizer.

```
from transformers import (
  BertTokenizerFast,
  AutoModel,
)

tokenizer = BertTokenizerFast.from_pretrained('bert-base-chinese')
model = AutoModel.from_pretrained('ckiplab/bert-base-chinese-ner')
```

For full usage and more information, please refer to https://github.com/ckiplab/ckip-transformers.
