# Usage

EdgeDeID Studio is an offline-first de-identification toolkit built around `deid_pipeline.DeidPipeline`.

## Python API

```python
from deid_pipeline import DeidPipeline

result = DeidPipeline(language="zh").process(
    "example.pdf",
    output_mode="replace",   # or: "blackbox"
    output_dir="out",        # optional: writes rebuilt artifacts when supported
)

print(result.text)
print(result.entities)
print(result.artifacts.get("output_path"))
```

## CLI

The repository ships a lightweight CLI entrypoint:

```bash
python main.py -i test_input/sample.txt --mode replace --json --output-dir out
python main.py -i test_input/sample.txt --mode detect
python main.py -i test_input/sample.txt --mode black
```

## Output Artifacts

When `--output-dir` / `output_dir` is provided, handlers may write rebuilt outputs:
- PDFs/images: redaction-style artifacts (black boxes)
- CSV/XLSX: rewritten data with replacements applied
- DOCX/PPTX: best-effort text replacement

Check `result.artifacts` for keys such as `output_path`, `rebuild_supported`, and redaction metadata.

## Offline-First Notes

- Runtime model loading is local-only (`local_files_only=True`).
- Dataset downloads/training are dev-only and are network-gated by default.
