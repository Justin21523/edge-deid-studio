from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class PromptTemplate:
    """Load and render a prompt template used for training and inference.

    The template must contain a `{RAW_TEXT}` placeholder.
    """

    template: str

    @classmethod
    def from_file(cls, path: str | Path) -> "PromptTemplate":
        template_path = Path(path).expanduser().resolve()
        if not template_path.exists():
            raise FileNotFoundError(f"Prompt template not found: {template_path}")
        return cls(template=template_path.read_text(encoding="utf-8"))

    def render(self, raw_text: str) -> str:
        return str(self.template).format(RAW_TEXT=str(raw_text or ""))

