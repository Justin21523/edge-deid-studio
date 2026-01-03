from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from deid_pipeline.storage.env import apply_cache_env_defaults
from deid_pipeline.storage.layout import StorageLayout
from deid_pipeline.training.deid_eval import (
    DeidExample,
    ExampleReport,
    aggregate_reports,
    detect_pii_blocks,
    evaluate_prediction,
    iter_deid_examples_jsonl,
    iter_predictions_jsonl,
    load_banned_phrases,
)
from deid_pipeline.training.prompts import PromptTemplate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate GPT-2 DeID predictions (dev-only).")
    parser.add_argument("--data-jsonl", required=True, help="JSONL with {id,input,output}.")
    parser.add_argument("--predictions-jsonl", default="", help="Optional JSONL with {id,prediction}.")
    parser.add_argument("--model-dir", default="", help="Optional local model dir to generate predictions.")
    parser.add_argument(
        "--prompt-template",
        default="configs/prompts/deid_zh_v1.txt",
        help="Prompt template file path (must include {RAW_TEXT}).",
    )
    parser.add_argument(
        "--banned-phrases",
        default="configs/eval/deid_format_zh.yaml",
        help="YAML/JSON list of banned output phrases.",
    )
    parser.add_argument("--output-dir", required=True, help="Directory for metrics/predictions outputs.")
    parser.add_argument("--max-examples", type=int, default=0, help="Optional cap on evaluated examples.")
    parser.add_argument(
        "--regex-rules",
        default="",
        help="Optional regex rules YAML for weak-label PII spans when markup is missing.",
    )
    parser.add_argument("--max-prompt-tokens", type=int, default=768, help="Max tokens for the input prompt.")
    parser.add_argument("--max-new-tokens", type=int, default=256, help="Max tokens to generate per chunk.")
    parser.add_argument(
        "--do-sample",
        action="store_true",
        help="Enable sampling (default: greedy).",
    )
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature.")
    parser.add_argument("--top-p", type=float, default=0.9, help="Nucleus sampling top-p.")
    parser.add_argument("--seed", type=int, default=0, help="Generation seed.")
    parser.add_argument(
        "--allow-network",
        action="store_true",
        help="Allow network access (not recommended; evaluation should be local-only).",
    )
    parser.add_argument(
        "--fluency-model-dir",
        default="",
        help="Optional model dir used only for perplexity-based fluency proxy.",
    )
    return parser.parse_args()


def _python_info() -> Dict[str, Any]:
    import platform
    import sys

    return {
        "python_executable": sys.executable,
        "python_version": sys.version,
        "platform": platform.platform(),
    }


def _try_nvidia_smi() -> str:
    try:
        out = subprocess.check_output(["nvidia-smi", "-L"], stderr=subprocess.STDOUT, text=True)
        return out.strip()
    except Exception:
        return ""


def _chunk_text_by_newlines(
    raw_text: str,
    *,
    tokenizer,
    prompt: PromptTemplate,
    max_prompt_tokens: int,
) -> List[str]:
    lines = str(raw_text or "").split("\n")
    if len(lines) <= 1:
        return [str(raw_text or "")]

    chunks: List[str] = []
    current: List[str] = []

    for line in lines:
        candidate = "\n".join(current + [line]) if current else line
        tok_len = len(tokenizer(prompt.render(candidate), add_special_tokens=False)["input_ids"])
        if tok_len <= int(max_prompt_tokens):
            current.append(line)
            continue

        if current:
            chunks.append("\n".join(current))
            current = [line]
        else:
            # Single line is too long: hard truncate by characters (best-effort).
            chunks.append(line[:2000])
            current = []

    if current:
        chunks.append("\n".join(current))
    return [c for c in chunks if str(c or "").strip()]


def _postprocess_prediction(text: str) -> str:
    out = str(text or "").strip()
    for prefix in ["輸出：", "Output:", "OUTPUT:"]:
        if out.startswith(prefix):
            out = out[len(prefix) :].lstrip()
    return out


def _generate_predictions(
    examples: Sequence[DeidExample],
    *,
    model_dir: Path,
    prompt_template: PromptTemplate,
    max_prompt_tokens: int,
    max_new_tokens: int,
    do_sample: bool,
    temperature: float,
    top_p: float,
    seed: int,
    allow_network: bool,
) -> Dict[str, str]:
    import torch  # type: ignore
    from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed  # type: ignore

    set_seed(int(seed))

    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), use_fast=True, local_files_only=not allow_network)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(str(model_dir), local_files_only=not allow_network)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    preds: Dict[str, str] = {}
    for ex in examples:
        chunks = _chunk_text_by_newlines(
            ex.input_text,
            tokenizer=tokenizer,
            prompt=prompt_template,
            max_prompt_tokens=int(max_prompt_tokens),
        )
        out_parts: List[str] = []
        for chunk in chunks:
            prompt_text = prompt_template.render(chunk)
            encoded = tokenizer(prompt_text, return_tensors="pt", add_special_tokens=False)
            input_ids = encoded["input_ids"].to(device)
            attention_mask = encoded.get("attention_mask")
            if attention_mask is not None:
                attention_mask = attention_mask.to(device)

            with torch.no_grad():
                generated = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=int(max_new_tokens),
                    do_sample=bool(do_sample),
                    temperature=float(temperature),
                    top_p=float(top_p),
                    num_beams=1,
                    pad_token_id=int(tokenizer.eos_token_id),
                    eos_token_id=int(tokenizer.eos_token_id),
                )

            gen_ids = generated[0][input_ids.shape[1] :]
            decoded = tokenizer.decode(gen_ids, skip_special_tokens=True)
            out_parts.append(_postprocess_prediction(decoded))

        preds[str(ex.example_id)] = "\n".join([p for p in out_parts if p.strip()]).strip()
    return preds


def _compute_perplexity(
    texts: Sequence[str],
    *,
    model_dir: Path,
    allow_network: bool,
    max_length: int = 512,
) -> float:
    import math

    import torch  # type: ignore
    from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore

    if not texts:
        return 0.0

    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), use_fast=True, local_files_only=not allow_network)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(str(model_dir), local_files_only=not allow_network)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    losses: List[float] = []
    for text in texts:
        enc = tokenizer(
            str(text or ""),
            return_tensors="pt",
            truncation=True,
            max_length=int(max_length),
            add_special_tokens=False,
        )
        input_ids = enc["input_ids"].to(device)
        attention_mask = enc.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)

        with torch.no_grad():
            out = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
        loss = float(out.loss.detach().cpu().item())
        losses.append(loss)

    avg_loss = float(sum(losses) / float(len(losses)))
    return float(math.exp(avg_loss))


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    layout = StorageLayout.from_project_root(repo_root)
    apply_cache_env_defaults(layout=layout)

    allow_network = bool(args.allow_network)

    out_dir = Path(args.output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    prompt_template = PromptTemplate.from_file(args.prompt_template)
    banned_phrases = load_banned_phrases(args.banned_phrases)

    examples: List[DeidExample] = []
    for idx, ex in enumerate(iter_deid_examples_jsonl(args.data_jsonl)):
        examples.append(ex)
        if int(args.max_examples) > 0 and idx + 1 >= int(args.max_examples):
            break

    preds: Dict[str, str] = {}
    pred_path: Optional[Path] = None
    if str(args.predictions_jsonl).strip():
        pred_path = Path(args.predictions_jsonl).expanduser().resolve()
        preds = {p.example_id: p.prediction_text for p in iter_predictions_jsonl(pred_path)}
    elif str(args.model_dir).strip():
        model_dir = Path(args.model_dir).expanduser().resolve()
        preds = _generate_predictions(
            examples,
            model_dir=model_dir,
            prompt_template=prompt_template,
            max_prompt_tokens=int(args.max_prompt_tokens),
            max_new_tokens=int(args.max_new_tokens),
            do_sample=bool(args.do_sample),
            temperature=float(args.temperature),
            top_p=float(args.top_p),
            seed=int(args.seed),
            allow_network=allow_network,
        )
        pred_path = out_dir / "predictions.jsonl"
        with pred_path.open("w", encoding="utf-8") as f:
            for ex in examples:
                f.write(
                    json.dumps(
                        {"id": ex.example_id, "prediction": preds.get(ex.example_id, "")},
                        ensure_ascii=False,
                    )
                    + "\n"
                )
    else:
        raise ValueError("Provide either --predictions-jsonl or --model-dir.")

    # Weak-label regex detector (optional).
    regex_spans_by_id: Dict[str, List[Tuple[int, int, str]]] = {}
    if str(args.regex_rules).strip():
        from deid_pipeline.pii.detectors.regex_detector import RegexDetector

        detector = RegexDetector(config_path=str(args.regex_rules))
        for ex in examples:
            entities = detector.detect(ex.input_text)
            regex_spans_by_id[str(ex.example_id)] = [
                (int(ent["span"][0]), int(ent["span"][1]), str(ent["type"])) for ent in entities
            ]

    reports: List[ExampleReport] = []
    detailed_path = out_dir / "detailed_predictions.jsonl"
    failures_path = out_dir / "error_samples.jsonl"

    failure_rows: List[Dict[str, Any]] = []
    with detailed_path.open("w", encoding="utf-8") as f:
        for ex in examples:
            pred_text = preds.get(str(ex.example_id), "")
            pii_blocks = detect_pii_blocks(
                ex.input_text,
                prefer_markup=True,
                regex_spans=regex_spans_by_id.get(str(ex.example_id)),
            )
            report = evaluate_prediction(ex, pred_text, pii_blocks=pii_blocks, banned_phrases=banned_phrases)
            reports.append(report)

            row = {
                "id": ex.example_id,
                "input": ex.input_text,
                "target": ex.target_text,
                "prediction": pred_text,
                "metrics": {
                    "pii_removal_recall": report.pii_removal_recall,
                    "pii_total": report.pii_total,
                    "pii_leak_count": report.pii_leak_count,
                    "non_pii_similarity": report.non_pii_similarity,
                    "over_rewrite_rate": report.over_rewrite_rate,
                    "type_consistency": report.type_consistency,
                    "type_details": report.type_details,
                    "repetition_3gram_rate": report.repetition_3gram_rate,
                    "format_compliant": report.format_compliant,
                    "format_triggers": report.format_triggers,
                },
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

            if report.pii_leak_count > 0 or (not report.format_compliant) or report.type_consistency < 1.0:
                failure_rows.append(row)

    # Keep only a small, readable failure sample file.
    max_failures = 50
    with failures_path.open("w", encoding="utf-8") as f:
        for row in failure_rows[:max_failures]:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    metrics = aggregate_reports(reports)
    metrics.update(
        {
            "created_unix_s": time.time(),
            "data_jsonl": str(Path(args.data_jsonl).expanduser().resolve()),
            "predictions_jsonl": str(pred_path) if pred_path else "",
            "model_dir": str(Path(args.model_dir).expanduser().resolve()) if str(args.model_dir).strip() else "",
            "prompt_template": str(Path(args.prompt_template).expanduser().resolve()),
            "banned_phrases": str(Path(args.banned_phrases).expanduser().resolve()),
            "regex_rules": str(Path(args.regex_rules).expanduser().resolve()) if str(args.regex_rules).strip() else "",
            "max_examples": int(args.max_examples),
            "generation": {
                "max_prompt_tokens": int(args.max_prompt_tokens),
                "max_new_tokens": int(args.max_new_tokens),
                "do_sample": bool(args.do_sample),
                "temperature": float(args.temperature),
                "top_p": float(args.top_p),
                "seed": int(args.seed),
            },
            "system": {
                **_python_info(),
                "nvidia_smi": _try_nvidia_smi(),
            },
        }
    )

    if str(args.fluency_model_dir).strip():
        ppl = _compute_perplexity(
            [preds.get(str(ex.example_id), "") for ex in examples],
            model_dir=Path(args.fluency_model_dir).expanduser().resolve(),
            allow_network=allow_network,
        )
        metrics["fluency_perplexity"] = float(ppl)

    (out_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
