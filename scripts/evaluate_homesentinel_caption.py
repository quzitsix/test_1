#!/usr/bin/env python3
"""Run a caption-only HomeSentinel probe with a local HuggingFace model.

This is intentionally separate from the video adapter: the existing visual
adapter consumes video frames and would silently ignore caption_path.  This
script gives the model only the ordered caption text available before each
query's video_cutoff_idx.
"""

from __future__ import annotations

import argparse
import json
import re
import string
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--model-path",
        default="/data/quzitsix/models/Qwen3-VL-2B-Instruct",
    )
    p.add_argument(
        "--data-root",
        type=Path,
        default=Path("/data/HomeSentinel/asuka"),
    )
    p.add_argument(
        "--output",
        type=Path,
        default=Path("runs/homesentinel-caption-qwen3-vl-2b/predictions.jsonl"),
    )
    p.add_argument("--category", choices=("owner", "home", "event"))
    p.add_argument("--limit", type=int, default=0, help="0 means all selected queries")
    p.add_argument(
        "--caption-mode",
        choices=("summary", "summary_speech"),
        default="summary",
        help="summary is activity/location/time; summary_speech also includes speech text",
    )
    p.add_argument("--max-context-tokens", type=int, default=200_000)
    p.add_argument("--max-new-tokens", type=int, default=128)
    p.add_argument("--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--resume", action="store_true")
    return p.parse_args()


def load_data(root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    captions = json.loads((root / "merged_captions.json").read_text(encoding="utf-8"))
    queries: list[dict[str, Any]] = []
    for name in ("owner", "home", "event"):
        path = root / "benchmark_queries" / f"{name}.json"
        for row in json.loads(path.read_text(encoding="utf-8")):
            queries.append({**row, "_category_file": name})
    captions.sort(key=lambda row: int(row["video_idx"]))
    queries.sort(key=lambda row: ({"owner": 0, "home": 1, "event": 2}[row["_category_file"]],
                                  row["query_id"]))
    return captions, queries


def select_queries(
    queries: list[dict[str, Any]], category: str | None, limit: int
) -> list[dict[str, Any]]:
    if category:
        queries = [q for q in queries if q["_category_file"] == category]
    if not limit or limit >= len(queries):
        return queries
    # A small pilot should cover all three question families when no category
    # was requested, rather than selecting only owner questions.
    if not category:
        groups = {name: [q for q in queries if q["_category_file"] == name]
                  for name in ("owner", "home", "event")}
        picked: list[dict[str, Any]] = []
        while len(picked) < limit and any(groups.values()):
            for name in ("owner", "home", "event"):
                if groups[name] and len(picked) < limit:
                    picked.append(groups[name].pop(0))
        return picked
    return queries[:limit]


def caption_context(
    captions: list[dict[str, Any]], cutoff: int, mode: str
) -> tuple[str, int, int]:
    visible = captions if cutoff == -1 else [v for v in captions if v["video_idx"] <= cutoff]
    chunks: list[str] = []
    n_segments = 0
    n_events = 0
    for video in visible:
        chunks.append(
            f"VIDEO {video['video_idx']} id={video['video_id']} "
            f"weekday={video.get('time_of_week', '')}"
        )
        for seg in video.get("segments", []):
            n_segments += 1
            chunks.append(
                f"SCENE {seg.get('scene_id', '')} "
                f"time_of_day={seg.get('time_of_day', '')} "
                f"place={seg.get('place', '')}"
            )
            activity = (seg.get("activity_summary") or "").strip()
            if activity:
                chunks.append(f"CAPTION: {activity}")
            if mode == "summary_speech":
                for speech in seg.get("speech") or []:
                    text = (speech.get("text") or "").strip()
                    if text:
                        chunks.append(f"SPEECH {speech.get('t', '')}: {text}")
            n_events += len(seg.get("events") or [])
    return "\n".join(chunks), len(visible), n_segments


def prompt_for(context: str, question: str) -> str:
    return (
        "You are answering a household long-term memory benchmark. "
        "Use only the caption observations below. They are ordered by video "
        "time and stop at the allowed cutoff. Do not invent facts that are not "
        "supported by the captions. If the captions do not support an answer, "
        "say that the information is not available. Answer the question in "
        "one or two concise sentences.\n\n"
        "CAPTION OBSERVATIONS\n"
        "---------------------\n"
        f"{context}\n\n"
        "QUESTION\n"
        "--------\n"
        f"{question}\n\n"
        "ANSWER:"
    )


def normalise(text: str) -> list[str]:
    text = text.lower().translate(str.maketrans("", "", string.punctuation))
    return re.findall(r"[a-z0-9]+", text)


def token_f1(pred: str, gold: str) -> float:
    p, g = Counter(normalise(pred)), Counter(normalise(gold))
    if not p or not g:
        return float(p == g)
    overlap = sum((p & g).values())
    precision = overlap / sum(p.values())
    recall = overlap / sum(g.values())
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def score_row(pred: str, gold: str) -> dict[str, float | bool]:
    pn, gn = " ".join(normalise(pred)), " ".join(normalise(gold))
    return {
        "exact_normalized": pn == gn,
        "gold_substring": bool(gn) and gn in pn,
        "token_f1": round(token_f1(pred, gold), 4),
    }


def load_model(args: argparse.Namespace):
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor

    dtype = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[args.dtype]
    processor = AutoProcessor.from_pretrained(args.model_path, local_files_only=True)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model_path,
        dtype=dtype,
        device_map=args.device,
        local_files_only=True,
    )
    model.eval()
    return torch, processor, model


def generate(
    torch: Any,
    processor: Any,
    model: Any,
    prompt: str,
    device: str,
    max_new_tokens: int,
) -> tuple[str, int]:
    messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )
    prompt_tokens = int(inputs["input_ids"].shape[1])
    inputs = inputs.to(device)
    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )
    continuation = output[:, prompt_tokens:]
    text = processor.batch_decode(
        continuation,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0].strip()
    return text, prompt_tokens


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[row["category"]].append(row)
    result: dict[str, Any] = {}
    for key, values in sorted(groups.items()):
        ok = [v for v in values if v["status"] == "ok"]
        result[key] = {
            "n": len(values),
            "ok": len(ok),
            "exact_normalized": sum(bool(v["score"]["exact_normalized"]) for v in ok),
            "gold_substring": sum(bool(v["score"]["gold_substring"]) for v in ok),
            "mean_token_f1": round(
                sum(float(v["score"]["token_f1"]) for v in ok) / len(ok), 4
            ) if ok else None,
        }
    return result


def main() -> int:
    args = parse_args()
    captions, all_queries = load_data(args.data_root)
    queries = select_queries(all_queries, args.category, args.limit)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    completed: set[str] = set()
    if args.resume and args.output.exists():
        for line in args.output.read_text(encoding="utf-8").splitlines():
            if line.strip():
                completed.add(json.loads(line)["query_id"])
    torch, processor, model = load_model(args)
    tokenizer = getattr(processor, "tokenizer", processor)
    context_cache: dict[tuple[int, str], tuple[str, int, int]] = {}
    rows: list[dict[str, Any]] = []
    with args.output.open("a" if args.resume else "w", encoding="utf-8") as out:
        for index, query in enumerate(queries, start=1):
            qid = query["query_id"]
            if qid in completed:
                continue
            category = query["_category_file"]
            cutoff = int(query.get("video_cutoff_idx", -1))
            key = (cutoff, args.caption_mode)
            if key not in context_cache:
                context_cache[key] = caption_context(captions, cutoff, args.caption_mode)
            context, n_videos, n_segments = context_cache[key]
            prompt = prompt_for(context, query["question"])
            prompt_tokens = len(tokenizer.encode(prompt, add_special_tokens=False))
            row: dict[str, Any] = {
                "query_id": qid,
                "category": category,
                "difficulty": query.get("difficulty"),
                "video_cutoff_idx": cutoff,
                "expects_episodic": query.get("expects_episodic"),
                "question": query["question"],
                "ground_truth": query.get("ground_truth", ""),
                "context_mode": args.caption_mode,
                "n_visible_videos": n_videos,
                "n_visible_segments": n_segments,
                "prompt_tokens_estimate": prompt_tokens,
            }
            started = time.perf_counter()
            try:
                if prompt_tokens > args.max_context_tokens:
                    raise ValueError(
                        f"prompt estimate {prompt_tokens} exceeds "
                        f"--max-context-tokens {args.max_context_tokens}"
                    )
                answer, encoded_tokens = generate(
                    torch, processor, model, prompt, args.device, args.max_new_tokens
                )
                row.update(
                    status="ok",
                    model_answer=answer,
                    prompt_tokens=encoded_tokens,
                    score=score_row(answer, query.get("ground_truth", "")),
                )
            except Exception as exc:  # retain per-question failures for audit
                row.update(
                    status="error",
                    error=f"{type(exc).__name__}: {exc}",
                    model_answer="",
                    score={"exact_normalized": False, "gold_substring": False, "token_f1": 0.0},
                )
            row["latency_sec"] = round(time.perf_counter() - started, 3)
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
            out.flush()
            rows.append(row)
            print(
                f"[{index}/{len(queries)}] {qid} {row['status']} "
                f"tokens={row.get('prompt_tokens', prompt_tokens)} "
                f"answer={row.get('model_answer', '')[:180]}",
                flush=True,
            )
    payload = {
        "schema": "homesentinel.caption-eval/1",
        "model_path": args.model_path,
        "caption_mode": args.caption_mode,
        "n_queries": len(queries),
        "n_completed_this_process": len(rows),
        "aggregate": aggregate(rows),
        "predictions_jsonl": str(args.output),
        "note": (
            "Heuristic lexical scores are diagnostics for open-ended answers, "
            "not a validated semantic judge."
        ),
    }
    summary_path = args.output.with_name("summary.json")
    summary_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0 if all(row["status"] == "ok" for row in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
