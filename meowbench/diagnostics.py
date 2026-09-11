"""Descriptive output diagnostics, separate from factual correctness scoring."""
from __future__ import annotations

from collections import Counter
import re


def output_diagnostics(text: str | None) -> dict:
    raw = (text or "").strip()
    tokens = re.findall(r"[a-z0-9]+(?:'[a-z]+)?|[\u4e00-\u9fff]", raw.lower())
    grams = [tuple(tokens[i:i + 4]) for i in range(max(len(tokens) - 3, 0))]
    counts = Counter(grams)
    ratio = 1 - len(counts) / len(grams) if grams else 0.0
    repeated = len(tokens) >= 20 and ratio >= 0.4 and max(counts.values(), default=0) >= 3
    lower = raw.lower().replace("’", "'")
    refusal = any(p in lower for p in (
        "don't have access", "do not have access", "don't have the video",
        "cannot access", "can't access", "can't provide information", "cannot view",
        "can't see", "cannot see", "don't have any visual", "don't have information",
        "can't watch", "cannot watch", "无法查看", "无法访问",
        "看不到视频", "没有视频", "未提供视频", "没有提供视频",
    ))
    return {"tokens_approx": len(tokens), "repeated_4gram_fraction": round(ratio, 4),
            "repetitive": repeated, "refusal_heuristic": refusal, "empty": not raw}


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)
