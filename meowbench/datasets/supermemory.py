"""Import SuperMemory-VQA without rewriting its questions, choices or answer keys.

Only pure visual, answerable questions with complete, strictly prior annotated
answer evidence enter v1. single-session is an explicit reduced-context pilot;
history includes every known prior recording listed by the original example.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

from meowbench.schema import (AnswerFormat, Audit, Certificate, EnvManifest, Evidence,
    EvidenceScope, GtExactness, Item, Provenance, SessionRef, Span)
from meowbench.suite import file_sha256, write_suite

SOURCE = "https://huggingface.co/datasets/OSU-AIoT-MLSys-Lab/SuperMemory-VQA"
REPO_ID = "OSU-AIoT-MLSys-Lab/SuperMemory-VQA"
PLAN_SCHEMA = "meowbench.supermemory-plan/1"
LICENSE = "CC-BY-NC-SA-4.0"
UNANSWERABLE = "This question can not be answered."
VIDEO_ID = re.compile(r"Person_\d+_session_\d+_[A-Za-z0-9_]+\Z")


def digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     allow_nan=False).encode()).hexdigest()


def load_annotations(path: Path) -> tuple[Path, list[dict]]:
    if path.is_dir():
        matches = sorted(path.rglob("all_qa.json"))
        if len(matches) != 1:
            raise ValueError(f"Expected one all_qa.json under {path}, found {len(matches)}; pass a file explicitly")
        path = matches[0]
    data = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(data, list) or not data or not all(isinstance(r, dict) for r in data):
        raise ValueError("Expected the official list of QA objects (all_qa.json or qa_person_N.json)")
    ids = [r.get("question_id") for r in data]
    if len(set(ids)) != len(ids) or None in ids:
        raise ValueError("Missing or duplicated question_id; do not concatenate all_qa and per-person copies")
    return path, data


def number(value: object) -> float:
    if isinstance(value, bool):
        raise ValueError("Boolean timestamp")
    v = float(value)
    if not math.isfinite(v) or v < 0:
        raise ValueError("Invalid timestamp")
    return v


def video_id(value: object) -> str:
    if not isinstance(value, str) or not VIDEO_ID.fullmatch(value):
        raise ValueError(f"Invalid official video_id: {value!r}")
    return value


def window(span: dict) -> tuple[float, float]:
    start, end = number(span["start_time"]), number(span["end_time"])
    if end <= start:
        raise ValueError("Empty or reversed time span")
    return start, end


def make_plan(path: Path, *, context: str = "single-session", limit: int = 8,
              max_videos: int = 2, max_current_seconds: float = 1200) -> dict:
    if context not in {"single-session", "history"}:
        raise ValueError("Unknown context")
    if limit < 1 or max_videos < 1 or not math.isfinite(max_current_seconds) or max_current_seconds <= 0:
        raise ValueError("Selection limits must be positive")
    path, rows = load_annotations(path)
    starts: dict[str, set[float]] = defaultdict(set)
    for row in rows:
        meta = row.get("metadata") or {}
        pairs = [(meta.get("primary_video_id"), meta.get("primary_video_start_time"))]
        for e in (row.get("answer_evidence") or {}).get("evidence_list", []):
            pairs.append((e.get("video_id"), e.get("start_time")))
        for vid, stamp in pairs:
            if vid is not None and stamp is not None:
                starts[video_id(vid)].add(number(stamp))
    known = {vid: next(iter(values)) for vid, values in starts.items() if len(values) == 1}
    counts: Counter = Counter()
    selected: list[dict] = []
    used: set[str] = set()
    rejected: list[dict] = []
    for row in rows:
        reason = ""
        try:
            choices, idx = row["choices"], row["correct_option_index"]
            if (not isinstance(choices, list) or len(choices) != 4 or
                any(not isinstance(x, str) or not x.strip() for x in choices) or
                type(idx) is not int or not 0 <= idx < 4 or
                choices[idx] != row["correct_answer"] or
                row["choice_types"][idx] != "correct" or choices.count(UNANSWERABLE) != 1):
                raise ValueError("Inconsistent original choice/answer fields")
            if row.get("is_answerable") is not True:
                reason = "unanswerable_not_in_visual_pilot"
            q = row["question_evidence"]
            ev = row["answer_evidence"]["evidence_list"]
            if not ev or not q.get("time_spans"):
                reason = reason or "missing_evidence"
            mods = [set(e.get("modalities", [])) for e in [q, *ev]]
            if any(not m or not m <= {"Video", "OCR"} for m in mods):
                reason = reason or "requires_nonvisual_or_unknown_modality"
            primary = video_id(row["metadata"]["primary_video_id"])
            if any(s.get("video_id", q.get("video_id")) != primary for s in q["time_spans"]):
                reason = reason or "ambiguous_query_recording"
            cutoff = min(window(s)[0] for s in q["time_spans"])
            if not 0 < cutoff <= max_current_seconds:
                reason = reason or "query_prefix_outside_budget"
            if primary not in known:
                reason = reason or "unknown_or_conflicting_recording_time"
            if reason:
                counts[reason] += 1
                continue
            boundary = known[primary] + cutoff
            for e in ev:
                vid = video_id(e["video_id"])
                _, end = window(e["time_span"])
                if vid not in known or known[vid] != number(e["start_time"]):
                    reason = "unknown_or_conflicting_recording_time"
                    break
                if known[vid] + end > boundary:
                    reason = "answer_evidence_after_query_boundary"
                    break
                if context == "single-session" and vid != primary:
                    reason = "requires_cross_session_context"
                    break
            if reason:
                counts[reason] += 1
                continue
            if context == "single-session":
                history = [primary]
            else:
                listed = set(map(video_id, row["video_ids"])) | {primary}
                if any(v not in known for v in listed):
                    counts["history_has_unknown_recording_times"] += 1
                    continue
                history = sorted((v for v in listed if known[v] < boundary), key=lambda v: (known[v], v))
                if any(e["video_id"] not in history for e in ev):
                    counts["evidence_not_in_official_history"] += 1
                    continue
            if len(used | set(history)) > max_videos:
                counts["video_budget"] += 1
                continue
            if len(selected) >= limit:
                counts["item_limit"] += 1
                continue
            selected.append({"source": row, "query_boundary_unix": boundary,
                             "recordings": [{"video_id": v, "start_unix": known[v],
                                "end_sec": boundary - known[v]} for v in history]})
            used.update(history)
        except (ValueError, KeyError, TypeError, IndexError) as exc:
            counts["invalid_source_row"] += 1
            rejected.append({"question_id": row.get("question_id"), "error": str(exc)})
    result = {"schema": PLAN_SCHEMA, "source": SOURCE, "source_file": path.name,
              "source_sha256": file_sha256(path), "license": LICENSE,
              "context": context, "query_boundary": "start of first question_evidence time span",
              "selection": {"limit": limit, "max_videos": max_videos,
                            "max_current_seconds": max_current_seconds,
                            "modalities": ["Video", "OCR"], "answerable_only": True},
              "counts": {"source_rows": len(rows), "selected": len(selected),
                         "excluded": dict(counts)}, "invalid_rows": rejected,
              "videos": [{"video_id": v, "hf_path": hf_path(v)} for v in sorted(used)],
              "examples": selected}
    result["plan_sha256"] = digest(result)
    return result


def hf_path(vid: str) -> str:
    video_id(vid)
    person = "_".join(vid.split("_")[:2])
    return f"data/video/{person}/{vid}.mp4"


def read_plan(path: Path) -> dict:
    p = json.loads(path.read_text(encoding="utf-8"))
    check = {k: v for k, v in p.items() if k != "plan_sha256"}
    if p.get("schema") != PLAN_SCHEMA or p.get("plan_sha256") != digest(check):
        raise ValueError("Plan schema/hash mismatch; regenerate the plan instead of editing it")
    if not p["examples"]:
        raise ValueError("Plan has zero questions; inspect counts and increase the selection budget")
    return p


def find_videos(root: Path, ids: list[str]) -> dict[str, Path]:
    wanted = set(map(video_id, ids))
    matches: dict[str, list[Path]] = defaultdict(list)
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() == ".mp4" and p.stem in wanted:
            matches[p.stem].append(p.resolve())
    ambiguous = [v for v, paths in matches.items() if len(set(paths)) > 1]
    if ambiguous:
        raise ValueError(f"Duplicate video basenames; choose a narrower --video-root: {ambiguous}")
    return {v: paths[0] for v, paths in matches.items()}


def prepare_suite(plan: dict, video_root: Path, out: Path, *, chunk_seconds: float = 60,
                  sample_fps: int = 2, max_side: int = 768) -> object:
    """Physically limit media before staging. No future video path reaches an adapter."""
    from meowbench.media import probe_duration
    from meowbench.datasets.video_windows import render_window
    if chunk_seconds <= 0 or sample_fps < 1 or max_side < 64:
        raise ValueError("Invalid video preparation settings")
    if out.exists():
        raise FileExistsError(f"Output already exists: {out}. Use a new release directory.")
    paths = find_videos(video_root, [v["video_id"] for v in plan["videos"]])
    missing = sorted(set(v["video_id"] for v in plan["videos"]) - set(paths))
    if missing:
        raise FileNotFoundError("Missing videos (run fetch, or correct --video-root): " + ", ".join(missing))
    durations = {v: probe_duration(p) for v, p in paths.items()}
    # Validate ALL source spans before spending time preparing media.
    for ex in plan["examples"]:
        r = ex["source"]
        for e in r["answer_evidence"]["evidence_list"]:
            if window(e["time_span"])[1] > durations[e["video_id"]] + .1:
                raise ValueError(f"Q{r['question_id']}: evidence exceeds video duration")
        primary = r["metadata"]["primary_video_id"]
        current = next(s for s in ex["recordings"] if s["video_id"] == primary)
        if current["end_sec"] > durations[primary] + .1:
            raise ValueError(f"Q{r['question_id']}: query boundary exceeds video duration")
    out.mkdir(parents=True)
    items, envs, media_index = [], {}, {}
    manifest_media = {}
    for ex in plan["examples"]:
        row = ex["source"]
        recordings = [{**s, "end_sec": min(s["end_sec"], durations[s["video_id"]])}
                      for s in ex["recordings"]]
        env_id = "sm-" + digest(recordings)[:20]
        if env_id not in envs:
            print(f"Preparing Q{row['question_id']}: {len(recordings)} recording(s), "
                  f"{sum(s['end_sec'] for s in recordings):.1f}s of context", flush=True)
            refs = []
            index = []
            for rec in recordings:
                vid, end = rec["video_id"], rec["end_sec"]
                start = 0.0
                while start < end - .001:
                    stop = min(start + chunk_seconds, end)
                    sid = "clip-" + digest([vid, start, stop, sample_fps, max_side])[:24]
                    relative = f"media/{sid}.mp4"
                    target = out / relative
                    if relative not in manifest_media:
                        render_window(paths[vid], target, start=start, end=stop,
                                      fps=sample_fps, max_side=max_side)
                        manifest_media[relative] = file_sha256(target)
                    refs.append(SessionRef(session_id=sid, order=len(refs),
                                           video_path=str(target.resolve()), duration_sec=stop-start))
                    index.append({"session_id": sid, "video_id": vid, "start_sec": start,
                                  "end_sec": stop, "recording_start_unix": rec["start_unix"]})
                    start = stop
            envs[env_id] = EnvManifest(env_id=env_id, dataset="SuperMemory-VQA", sessions=refs)
            media_index[env_id] = index
        spans, evidence_sids = [], []
        for ev in row["answer_evidence"]["evidence_list"]:
            a, b = window(ev["time_span"])
            for clip in media_index[env_id]:
                if clip["video_id"] != ev["video_id"]:
                    continue
                lo, hi = max(a, clip["start_sec"]), min(b, clip["end_sec"])
                if hi > lo:
                    spans.append(Span(session_id=clip["session_id"], start_sec=lo-clip["start_sec"],
                                      end_sec=hi-clip["start_sec"]))
                    evidence_sids.append(clip["session_id"])
        options = dict(zip("ABCD", row["choices"]))
        source_sessions = {e["video_id"] for e in row["answer_evidence"]["evidence_list"]}
        # Evidence-session count is not the number of computational chunks.
        cross = len(source_sessions) > 1
        bounds = [(number(e["start_time"])+window(e["time_span"])[0],
                   number(e["start_time"])+window(e["time_span"])[1])
                  for e in row["answer_evidence"]["evidence_list"]]
        items.append(Item(item_id=f"supermemory-{row['question_id']}", env_id=env_id,
            session_ids=[s.session_id for s in envs[env_id].ordered()],
            axis="SM_" + row["metadata"]["skill"], answer_format=AnswerFormat.MCQ,
            question=row["question"], options=options, answer="ABCD"[row["correct_option_index"]],
            answer_text=row["correct_answer"], abstention_option="ABCD"[row["choices"].index(UNANSWERABLE)],
            evidence=Evidence(session_ids=list(dict.fromkeys(evidence_sids)), spans=spans,
                source_rows=[f"{plan['source_file']}#question_id={row['question_id']}"],
                notes="Original annotated evidence; passed only to evaluator, never to model."),
            certificate=Certificate(n_sessions=len(source_sessions), cross_session=cross,
                scope=EvidenceScope.CROSS_SESSION if cross else EvidenceScope.SINGLE_SESSION,
                span_seconds=max(b for _, b in bounds)-min(a for a, _ in bounds)),
            provenance=Provenance(miner="supermemory-native/1", dataset="SuperMemory-VQA",
                                  license=LICENSE, gt_exactness=GtExactness.DERIVED),
            audit=Audit(notes="Official answer key preserved; no independent human re-audit.")))
    (out / "source_plan.json").write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "media_index.json").write_text(json.dumps(media_index, indent=2), encoding="utf-8")
    return write_suite(out, items, envs, name=out.name, extra={
        "profile": "supermemory-native-visual/1", "context_policy": plan["context"],
        "source": SOURCE, "source_sha256": plan["source_sha256"], "plan_sha256": plan["plan_sha256"],
        "license": LICENSE, "media_sha256": manifest_media,
        "preparation": {"chunk_seconds": chunk_seconds, "sample_fps": sample_fps, "max_side": max_side},
        "notes": "Real video, original QA. Visual answerable subset, not the full official benchmark. "
                 "Chunks are not separate capture sessions. Frames sampled independently of answer evidence. "
                 "No audio, transcripts, gaze or geometry are sent. Single-session removes earlier recordings."})
