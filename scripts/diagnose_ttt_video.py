"""Readout diagnostics for one saved video memory; video QA has no automatic gold.

The fixed probes are never passed to test-time training. Arithmetic/copy probes
check instruction preservation separately. The visual arm explicitly sees pixels.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

PROBES = [
    ("overview", "What objects were visible, and where were they?", None),
    ("paraphrase", "Describe the visible objects and their locations in the recorded scene.", None),
    ("objects", "Name the main objects seen in the video. Answer briefly.", None),
    ("action", "What did the camera wearer do with their hands?", None),
    ("appearance", "Describe the color and appearance of the object being handled.", None),
    ("sequence", "Describe the sequence of visible actions. Do not infer unseen actions.", None),
    ("location", "Where was the handled object at the end of the video?", None),
    ("people", "Were any people other than the camera wearer visible?", None),
    ("training_template_1", "What was observed in session 1, segment 1?", None),
    ("training_template_2", "What was observed in session 1, segment 2?", None),
    ("training_template_3", "What was observed in session 1, segment 3?", None),
    ("arithmetic_control", "What is 2 + 3? Answer with the single number only.", "5"),
    ("copy_control", "Copy exactly these two words and nothing else: blue triangle", "blue triangle"),
]


def main():
    from ttt_frame.video import iter_video_chunks
    from ttt_frame.videoqa import OBSERVE_PROMPT, VideoTTTConfig, VideoTTTMemory

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", required=True)
    parser.add_argument("--memory", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--retrain-steps", type=int)
    parser.add_argument("--learning-rate", type=float)
    args = parser.parse_args()
    if not Path(args.video).is_file():
        parser.error("video missing")
    saved = json.loads((args.memory / "memory.json").read_text(encoding="utf-8"))
    config = saved["config"].copy()
    config.update(device="cuda:0", local_files_only=True)
    if args.retrain_steps is not None:
        config["steps_per_chunk"] = args.retrain_steps
    if args.learning_rate is not None:
        if args.retrain_steps is None:
            parser.error("--learning-rate requires --retrain-steps")
        config["learning_rate"] = args.learning_rate
    cfg = VideoTTTConfig(**config)
    args.out.mkdir(parents=True, exist_ok=False)
    model = VideoTTTMemory(cfg, trace_file=args.out / "teacher_trace.jsonl")
    if args.retrain_steps is None:
        model.load_memory(args.memory)
    else:
        model.ingest_video(args.video)
        model.finish_ingest()
        model.save(args.out / "memory")
    metadata = {"config": asdict(cfg), "source_memory": str(args.memory),
                "source_video": args.video, "retrained": args.retrain_steps is not None,
                "stats": model.finish_ingest(), "video_gold_available": False,
                "note": "Only arithmetic/copy probes have gold; video questions are qualitative."}
    (args.out / "diagnostic.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    frames, observations = [], []
    for chunk in iter_video_chunks(args.video, chunk_seconds=cfg.chunk_seconds,
                                   frames_per_chunk=cfg.frames_per_chunk, max_side=cfg.max_side):
        frames.extend(chunk.images)
        if len(frames) > 64:
            raise ValueError("one-video diagnosis is limited to 64 sampled frames; select a short clip")
        observations.append({"segment": chunk.index + 1, "timestamps": chunk.timestamps,
                             "text": model.answer_with_images(
                                 OBSERVE_PROMPT.format(timestamps=chunk.timestamps), chunk.images)})
    (args.out / "observations.json").write_text(json.dumps(observations, indent=2), encoding="utf-8")
    with (args.out / "probes.jsonl").open("w", encoding="utf-8") as sink:
        for probe_id, question, gold in PROBES:
            for arm in ("memory", "base-read", "visual"):
                began = time.perf_counter()
                answer = (model.answer_with_images(question, frames) if arm == "visual" else
                          model.answer(question, use_memory=arm == "memory"))
                row = dict(probe_id=probe_id, arm=arm, question=question, answer=answer,
                           expected_control=gold,
                           control_correct=answer.strip() == gold if gold is not None else None,
                           latency_ms=1000 * (time.perf_counter() - began))
                sink.write(json.dumps(row, ensure_ascii=False) + "\n")
                sink.flush()
                print(f"{probe_id} {arm}: {answer[:120]}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
