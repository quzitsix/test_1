"""Adapter for a LaCT test-time-training memory over a frozen vision encoder.

WHAT THIS IS FOR

The scientific claim under test is narrow and worth stating precisely: *do
fast weights retain a person-object binding across a change of scene?* Answering
it needs an arm in which the memory is genuinely parametric — no notes, no
retrieval index, no frames kept — so that whatever survives the video revocation
survives *in weights*.

`hf_vlm` cannot be that arm. In `memory` mode it writes natural-language notes
during ingestion and reads them back at query time, which is retrieval wearing a
VLM's clothes; its own docstring says a state-carrying architecture belongs in a
separate adapter. This is that adapter.

THE ARCHITECTURE, AND WHY IT IS SHAPED THIS WAY

    frames --(frozen encoder)--> embeddings --(LaCT write)--> fast weights
                                                                   |
                                          question --(read)--> retrieved vector
                                                                   |
                                              nearest option in the same space

Ingestion folds each session's frame embeddings into a fixed-size SwiGLU fast
weight by one large-chunk gradient step on `-f_W(k)ᵀv` (Zhang et al., 2505.23884;
the memory lives in the `ttt-frame` package). Session boundaries are the
chunk boundaries, which is LaCT's own advice — align the chunk with the data's
structure rather than a token count — and here it means the memory updates once
per visit to the home.

At query time the video is gone. The question is encoded by the *same* frozen
encoder, used as a query into the fast weights, and the returned vector is
matched against the encoded answer options. Nothing else is retained: no frames,
no embeddings, no text. That is the point.

WHY A FROZEN ENCODER AND NOT A FINE-TUNED VLM

Two reasons, one scientific and one practical.

Scientific: with the encoder frozen and the projections fixed at a seeded random
init, the *only* thing that can carry information from ingestion to query is the
fast weight itself. If a binding survives, it survived in weight space. A trained
system would leave open the objection that the answer came from the readout head
having learned the task.

Practical: it fits anywhere. A CLIP-class encoder in fp16 is well under 1 GB, so
the same code runs on a laptop for development and on a multi-GPU server for the
real sweep, and iteration costs seconds rather than GPU-hours.

The cost is real and should be stated: this measures the *mechanism*, not a
competitive system. It will not beat `hf_vlm`'s note-taking arm on the probe
suite, and it is not meant to. It answers whether the substrate can hold a
binding at all — a question that has to be settled before spending GPU-months on
a full VLM integration.

READ THE `--memory` FLAG AS THE EXPERIMENT'S INDEPENDENT VARIABLE

`--memory lact` is the treatment. The two controls exist because a failure in the
treatment arm is uninterpretable on its own:

  `--memory none`     ignores ingestion entirely; scores the encoder's priors.
                      Anything the treatment scores above this came from memory.
  `--memory mean`     replaces the fast weights with a running mean of the same
                      embeddings — the simplest possible fixed-size memory.
                      If `lact` cannot beat `mean`, its update rule is buying
                      nothing, and that is a finding rather than a bug.

Running all three on one fixture is what turns "TTT failed" into "TTT failed
*and here is the substrate that did not*".
"""

from __future__ import annotations

import logging
import sys
from typing import Any

import torch
import torch.nn.functional as F

from meowbench.adapters.base import (
    AdapterBase,
    common_args,
    configure_logging,
)
from meowbench.media import sample_frames

# The memory itself lives in a separate repo (github.com/quzitsix/TTT_frame) so
# that the TTT research and the benchmark harness can evolve independently: the
# harness must stay model-agnostic, and a fast-weight implementation is exactly
# the kind of system-under-test it is not supposed to know about. This adapter is
# the bridge, and the only place the two touch.
#     pip install -e path/to/TTT_frame
try:
    from ttt_frame.lact import LaCTMemory
except ImportError as exc:  # pragma: no cover - depends on the optional install
    raise SystemExit(
        "this adapter needs the ttt-frame package:\n"
        "    git clone git@github.com:quzitsix/TTT_frame.git\n"
        "    pip install -e TTT_frame\n"
        f"(import failed: {exc})"
    ) from exc

logger = logging.getLogger(__name__)

#: Query template. Deliberately plain: any cleverness here would be prompt
#: engineering leaking into a measurement about memory.
QUERY_TEMPLATE = "{question}"


class TTTAdapter(AdapterBase):
    def __init__(
        self,
        *,
        model_path: str,
        context_mode: str = "memory",
        memory: str = "lact",
        system_id: str | None = None,
        n_frames: int = 8,
        head_dim: int = 64,
        base_lr: float = 0.05,
        use_muon: bool = True,
        inter_multi: float = 1.0,
        device: str | None = None,
        dtype: str = "float32",
        seed: int = 0,
    ) -> None:
        super().__init__(system_id or f"ttt:{memory}:{model_path.split('/')[-1]}")
        self.context_mode = context_mode
        self.memory_kind = memory
        self._n_frames = n_frames

        from transformers import AutoModel, AutoProcessor

        self._device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        # fp32 by default: the fast-weight update does a Newton-Schulz iteration
        # and an L2 renormalisation per chunk, both of which lose meaningful
        # precision in fp16 on a state that is written hundreds of times. The
        # encoder is small enough that this costs little.
        self._dtype = {"float32": torch.float32, "float16": torch.float16,
                       "bfloat16": torch.bfloat16}[dtype]

        logger.info("loading encoder %s onto %s", model_path, self._device)
        self._processor = AutoProcessor.from_pretrained(model_path)
        self._model = AutoModel.from_pretrained(model_path, dtype=self._dtype)
        self._model.eval().to(self._device)

        self._dim = self._infer_width()
        logger.info("encoder ready (joint embedding width=%d)", self._dim)

        self._mem = LaCTMemory(
            dim=self._dim,
            head_dim=head_dim,
            inter_multi=inter_multi,
            base_lr=base_lr,
            use_muon=use_muon,
            learn_projections=False,
            seed=seed,
        ).to(device=self._device, dtype=self._dtype)

        #: The `mean` control's state. Kept separate so the two substrates never
        #: interact and a run can be attributed to exactly one of them.
        self._running: torch.Tensor | None = None
        self._n_chunks = 0

    def _infer_width(self) -> int:
        """The shared image/text embedding width.

        A CLIP-class model exposes `projection_dim`; the fallback probes the
        text tower, because guessing wrong here surfaces much later as a shape
        error inside the memory rather than as a clear failure now.
        """
        for attr in ("projection_dim", "hidden_size"):
            value = getattr(self._model.config, attr, None)
            if isinstance(value, int):
                return value
        text_config = getattr(self._model.config, "text_config", None)
        value = getattr(text_config, "hidden_size", None)
        if isinstance(value, int):
            return value
        raise RuntimeError(
            f"could not determine the embedding width of {type(self._model).__name__}; "
            "this adapter expects a CLIP/SigLIP-style dual encoder"
        )

    # -- encoding ------------------------------------------------------------

    @torch.no_grad()
    def _encode_images(self, images: list[Any]) -> torch.Tensor:
        inputs = self._processor(images=images, return_tensors="pt")
        inputs = {k: v.to(self._device) for k, v in inputs.items()}
        if "pixel_values" in inputs:
            inputs["pixel_values"] = inputs["pixel_values"].to(self._dtype)
        feats = self._model.get_image_features(**inputs)
        return F.normalize(feats.to(self._dtype), dim=-1)

    @torch.no_grad()
    def _encode_texts(self, texts: list[str]) -> torch.Tensor:
        inputs = self._processor(
            text=texts, return_tensors="pt", padding=True, truncation=True
        )
        inputs = {k: v.to(self._device) for k, v in inputs.items()}
        feats = self._model.get_text_features(**inputs)
        return F.normalize(feats.to(self._dtype), dim=-1)

    # -- lifecycle -----------------------------------------------------------

    def on_env_begin(self, env_id: str, n_sessions: int) -> None:
        # A household must not inherit the previous one's memory; that would
        # read as recall while actually being cross-environment leakage.
        self._mem.reset()
        self._running = None
        self._n_chunks = 0

    def ingest(self, msg: dict[str, Any]) -> dict[str, Any]:
        path = msg.get("video_path")
        if not path:
            return {"frames": 0}
        try:
            return self._ingest_session(msg, path)
        except Exception as exc:  # noqa: BLE001 - one bad clip must not end the run
            logger.exception("ingest of %s failed; continuing", msg.get("session_id"))
            return {"frames": 0, "error": f"{type(exc).__name__}: {exc}"}

    def _ingest_session(self, msg: dict[str, Any], path: str) -> dict[str, Any]:
        frames = sample_frames(path, n_frames=self._n_frames, max_side=384)
        if not frames:
            logger.warning("no frames decoded from %s", path)
            return {"frames": 0}

        # `sample_frames` returns decoded PIL images, so the file handle is
        # already closed here. Nothing below touches the path again — the
        # harness truncates it at ingest_end and a retained handle would be
        # latched as `revocation_contested`.
        embeddings = self._encode_images([f.image for f in frames])

        stats: dict[str, Any] = {"frames": len(frames)}
        if self.memory_kind == "lact":
            stats.update(self._mem.write(embeddings))
        elif self.memory_kind == "mean":
            chunk = embeddings.mean(dim=0, keepdim=True)
            self._running = chunk if self._running is None else self._running + chunk
            self._n_chunks += 1
            stats["state_norm_after"] = float(self._running.norm())
        elif self.memory_kind == "none":
            stats["ignored"] = True
        else:
            raise ValueError(f"unknown memory kind {self.memory_kind!r}")
        return stats

    def on_ingest_end(self) -> dict[str, Any]:
        if self.memory_kind == "lact":
            return {
                "n_records": self._mem.n_writes,
                "memory_bytes": self._mem.memory_bytes(),
                "state_norm": self._mem.state_norm(),
            }
        if self.memory_kind == "mean":
            return {
                "n_records": self._n_chunks,
                "memory_bytes": 0 if self._running is None else
                self._running.numel() * self._running.element_size(),
            }
        return {"n_records": 0, "memory_bytes": 0}

    # -- answering -----------------------------------------------------------

    def answer(self, msg: dict[str, Any]) -> dict[str, Any]:
        """Score each option against what the memory returns for the question.

        This is a *forced-choice readout*, not generation: the adapter has no
        language model, so it cannot produce prose. It encodes the question,
        reads the memory with it, and picks the option whose text embedding is
        closest to the result.
        """
        options: dict[str, str] = msg.get("options") or {}
        if msg.get("answer_format") != "mcq5" or not options:
            # No language model here, so free-form and numeric items cannot be
            # attempted. Return an empty answer in the field the format expects
            # rather than an explanatory extra key: `AnswerMsg` forbids unknown
            # fields, and a malformed reply would be recorded as a protocol
            # error against the whole item instead of a blank answer.
            if msg.get("answer_format") == "open":
                return {"answer_text": "", "raw": ""}
            return {"answer": "", "raw": ""}

        question = QUERY_TEMPLATE.format(question=msg["question"])
        q_vec = self._encode_texts([question])            # [1, dim]
        letters = sorted(options)
        opt_vecs = self._encode_texts([options[letter] for letter in letters])

        if self.memory_kind == "lact":
            retrieved = self._mem.read(q_vec)
        elif self.memory_kind == "mean" and self._running is not None:
            retrieved = self._running / max(self._n_chunks, 1)
        else:
            retrieved = torch.zeros_like(q_vec)

        # Score options by what the MEMORY returns, with each option's
        # question-independent affinity removed.
        #
        # The calibration is not optional. CLIP's text similarity is strongly
        # length- and phrasing-biased: measured here, the option-E sentence
        # ("The information is not available based on the given context")
        # out-scores every bare name for *every* question, including
        # nonsensical ones. Uncalibrated, all three memory arms therefore
        # answered E on all 14 items and produced byte-identical reports — a
        # readout artifact indistinguishable from "memory does nothing".
        #
        # Subtracting the affinity of an empty-memory probe removes exactly the
        # part of the score that does not depend on what was ingested, so what
        # remains is the memory's contribution. `none` is then flat by
        # construction, which is the correct floor for this control.
        mem_scores = (F.normalize(retrieved, dim=-1) @ opt_vecs.T).squeeze(0)
        base_scores = (F.normalize(q_vec, dim=-1) @ opt_vecs.T).squeeze(0)
        scores = mem_scores - base_scores

        if self.memory_kind == "none" or float(retrieved.abs().sum()) == 0.0:
            # With no memory every option's calibrated score is identical, and
            # argmax would silently return the first letter for every item —
            # a constant-A system that looks like a decision. Abstain instead,
            # which is the honest answer for a system that ingested nothing and
            # is scored correct only on the unanswerable controls.
            choice = "E" if "E" in options else letters[0]
            return {"answer": choice, "raw": f"{choice}  (no memory; abstained)"}

        choice = letters[int(scores.argmax())]

        detail = " ".join(
            f"{letter}={float(score):+.4f}" for letter, score in zip(letters, scores)
        )
        return {"answer": choice, "raw": f"{choice}  ({detail})"}


class LoRAVideoAdapter(AdapterBase):
    """Thin protocol bridge; all perception/training lives in ttt-frame.

    --backend lora is a separate generative self-distillation baseline, not an
    alternative name for the CLIP/LaCT mechanism above. Gold items are never
    passed to the learner. The harness retains responsibility for video revocation.
    """

    def __init__(self, config, *, context_mode="memory", system_id=None,
                 read_base=False, metrics_path=None, engine=None):
        if context_mode not in {"memory", "blind"}:
            raise ValueError("LoRA supports memory/blind; use hf_vlm for the video oracle")
        super().__init__(system_id or f"video-lora-{'base-read' if read_base else context_mode}")
        self.context_mode = context_mode
        self.read_base = read_base
        self.metrics_path = metrics_path
        self._env_id = None
        if engine is None:
            try:
                from ttt_frame.videoqa import VideoTTTMemory
                engine = VideoTTTMemory(config)
            except ImportError as exc:
                raise ImportError('install the sibling repo: pip install -e "../TTT_frame[video]"') from exc
        self.engine = engine

    def capabilities(self):
        return {"context_mode": self.context_mode, "accepts": ["video_path"],
                "memory_kind": "lora_self_distillation", "read_base": self.read_base}

    def on_env_begin(self, env_id, n_sessions):
        self.engine.reset()
        self._env_id = env_id

    def ingest(self, msg):
        if self.context_mode == "blind":
            return {"frames": 0, "optimizer_steps": 0}
        path = msg.get("video_path")
        if not path:
            raise ValueError("LoRA video memory requires a video_path during ingestion")
        # Only media enters the learner: not item_ids, questions, options or gold.
        return self.engine.ingest_video(path)

    def on_ingest_end(self):
        summary = self.engine.finish_ingest()
        if self.metrics_path:
            import json
            from pathlib import Path

            target = Path(self.metrics_path)
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps({"env_id": self._env_id, **summary}) + "\n")
        return summary

    def answer(self, msg):
        import time
        from meowbench.adapters.base import build_prompt, parse_reply

        started = time.perf_counter()
        answer = self.engine.answer(build_prompt(msg), use_memory=not self.read_base)
        return {**parse_reply(msg, answer),
                "latency_ms": (time.perf_counter() - started) * 1000}

    def on_env_end(self):
        self.engine.reset()


def _lora_main(argv):
    from ttt_frame.videoqa import add_video_arguments, config_from_args

    parser = common_args("Video self-distillation into LoRA, followed by parameter-only QA")
    parser.add_argument("--backend", choices=("lora",), default="lora")
    parser.add_argument("--read-base", action="store_true",
                        help="same ingestion/training budget, but disable LoRA during answering")
    parser.add_argument("--metrics-path", help="optional JSONL of numeric ingestion diagnostics")
    add_video_arguments(parser)
    args = parser.parse_args(argv)
    configure_logging(args.log_level)
    return LoRAVideoAdapter(
        config_from_args(args), context_mode=args.context_mode, system_id=args.system_id,
        read_base=args.read_base, metrics_path=args.metrics_path,
    ).run()


def main(argv: list[str] | None = None) -> int:
    import argparse

    selector = argparse.ArgumentParser(add_help=False)
    selector.add_argument("--backend", choices=("lact", "lora"), default="lact")
    selected, _ = selector.parse_known_args(argv)
    if selected.backend == "lora":
        return _lora_main(argv)
    parser = common_args(__doc__ or "")
    parser.add_argument("--backend", choices=("lact", "lora"), default="lact",
                        help="lora enables generative VideoQA; use --backend lora --help")
    parser.add_argument(
        "--model-path",
        default="openai/clip-vit-base-patch32",
        help="a CLIP/SigLIP-style dual encoder; must expose get_image_features "
        "and get_text_features",
    )
    parser.add_argument(
        "--memory",
        choices=("lact", "mean", "none"),
        default="lact",
        help="lact = the fast-weight treatment; mean = running-mean control; "
        "none = ignore ingestion entirely (prior-only floor)",
    )
    parser.add_argument("--n-frames", type=int, default=8)
    parser.add_argument("--head-dim", type=int, default=64)
    parser.add_argument("--base-lr", type=float, default=0.05)
    parser.add_argument("--inter-multi", type=float, default=1.0)
    parser.add_argument(
        "--no-muon",
        action="store_true",
        help="use plain gradient descent instead of Newton-Schulz orthogonalisation",
    )
    parser.add_argument("--device", default=None)
    parser.add_argument("--dtype", default="float32",
                        choices=("float32", "float16", "bfloat16"))
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    configure_logging(args.log_level)

    adapter = TTTAdapter(
        model_path=args.model_path,
        context_mode=args.context_mode,
        memory=args.memory,
        system_id=args.system_id,
        n_frames=args.n_frames,
        head_dim=args.head_dim,
        base_lr=args.base_lr,
        inter_multi=args.inter_multi,
        use_muon=not args.no_muon,
        device=args.device,
        dtype=args.dtype,
        seed=args.seed,
    )
    return adapter.run()


if __name__ == "__main__":
    sys.exit(main())
