"""Adapter for a local HuggingFace vision-language model.

Loads weights once in-process — no HTTP server to start, which is why this is
usually the easiest thing to get working on a cluster node where the weights are
already on disk. Tested shape covers the Qwen2.5-VL / Qwen3-VL family and any
model exposing `AutoModelForImageTextToText` + `AutoProcessor` with a chat
template that accepts interleaved image parts.

Track realisation mirrors `openai_compat`, with one real difference: because the
model is in-process we *could* hold KV cache across questions, but we do not.
Doing so would leak the ingest phase into the query phase through a channel the
harness cannot see or revoke, which is exactly the thing the two-phase protocol
exists to prevent. So the memory track writes text notes, same as the API path,
and any genuine state-carrying architecture belongs in its own adapter.
"""

from __future__ import annotations

import logging
import os
import sys
from typing import Any

from meowbench.adapters.base import (
    AdapterBase,
    build_prompt,
    common_args,
    configure_logging,
    parse_reply,
)
from meowbench.adapters.openai_compat import NOTE_PROMPT
from meowbench.media import sample_frames

logger = logging.getLogger(__name__)


class HFVLMAdapter(AdapterBase):
    def __init__(
        self,
        *,
        model_path: str,
        context_mode: str,
        system_id: str | None = None,
        n_frames: int = 8,
        max_side: int = 768,
        max_oracle_frames: int = 32,
        max_new_tokens: int = 512,
        note_max_new_tokens: int = 900,
        temperature: float = 0.0,
        dtype: str = "bfloat16",
        device_map: str = "auto",
        attn_implementation: str | None = None,
        trust_remote_code: bool = False,
    ) -> None:
        # Computed outside the f-string: backslashes inside f-string expressions
        # are a syntax error before Python 3.12 (PEP 701 relaxed it), and the
        # project supports 3.11.
        model_label = os.path.basename(model_path.rstrip("/\\")) or model_path
        super().__init__(system_id or f"hf_vlm:{model_label}")
        self.context_mode = context_mode
        self._n_frames = n_frames
        self._max_side = max_side
        self._max_oracle_frames = max_oracle_frames
        self._max_new_tokens = max_new_tokens
        self._note_max_new_tokens = note_max_new_tokens
        self._temperature = temperature

        import torch
        from transformers import AutoProcessor

        self._torch = torch
        resolved = _resolve_dtype(torch, dtype)
        logger.info("loading %s (dtype=%s, device_map=%s)", model_path, dtype, device_map)

        kwargs: dict[str, Any] = {
            "dtype": resolved,
            "device_map": device_map,
            "trust_remote_code": trust_remote_code,
        }
        if attn_implementation:
            kwargs["attn_implementation"] = attn_implementation
        self._model = _load_model(model_path, kwargs)
        self._processor = AutoProcessor.from_pretrained(
            model_path, trust_remote_code=trust_remote_code
        )
        self._model.eval()
        self._input_device = _input_device(torch, self._model)
        logger.info("model ready (inputs go to %s)", self._input_device)

        self._notes: list[str] = []
        self._session_paths: list[str] = []
        #: Oracle frames, decoded lazily on the first query and reused for the
        #: rest of the environment. None means "not yet decoded".
        self._oracle_frames: list[Any] | None = None

    # -- lifecycle -----------------------------------------------------------

    def on_env_begin(self, env_id: str, n_sessions: int) -> None:
        self._notes.clear()
        self._session_paths.clear()
        self._oracle_frames = None

    def ingest(self, msg: dict[str, Any]) -> dict[str, Any]:
        path = msg.get("video_path")
        if not path:
            return {"frames": 0, "note_chars": 0}
        if self.context_mode == "oracle":
            self._session_paths.append(path)
            return {"frames": 0, "deferred": True}

        # One unusable session must not cost the run. A corrupt video, a decode
        # error, or an OOM on a single clip is recoverable: that session simply
        # contributes nothing to memory, which the harness already surfaces as
        # `sessions_without_frames`. Letting the exception escape would exit the
        # adapter (see AdapterBase.run), and the *next* environment's questions
        # would then all be recorded as crashes — on a two-environment suite,
        # half the run lost to one bad file.
        try:
            return self._ingest_session(msg, path)
        except Exception as exc:  # noqa: BLE001 - deliberately broad; see above
            logger.exception("ingest of %s failed; continuing without it", msg["session_id"])
            return {"frames": 0, "note_chars": 0, "error": f"{type(exc).__name__}: {exc}"}

    def _ingest_session(self, msg: dict[str, Any], path: str) -> dict[str, Any]:
        frames = sample_frames(path, n_frames=self._n_frames, max_side=self._max_side)
        if not frames:
            logger.warning("no frames decoded from %s", path)
            return {"frames": 0, "note_chars": 0}

        note = self._generate(
            [f.image for f in frames], NOTE_PROMPT, max_new_tokens=self._note_max_new_tokens
        )
        self._notes.append(f"[session {msg['session_id']}]\n{note}")
        return {"frames": len(frames), "note_chars": len(note)}

    def on_ingest_end(self) -> dict[str, Any]:
        return {
            "n_records": len(self._notes),
            "memory_bytes": sum(len(n.encode("utf-8")) for n in self._notes),
        }

    def answer(self, msg: dict[str, Any]) -> dict[str, Any]:
        prompt = build_prompt(msg)
        images: list[Any] = []
        if self.context_mode == "oracle":
            images.extend(self._oracle_images())
        elif self.context_mode == "memory":
            if self._notes:
                prompt = (
                    "Here are your own notes from watching this home. The video is no "
                    "longer available; answer from these notes.\n\n"
                    + "\n\n".join(self._notes)
                    + "\n\n"
                    + prompt
                )
            else:
                # With no notes the memory prompt is byte-identical to the blind
                # one, so this track would silently measure priors while being
                # reported as memory — and Memory Gain would come out at zero
                # for a plumbing reason, not a scientific one. Say so loudly;
                # the runner also surfaces n_records in the report.
                logger.warning(
                    "memory track has no notes for this environment; answering "
                    "from priors alone, which is the blind condition"
                )
        text = self._generate(images, prompt, max_new_tokens=self._max_new_tokens)
        return parse_reply(msg, text)

    # -- generation ----------------------------------------------------------

    def _oracle_images(self) -> list[Any]:
        """Frames for every retained session, decoded once per environment.

        Oracle media is never revoked, so the frames cannot change mid-run and
        re-decoding them per question is pure waste: on a 3-session environment
        with 28 questions at 8 frames that is 672 decodes instead of 24. The
        cache is cleared in `on_env_begin`, so no environment can see another's
        frames.

        The total is capped, because oracle attaches every session's frames to
        *one* prompt: cost grows as sessions x frames, so an 8-session
        environment at 8 frames is ~19k visual tokens before any text. Left
        uncapped this OOMs mid-run on the track that defines the ceiling. When
        the cap bites, frames are thinned evenly across the whole environment
        rather than truncated, so late sessions are still represented — dropping
        the tail would quietly turn the ceiling into "an early-sessions
        baseline" — and the reduction is logged, since a silently thinned
        oracle would understate the headroom it exists to measure.
        """
        if self._oracle_frames is None:
            frames: list[Any] = []
            for path in self._session_paths:
                frames.extend(
                    f.image
                    for f in sample_frames(
                        path, n_frames=self._n_frames, max_side=self._max_side
                    )
                )
            logger.info("oracle: cached %d frame(s) from %d session(s)",
                        len(frames), len(self._session_paths))
            if len(frames) > self._max_oracle_frames:
                kept = _thin_evenly(frames, self._max_oracle_frames)
                logger.warning(
                    "oracle: %d frame(s) exceeds --max-oracle-frames=%d; thinning "
                    "evenly to %d. The ceiling is measured on a subsample, so it "
                    "understates the true headroom.",
                    len(frames), self._max_oracle_frames, len(kept),
                )
                frames = kept
            self._oracle_frames = frames
        return self._oracle_frames

    def _generate(self, images: list[Any], prompt: str, *, max_new_tokens: int) -> str:
        # The PIL object goes under the "image" key, not just {"type": "image"}.
        # transformers' fused apply_chat_template collects visuals by looking for
        # the keys ("image", "url", "path", "base64") on each content part
        # (processing_utils.py, the image_fnames comprehension). A bare
        # {"type": "image"} matches none of them, so it computes
        # `images_exist = False` and calls the processor with images=None: the
        # prompt keeps its image placeholder tokens while no pixel_values are
        # produced, and the model answers *without ever seeing the video*.
        # Measured directly against the installed 4.57.6 source.
        content: list[dict[str, Any]] = [
            {"type": "image", "image": image} for image in images
        ]
        content.append({"type": "text", "text": prompt})
        messages = [{"role": "user", "content": content}]

        inputs = self._encode(messages, images)
        _assert_images_reached_the_model(inputs, len(images))
        inputs = self._to_model_device(inputs)

        gen: dict[str, Any] = {"max_new_tokens": max_new_tokens}
        if self._temperature and self._temperature > 0:
            gen.update(do_sample=True, temperature=self._temperature)
        else:
            gen["do_sample"] = False  # greedy, so repeat runs are comparable

        prompt_len = int(inputs["input_ids"].shape[1])
        logger.debug("prompt is %d token(s) for %d image(s)", prompt_len, len(images))

        with self._torch.inference_mode():
            output = self._model.generate(**inputs, **gen)

        return self._decode(output, prompt_len)

    def _encode(self, messages: list[dict[str, Any]], images: list[Any]) -> Any:
        """Render the chat template and tokenise, preferring the fused call.

        The fused `apply_chat_template(tokenize=True)` path is not a style
        choice. Templates for several families (Gemma3, Idefics3) emit the BOS
        token literally, and transformers suppresses the tokeniser's own BOS
        only inside that call:

            if self.tokenizer.bos_token is not None and prompt.startswith(...):
                kwargs["add_special_tokens"] = False

        Rendering with `tokenize=False` and then calling the processor
        separately bypasses that guard and yields two leading BOS tokens —
        silent quality loss with no exception. Older processors do not accept
        the fused form, so fall back to the two-step call for them.
        """
        try:
            return self._processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
            )
        except (TypeError, ValueError, KeyError) as exc:
            logger.debug("fused apply_chat_template unavailable (%s); using two steps", exc)

        chat_text = self._processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        kwargs: dict[str, Any] = {}
        bos = getattr(getattr(self._processor, "tokenizer", None), "bos_token", None)
        if bos and chat_text.startswith(bos):
            # Reproduce the guard the fused path would have applied.
            kwargs["add_special_tokens"] = False
        return self._processor(
            text=[chat_text],
            images=images or None,
            return_tensors="pt",
            padding=True,
            **kwargs,
        )

    def _to_model_device(self, inputs: Any) -> Any:
        """Move and dtype-align the batch in one step.

        `BatchFeature.to(device=..., dtype=...)` casts only floating-point
        entries, so `pixel_values` follows the model's dtype while `input_ids`
        and `image_grid_thw` stay integral and are merely moved. A hand-rolled
        dict comprehension moves without casting, which breaks on the families
        whose vision tower does not cast internally.
        """
        target = self._input_device
        if hasattr(inputs, "to"):
            try:
                return inputs.to(device=target, dtype=self._model.dtype)
            except (TypeError, NotImplementedError):
                return inputs.to(target)
        return {k: (v.to(target) if hasattr(v, "to") else v) for k, v in inputs.items()}

    def _decode(self, output: Any, prompt_len: int) -> str:
        """Strip the prompt from `generate()`'s output and decode.

        Decoder-only models return prompt + continuation; encoder-decoder models
        return only the continuation, so slicing at `prompt_len` would discard
        the answer entirely and hand back "". An empty string is scored as a
        wrong answer with `status=ok`, so this must fail loudly instead.
        """
        if getattr(self._model.config, "is_encoder_decoder", False):
            trimmed = output
        elif output.shape[1] > prompt_len:
            trimmed = output[:, prompt_len:]
        else:
            raise RuntimeError(
                f"generate() returned {output.shape[1]} token(s) for a "
                f"{prompt_len}-token prompt; cannot separate the continuation. "
                "This model may be encoder-decoder without declaring it."
            )
        decoded = self._processor.batch_decode(
            trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        return (decoded[0] if decoded else "").strip()


def _thin_evenly(items: list[Any], keep: int) -> list[Any]:
    """Keep `keep` items spread across the whole list, preserving order.

    Even spacing rather than truncation: the oracle track's frames arrive in
    session order, so slicing the front would drop the most recent sessions
    entirely and turn the long-context ceiling into an early-sessions baseline.
    Several probe questions ask specifically about the *most recent* session.
    """
    if keep >= len(items) or keep <= 0:
        return items
    step = len(items) / keep
    return [items[min(int(i * step), len(items) - 1)] for i in range(keep)]


def _assert_images_reached_the_model(inputs: Any, n_images: int) -> None:
    """Fail loudly when frames were sampled but no pixels were encoded.

    This is the one failure the rest of the harness cannot see. If the processor
    silently drops the images, the prompt still carries its image placeholder
    tokens, generation still succeeds, and the model answers from text alone —
    so the oracle and memory tracks report healthy frame counts while measuring
    priors. Memory Gain then collapses toward zero for a plumbing reason that
    looks exactly like a scientific result.

    It is a real hazard rather than a theoretical one: passing
    `{"type": "image"}` without an `"image"` key does precisely this, because
    transformers collects visuals by key name.
    """
    if not n_images:
        return
    for key in ("pixel_values", "pixel_values_videos", "image_patches"):
        value = inputs.get(key) if hasattr(inputs, "get") else None
        if value is not None and getattr(value, "numel", lambda: 1)():
            return
    raise RuntimeError(
        f"{n_images} frame(s) were sampled but the processor produced no pixel "
        "values, so the model would answer without seeing the video. This is a "
        "chat-template/processor mismatch, not a model failure — refusing to "
        "report an unmeasured track as a measured one."
    )


def _input_device(torch: Any, model: Any) -> Any:
    """Where to put input tensors, refusing the meta device.

    `model.device` is the device of the *first* parameter. Under
    `device_map="auto"` with too little VRAM, accelerate offloads rather than
    raising, and if the first parameter is offloaded that property reads `meta`.
    Moving inputs to `meta` yields storage-free tensors, so generation either
    raises deep inside the model or returns garbage token ids that get scored as
    a wrong answer — a silent hit to the very number being measured. Prefer a
    real device from `hf_device_map`, and refuse outright if none exists.
    """
    mapping = getattr(model, "hf_device_map", None) or {}
    for value in mapping.values():
        # Entries are torch devices, device strings, or bare ints (a CUDA
        # ordinal). Normalise all three before comparing.
        text = f"cuda:{value}" if isinstance(value, int) else str(value)
        if text not in {"meta", "disk"}:
            return torch.device(text)
    device = model.device
    if getattr(device, "type", None) == "meta":
        raise RuntimeError(
            "the model loaded onto the meta device, which means accelerate "
            "offloaded it for lack of memory. Free a GPU, pin one with "
            "CUDA_VISIBLE_DEVICES, or pass --device-map cuda:0 to fail fast "
            "instead of producing untrustworthy answers."
        )
    return device


def _load_model(model_path: str, kwargs: dict[str, Any]):
    """Try the generic image-text class, then fall back to Qwen-VL specifics.

    `AutoModelForImageTextToText` covers current transformers; the explicit Qwen
    classes are the fallback for older pins where the auto class does not map.

    The fallback triggers only on a genuine "unrecognised architecture" error.
    Catching every ValueError/KeyError/OSError also swallowed missing-accelerate,
    bad-path and corrupt-shard failures, then re-raised a RuntimeError blaming
    the architecture — with the real message logged at INFO, i.e. invisible at
    the default WARNING level. That turned a one-line install problem into a
    misleading dead end.
    """
    from transformers import AutoModelForImageTextToText

    unsupported = ("unrecognized configuration class", "does not recognize this architecture")
    try:
        return AutoModelForImageTextToText.from_pretrained(model_path, **kwargs)
    except ValueError as exc:
        if not any(marker in str(exc).lower() for marker in unsupported):
            raise
        first_error = exc
        logger.info("architecture not in the auto mapping (%s); trying Qwen classes", exc)

    for name in ("Qwen3VLForConditionalGeneration", "Qwen2_5_VLForConditionalGeneration"):
        try:
            import transformers

            cls = getattr(transformers, name)
        except AttributeError:
            continue
        try:
            return cls.from_pretrained(model_path, **kwargs)
        except (ValueError, KeyError, OSError) as exc:
            logger.info("%s did not load: %s", name, exc)
    raise RuntimeError(
        f"could not load a vision-language model from {model_path}. If this is a "
        "Qwen3-VL checkpoint, transformers must be >= 4.57 (qwen3_vl is absent "
        "from the auto mappings before then)."
    ) from first_error


def _resolve_dtype(torch: Any, name: str) -> Any:
    mapping = {
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float16": torch.float16,
        "fp16": torch.float16,
        "float32": torch.float32,
        "fp32": torch.float32,
        "auto": "auto",
    }
    if name not in mapping:
        raise SystemExit(f"unknown dtype {name!r}; choose from {sorted(mapping)}")
    return mapping[name]


def main(argv: list[str] | None = None) -> int:
    parser = common_args(__doc__ or "")
    parser.add_argument("--model-path", required=True, help="local directory or HF repo id")
    parser.add_argument("--n-frames", type=int, default=8)
    parser.add_argument("--max-side", type=int, default=768)
    parser.add_argument(
        "--max-oracle-frames",
        type=int,
        default=32,
        help="cap on frames attached to one oracle prompt (sessions x n-frames); "
        "beyond this, frames are thinned evenly and the reduction is logged",
    )
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--note-max-new-tokens", type=int, default=900)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument(
        "--attn-implementation",
        default=None,
        help="e.g. flash_attention_2 or sdpa, if installed",
    )
    parser.add_argument("--trust-remote-code", action="store_true")
    args = parser.parse_args(argv)
    configure_logging(args.log_level)

    adapter = HFVLMAdapter(
        model_path=args.model_path,
        context_mode=args.context_mode,
        system_id=args.system_id,
        n_frames=args.n_frames,
        max_side=args.max_side,
        max_oracle_frames=args.max_oracle_frames,
        max_new_tokens=args.max_new_tokens,
        note_max_new_tokens=args.note_max_new_tokens,
        temperature=args.temperature,
        dtype=args.dtype,
        device_map=args.device_map,
        attn_implementation=args.attn_implementation,
        trust_remote_code=args.trust_remote_code,
    )
    return adapter.run()


if __name__ == "__main__":
    sys.exit(main())
