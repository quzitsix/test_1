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
        max_new_tokens: int = 512,
        note_max_new_tokens: int = 900,
        temperature: float = 0.0,
        dtype: str = "bfloat16",
        device_map: str = "auto",
        attn_implementation: str | None = None,
        trust_remote_code: bool = False,
    ) -> None:
        super().__init__(system_id or f"hf_vlm:{os.path.basename(model_path.rstrip('/\\'))}")
        self.context_mode = context_mode
        self._n_frames = n_frames
        self._max_side = max_side
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
        logger.info("model ready")

        self._notes: list[str] = []
        self._session_paths: list[str] = []

    # -- lifecycle -----------------------------------------------------------

    def on_env_begin(self, env_id: str, n_sessions: int) -> None:
        self._notes.clear()
        self._session_paths.clear()

    def ingest(self, msg: dict[str, Any]) -> dict[str, Any]:
        path = msg.get("video_path")
        if not path:
            return {"frames": 0, "note_chars": 0}
        if self.context_mode == "oracle":
            self._session_paths.append(path)
            return {"frames": 0, "deferred": True}

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
            for path in self._session_paths:
                images.extend(
                    f.image
                    for f in sample_frames(
                        path, n_frames=self._n_frames, max_side=self._max_side
                    )
                )
        elif self.context_mode == "memory" and self._notes:
            prompt = (
                "Here are your own notes from watching this home. The video is no "
                "longer available; answer from these notes.\n\n"
                + "\n\n".join(self._notes)
                + "\n\n"
                + prompt
            )
        text = self._generate(images, prompt, max_new_tokens=self._max_new_tokens)
        return parse_reply(msg, text)

    # -- generation ----------------------------------------------------------

    def _generate(self, images: list[Any], prompt: str, *, max_new_tokens: int) -> str:
        content: list[dict[str, Any]] = [{"type": "image"} for _ in images]
        content.append({"type": "text", "text": prompt})
        messages = [{"role": "user", "content": content}]

        chat_text = self._processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self._processor(
            text=[chat_text],
            images=images or None,
            return_tensors="pt",
            padding=True,
        )
        inputs = {
            k: (v.to(self._model.device) if hasattr(v, "to") else v) for k, v in inputs.items()
        }

        gen: dict[str, Any] = {"max_new_tokens": max_new_tokens}
        if self._temperature and self._temperature > 0:
            gen.update(do_sample=True, temperature=self._temperature)
        else:
            gen["do_sample"] = False  # greedy, so repeat runs are comparable

        with self._torch.inference_mode():
            output = self._model.generate(**inputs, **gen)

        # Strip the prompt; generate() returns prompt + continuation.
        prompt_len = inputs["input_ids"].shape[1]
        trimmed = output[:, prompt_len:]
        decoded = self._processor.batch_decode(
            trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        return (decoded[0] if decoded else "").strip()


def _load_model(model_path: str, kwargs: dict[str, Any]):
    """Try the generic image-text class, then fall back to Qwen-VL specifics.

    `AutoModelForImageTextToText` covers current transformers; the explicit Qwen
    classes are the fallback for older pins where the auto class does not map.
    """
    from transformers import AutoModelForImageTextToText

    try:
        return AutoModelForImageTextToText.from_pretrained(model_path, **kwargs)
    except (ValueError, KeyError, OSError) as exc:
        logger.info("AutoModelForImageTextToText did not load (%s); trying Qwen classes", exc)

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
        f"could not load a vision-language model from {model_path}. Check the path, "
        "and that your transformers version supports this architecture."
    )


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
