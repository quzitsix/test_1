"""Suite loading: read a frozen release off disk and check its integrity.

A release is two JSONL files plus a manifest:

    releases/v0.1/
      items.jsonl     one Item per line
      envs.jsonl      one EnvManifest per line
      manifest.json   checksums + counts, written by `build`

`suite_sha` is recorded on every run so a number can always be traced to the
exact corpus that produced it. Loading verifies it, because a suite edited after
the fact silently invalidates every stored result — and that failure is
otherwise invisible.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

from meowbench.schema import EnvManifest, Item

MANIFEST_NAME = "manifest.json"
ITEMS_NAME = "items.jsonl"
ENVS_NAME = "envs.jsonl"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass
class Suite:
    """A loaded, verified benchmark release."""

    name: str
    path: Path
    items: list[Item]
    envs: dict[str, EnvManifest]
    suite_sha: str = ""
    manifest: dict[str, object] = field(default_factory=dict)

    @property
    def axes(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for item in self.items:
            counts[item.axis] = counts.get(item.axis, 0) + 1
        return dict(sorted(counts.items()))

    def filter(
        self,
        *,
        axes: set[str] | None = None,
        env_ids: set[str] | None = None,
        limit: int | None = None,
    ) -> Suite:
        """A narrowed view, for smoke runs and per-axis debugging.

        The sha is deliberately suffixed rather than recomputed: a filtered run
        must never be mistaken for a full-suite result.
        """
        items = self.items
        if axes:
            items = [i for i in items if i.axis in axes]
        if env_ids:
            items = [i for i in items if i.env_id in env_ids]
        if limit is not None:
            items = items[:limit]
        kept = {i.env_id for i in items}
        suffix = []
        if axes:
            suffix.append("axes=" + ",".join(sorted(axes)))
        if env_ids:
            suffix.append("envs=" + ",".join(sorted(env_ids)))
        if limit is not None:
            suffix.append(f"limit={limit}")
        return Suite(
            name=self.name + (f"[{';'.join(suffix)}]" if suffix else ""),
            path=self.path,
            items=items,
            envs={k: v for k, v in self.envs.items() if k in kept},
            suite_sha=self.suite_sha + ("+filtered" if suffix else ""),
            manifest=self.manifest,
        )

    def describe(self) -> str:
        lines = [
            f"suite: {self.name}",
            f"path:  {self.path}",
            f"sha:   {self.suite_sha or '(none)'}",
            f"items: {len(self.items)}  envs: {len(self.envs)}",
        ]
        cross = sum(
            1 for i in self.items if i.certificate and i.certificate.cross_session
        )
        if self.items:
            lines.append(
                f"cross-session: {cross}/{len(self.items)} "
                f"({100.0 * cross / len(self.items):.0f}%)"
            )
        for axis, n in self.axes.items():
            lines.append(f"  {axis:28} {n}")
        return "\n".join(lines)


def load_suite(path: Path | str, *, verify: bool = True) -> Suite:
    """Load a release directory, verifying checksums when a manifest exists."""
    root = Path(path)
    items_path, envs_path = root / ITEMS_NAME, root / ENVS_NAME
    for required in (items_path, envs_path):
        if not required.is_file():
            raise FileNotFoundError(f"not a suite directory (missing {required.name}): {root}")

    items = [Item.model_validate_json(line) for line in _lines(items_path)]
    envs = {
        env.env_id: env
        for env in (EnvManifest.model_validate_json(line) for line in _lines(envs_path))
    }

    manifest: dict[str, object] = {}
    manifest_path = root / MANIFEST_NAME
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    suite_sha = str(manifest.get("suite_sha", "")) if manifest else ""
    if verify and manifest:
        _verify(root, manifest, items, envs)

    missing = sorted({i.env_id for i in items} - set(envs))
    if missing:
        raise ValueError(f"items reference environments absent from {ENVS_NAME}: {missing}")

    # Media paths may be recorded relative to the suite (portable fixtures) or
    # absolutely (pointing into a dataset root). Resolve the relative ones now
    # so the runner never has to know where the suite lives.
    for env in envs.values():
        for session in env.sessions:
            for attr in ("video_path", "asr_path", "caption_path"):
                value = getattr(session, attr)
                if value and not Path(value).is_absolute():
                    setattr(session, attr, str((root / value).resolve()))

    return Suite(
        name=str(manifest.get("name") or root.name),
        path=root,
        items=items,
        envs=envs,
        suite_sha=suite_sha or compute_suite_sha(items_path, envs_path),
        manifest=manifest,
    )


def compute_suite_sha(items_path: Path, envs_path: Path) -> str:
    """Digest of both files, order-independent between them."""
    combined = hashlib.sha256()
    for path in (items_path, envs_path):
        combined.update(file_sha256(path).encode("ascii"))
    return combined.hexdigest()


def write_suite(
    root: Path | str,
    items: list[Item],
    envs: dict[str, EnvManifest],
    *,
    name: str = "",
    extra: dict[str, object] | None = None,
) -> Suite:
    """Freeze a suite to disk with a manifest, then load it back verified."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    items_path, envs_path = root / ITEMS_NAME, root / ENVS_NAME

    with items_path.open("w", encoding="utf-8", newline="\n") as fh:
        for item in items:
            fh.write(item.model_dump_json() + "\n")
    with envs_path.open("w", encoding="utf-8", newline="\n") as fh:
        for env in envs.values():
            fh.write(env.model_dump_json() + "\n")

    axes: dict[str, int] = {}
    for item in items:
        axes[item.axis] = axes.get(item.axis, 0) + 1
    manifest: dict[str, object] = {
        "schema": "meowbench.suite/1",
        "name": name or root.name,
        "n_items": len(items),
        "n_envs": len(envs),
        "axes": dict(sorted(axes.items())),
        "items_sha256": file_sha256(items_path),
        "envs_sha256": file_sha256(envs_path),
        "suite_sha": compute_suite_sha(items_path, envs_path),
    }
    if extra:
        manifest.update(extra)
    (root / MANIFEST_NAME).write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return load_suite(root)


def _lines(path: Path) -> list[str]:
    return [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _verify(
    root: Path,
    manifest: dict[str, object],
    items: list[Item],
    envs: dict[str, EnvManifest],
) -> None:
    for key, filename in (("items_sha256", ITEMS_NAME), ("envs_sha256", ENVS_NAME)):
        expected = manifest.get(key)
        if not expected:
            continue
        actual = file_sha256(root / filename)
        if actual != expected:
            raise ValueError(
                f"{filename} does not match the manifest checksum "
                f"(expected {expected[:12]}..., got {actual[:12]}...). "
                "The suite was edited after freezing; results keyed to the old "
                "sha are no longer comparable."
            )
    for key, actual_n in (("n_items", len(items)), ("n_envs", len(envs))):
        expected_n = manifest.get(key)
        if expected_n is not None and int(expected_n) != actual_n:
            raise ValueError(f"manifest says {key}={expected_n} but found {actual_n}")


__all__ = [
    "ENVS_NAME",
    "ITEMS_NAME",
    "MANIFEST_NAME",
    "Suite",
    "compute_suite_sha",
    "file_sha256",
    "load_suite",
    "write_suite",
]
