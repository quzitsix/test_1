"""3RScan: the same room scanned two to eleven times, with what moved labelled.

Why this dataset earns a loader of its own: it is the only public resource
found that satisfies MEOWBench's decisive requirement outright and at scale —
478 environments, each rescanned 2 to 11 times, with **instance ids held fixed
across scans** and a ground-truth rigid transform for every object that moved.
"Where did the mug end up in the most recent session" is therefore a derivable
label, not something to infer from narration text. Measured on the released
metadata: 1,482 scans, 2,933 rigid moves carrying a transform, 542 removals,
558 non-rigid changes.

The whole mining signal is two files totalling under 6 MiB, so items can be
authored before a single scan is downloaded. Scans are only needed to *run* a
benchmark, not to build one.

FACTS ESTABLISHED BY MEASUREMENT, NOT BY READING THE PAPER

* **Transforms are column-major with the translation at indices 12–14, in
  metres.** Read as row-major, every one of the 2,933 moves has a displacement
  of exactly 0.000 m; read as column-major the median is 1.545 m and the
  maximum 19.737 m, which is the scale of a room. The wrong convention yields
  a silent all-zeros dataset, so `displacement_m` is derived here once.
* **The test split withholds its answers.** In `train` and `validation`,
  `rigid` entries are objects with `instance_reference`, `instance_rescan`,
  `symmetry` and `transform`. In `test` (46 environments, 390 moves) they are
  bare integers — the transform is stripped. That is a natural held-out set,
  not a gap, but a loader that assumed one shape would crash or silently skip.
* **Instance ids resolve to object names 99.9% of the time** (2,930 of 2,933)
  against 3DSSG's `objects.json`, which covers all 1,482 scans. The three
  misses are dropped rather than emitted with a null label.

WHAT THIS DATASET CANNOT DO, STATED UP FRONT

There are **no people** in it — these are static reconstructions of empty
rooms. It cannot serve the person-relations axis. Scans are room-scale rather
than whole-home. And critically for any claim about *long-term* memory: the
metadata carries **no dates or ordering of any kind** (verified by scanning
every key in the file), while the paper describes some rescans as minutes apart
under controlled change and others as up to months apart under natural change.
So the elapsed time between two sessions of the same room is unknown, and items
mined here must not claim a specific gap.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

#: Index of the translation column in 3RScan's column-major 4x4 transforms.
#: See the module docstring: reading these as row-major zeroes every move.
_TX, _TY, _TZ = 12, 13, 14

#: Labels that name the room itself rather than a thing in it. An A3 question
#: about a wall "moving" is a reconstruction artefact, not a relocation.
STRUCTURAL_LABELS = frozenset({
    "wall", "floor", "ceiling", "door", "doorframe", "window", "windowframe",
    "stairs", "column", "beam", "ceiling light", "light", "heater", "radiator",
    "pipe", "wall frame", "floor mat", "carpet", "rug",
})

#: 3DSSG predicates that describe where something is, as opposed to what it is
#: like. Taken from the released `relationships.txt`, not invented: the file
#: also carries comparative predicates (`same color`, `bigger than`,
#: `messier than`) which say nothing about position. Measured counts in
#: `relationships.json`: left 45,357, right 45,357, close by 26,500,
#: behind 24,165, front 24,165, standing on 12,672, attached to 12,761.
SPATIAL_PREDICATES = frozenset({
    "close by", "left", "right", "front", "behind", "supported by",
    "standing on", "lying on", "hanging on", "attached to", "inside",
    "standing in", "lying in", "hanging in", "leaning against",
    "connected to", "build in",
})

#: The subset that means "near", which is what a "closest to" question needs.
PROXIMITY_PREDICATES = frozenset({"close by", "standing on", "lying on", "inside"})


@dataclass(frozen=True)
class Relation:
    """A spatial fact about one scan: `subject predicate object`.

    3DSSG ships these as `[subject_id, object_id, predicate_id, predicate]`,
    which is why the field order here looks transposed relative to the file.

    These matter more than they first appear. 3RScan gives a *displacement
    vector* for each moved object and no absolute coordinates, and 3DSSG's
    `objects.json` has no centroid or bounding box either — so "which object is
    the chair closest to" cannot be computed from geometry without downloading
    the per-scan meshes. The relationship graph answers it symbolically
    instead, with no coordinates needed at all.
    """

    scan_id: str
    subject_id: int
    predicate: str
    object_id: int
    subject_label: str
    object_label: str

    @property
    def is_spatial(self) -> bool:
        return self.predicate in SPATIAL_PREDICATES

    @property
    def is_proximity(self) -> bool:
        return self.predicate in PROXIMITY_PREDICATES


@dataclass(frozen=True)
class ObjectMove:
    """One object that changed pose between a reference scan and a rescan."""

    env_id: str
    reference_scan: str
    rescan: str
    instance_id: int
    instance_id_rescan: int
    label: str
    displacement_m: float
    #: 0 when the object has no rotational symmetry. Non-zero means the
    #: rotation is ambiguous, so questions about *orientation* are unsafe even
    #: though questions about position remain fine.
    symmetry: int
    transform: tuple[float, ...]

    @property
    def is_structural(self) -> bool:
        return self.label.lower() in STRUCTURAL_LABELS


@dataclass(frozen=True)
class ObjectRemoval:
    """An object present in the reference scan and absent from a rescan.

    Directly usable as an unanswerable control: after the removal, "where is
    the X?" has no answer in that session, which is exactly what option E
    exists for — and unlike a synthetic control, the absence is real.
    """

    env_id: str
    reference_scan: str
    rescan: str
    instance_id: int
    label: str


@dataclass
class Environment:
    """One physical room, with every scan of it."""

    env_id: str
    split: str
    reference_scan: str
    rescans: list[str] = field(default_factory=list)
    moves: list[ObjectMove] = field(default_factory=list)
    removals: list[ObjectRemoval] = field(default_factory=list)
    #: True when this environment's transforms are withheld (the test split).
    answers_withheld: bool = False

    @property
    def n_sessions(self) -> int:
        """Scans of this room, counting the reference as the first session."""
        return 1 + len(self.rescans)

    def sessions(self) -> list[str]:
        return [self.reference_scan, *self.rescans]

    def moved_objects(self, *, min_displacement_m: float = 0.0) -> list[ObjectMove]:
        """Non-structural moves, optionally filtered by how far they travelled.

        A threshold matters because a scan-to-scan pose difference of a few
        centimetres is as likely to be reconstruction noise as a real
        relocation, and an item built on noise is unanswerable from the video.
        """
        return [
            m for m in self.moves
            if not m.is_structural and m.displacement_m >= min_displacement_m
        ]


class ThreeRScan:
    """Loads 3RScan's metadata and 3DSSG's labels into one queryable object."""

    def __init__(self, root: Path | str) -> None:
        """`root` holds `3RScan.json` and the unpacked `3DSSG/` directory."""
        self.root = Path(root)
        self._labels: dict[str, dict[int, str]] = {}
        self._attributes: dict[str, dict[int, dict]] = {}
        self._relations: dict[str, list[Relation]] = {}
        self.environments: dict[str, Environment] = {}
        self._load()

    # -- loading -------------------------------------------------------------

    def _load(self) -> None:
        objects = self._find("3DSSG/objects.json", "objects.json")
        if objects:
            self._load_labels(objects)
        relationships = self._find("3DSSG/relationships.json", "relationships.json")
        if relationships:
            self._load_relations(relationships)
        meta = self._find("3RScan.json")
        if not meta:
            raise FileNotFoundError(f"3RScan.json not found under {self.root}")
        self._load_environments(meta)

    def _find(self, *relative: str) -> Path | None:
        for name in relative:
            path = self.root / name
            if path.is_file():
                return path
        return None

    def _load_labels(self, path: Path) -> None:
        payload = json.loads(path.read_text(encoding="utf-8"))
        for scan in payload.get("scans", []):
            scan_id = scan.get("scan")
            if not scan_id:
                continue
            labels: dict[int, str] = {}
            attributes: dict[int, dict] = {}
            for obj in scan.get("objects", []):
                try:
                    instance = int(obj["id"])
                except (KeyError, TypeError, ValueError):
                    continue
                labels[instance] = obj.get("label", "")
                attributes[instance] = obj.get("attributes", {}) or {}
            self._labels[scan_id] = labels
            self._attributes[scan_id] = attributes

    def _load_relations(self, path: Path) -> None:
        payload = json.loads(path.read_text(encoding="utf-8"))
        for scan in payload.get("scans", []):
            scan_id = scan.get("scan")
            if not scan_id:
                continue
            labels = self._labels.get(scan_id, {})
            found: list[Relation] = []
            for row in scan.get("relationships", []):
                # [subject_id, object_id, predicate_id, predicate_name]
                if not isinstance(row, list) or len(row) < 4:
                    continue
                try:
                    subject, obj = int(row[0]), int(row[1])
                except (TypeError, ValueError):
                    continue
                subject_label, object_label = labels.get(subject), labels.get(obj)
                if not subject_label or not object_label:
                    continue  # an unnameable endpoint cannot appear in a question
                found.append(
                    Relation(
                        scan_id=scan_id,
                        subject_id=subject,
                        predicate=str(row[3]),
                        object_id=obj,
                        subject_label=subject_label,
                        object_label=object_label,
                    )
                )
            self._relations[scan_id] = found

    def _load_environments(self, path: Path) -> None:
        payload = json.loads(path.read_text(encoding="utf-8"))
        for entry in payload:
            reference = entry.get("reference")
            if not reference:
                continue
            env = Environment(
                env_id=reference,
                split=entry.get("type", "unknown"),
                reference_scan=reference,
            )
            labels = self._labels.get(reference, {})
            for scan in entry.get("scans") or []:
                rescan = scan.get("reference")
                if not rescan:
                    continue
                env.rescans.append(rescan)

                for item in scan.get("rigid") or []:
                    if not isinstance(item, dict):
                        # Bare int: the test split withholds the transform.
                        env.answers_withheld = True
                        continue
                    move = self._make_move(env, rescan, item, labels)
                    if move is not None:
                        env.moves.append(move)

                for instance in scan.get("removed") or []:
                    try:
                        instance = int(instance)
                    except (TypeError, ValueError):
                        continue
                    label = labels.get(instance)
                    if not label:
                        continue
                    env.removals.append(
                        ObjectRemoval(reference, reference, rescan, instance, label)
                    )
            self.environments[reference] = env

    def _make_move(
        self, env: Environment, rescan: str, item: dict, labels: dict[int, str]
    ) -> ObjectMove | None:
        transform = item.get("transform")
        if not transform or len(transform) < 16:
            return None
        try:
            instance = int(item["instance_reference"])
        except (KeyError, TypeError, ValueError):
            return None
        label = labels.get(instance)
        if not label:
            # 3 of 2,933 moves reference an instance 3DSSG does not label.
            # An item with a nameless object cannot be asked, so drop it.
            return None
        try:
            rescan_instance = int(item.get("instance_rescan", instance))
        except (TypeError, ValueError):
            rescan_instance = instance
        return ObjectMove(
            env_id=env.env_id,
            reference_scan=env.reference_scan,
            rescan=rescan,
            instance_id=instance,
            instance_id_rescan=rescan_instance,
            label=label,
            displacement_m=math.dist(
                (0.0, 0.0, 0.0), (transform[_TX], transform[_TY], transform[_TZ])
            ),
            symmetry=int(item.get("symmetry") or 0),
            transform=tuple(float(x) for x in transform),
        )

    # -- queries -------------------------------------------------------------

    def label_of(self, scan_id: str, instance_id: int) -> str | None:
        return self._labels.get(scan_id, {}).get(instance_id)

    def attributes_of(self, scan_id: str, instance_id: int) -> dict:
        return self._attributes.get(scan_id, {}).get(instance_id, {})

    def objects_in(self, scan_id: str) -> dict[int, str]:
        """Every labelled instance in a scan — the pool for distractors."""
        return dict(self._labels.get(scan_id, {}))

    def relations_in(
        self, scan_id: str, *, spatial_only: bool = True
    ) -> list[Relation]:
        """Relations recorded for a scan.

        `spatial_only` drops the comparative predicates (`same color`,
        `bigger than`, `messier than`), which are two thirds of the file and
        say nothing about where anything is.
        """
        relations = self._relations.get(scan_id, [])
        return [r for r in relations if r.is_spatial] if spatial_only else list(relations)

    def relations_for(
        self, scan_id: str, instance_id: int, *, proximity_only: bool = False
    ) -> list[Relation]:
        """Where this object is, in this scan, as symbolic facts.

        This is the primitive an A3 item is built from: take an object known to
        have moved, then ask what it ended up next to.
        """
        wanted = PROXIMITY_PREDICATES if proximity_only else SPATIAL_PREDICATES
        return [
            r for r in self._relations.get(scan_id, [])
            if r.subject_id == instance_id and r.predicate in wanted
        ]

    def has_relations(self, scan_id: str) -> bool:
        """3DSSG annotates 1,335 of the 1,482 scans, so this must be checked."""
        return bool(self._relations.get(scan_id))

    def multi_session(
        self, *, min_sessions: int = 2, splits: tuple[str, ...] = ("train", "validation")
    ) -> Iterator[Environment]:
        """Environments usable for cross-session items.

        Defaults exclude `test`, whose transforms are withheld, so callers do
        not silently mine environments with no answers.
        """
        for env in self.environments.values():
            if env.split not in splits:
                continue
            if env.n_sessions >= min_sessions and not env.answers_withheld:
                yield env

    def summary(self) -> dict[str, object]:
        envs = list(self.environments.values())
        moves = [m for e in envs for m in e.moves]
        spatial = sum(
            1 for rels in self._relations.values() for r in rels if r.is_spatial
        )
        return {
            "environments": len(envs),
            "scans": sum(e.n_sessions for e in envs),
            "splits": {
                s: sum(1 for e in envs if e.split == s)
                for s in sorted({e.split for e in envs})
            },
            "moves_with_transform": len(moves),
            "moves_non_structural": sum(1 for m in moves if not m.is_structural),
            "removals": sum(len(e.removals) for e in envs),
            "labelled_scans": len(self._labels),
            "scans_with_relations": len(self._relations),
            "spatial_relations": spatial,
            "median_displacement_m": (
                sorted(m.displacement_m for m in moves)[len(moves) // 2]
                if moves else 0.0
            ),
        }


__all__ = [
    "Environment",
    "ObjectMove",
    "ObjectRemoval",
    "PROXIMITY_PREDICATES",
    "Relation",
    "SPATIAL_PREDICATES",
    "STRUCTURAL_LABELS",
    "ThreeRScan",
]
