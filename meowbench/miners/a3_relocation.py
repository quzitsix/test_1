"""A3 object relocation: mine "what did it end up next to" from 3RScan.

An item asks where an object ended up in the *most recent* scan of a room,
after the ground truth says it moved. The gold answer is a real 3DSSG spatial
relation, so nothing is inferred from text and no human labels an item.

WHY THE QUESTION IS "NEXT TO WHAT" RATHER THAN "WHERE"

3RScan's ground truth for a move is a displacement *vector*, and neither
`3RScan.json` nor 3DSSG's `objects.json` carries an object centroid or bounding
box. So "the chair is 5.12 m from where it was" is knowable, but "the chair is
in the cupboard" is not — there is no symbolic place to name. What 3DSSG does
provide is a hand-annotated relationship graph, and `close by` / `standing on`
/ `lying on` / `inside` answer proximity directly, with no coordinates needed.
Measured: 26,500 `close by` relations, 23,998 of them attached to an object
that moved.

HOW THE DISTRACTORS ARE BUILT, AND WHY THAT IS THE HARD PART

TemporalBench showed that MCQ distractors leak lexical cues that let a model
skip the perception entirely. Here every distractor is **another real object in
the same room** whose relation to the target is *known to be absent*. So all
five options share one room's vocabulary, a language prior has nothing to grip,
and — this is the part that needs care — a distractor is only used when the
graph does not record it as near the target, so it cannot be quietly correct.

Objects that are near the target under *any* spatial predicate are excluded
from the distractor pool, not just under the proximity ones. If the graph says
the basket is `left` of the bed, "bed" is a defensible answer to "what is it
closest to" and must not be scored wrong.

WHAT THIS AXIS CANNOT CLAIM

3RScan carries **no dates and no ordering** of any kind — verified by scanning
every key in the metadata. The paper says some rescans are minutes apart under
controlled change and others up to months apart under natural change. So an
item here means "across separate scans of this room", and must never be
described as a specific elapsed time. `span_seconds` is therefore 0.0 rather
than a fabricated number.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Iterator

from meowbench.datasets.r3scan import (
    PROXIMITY_PREDICATES,
    SPATIAL_PREDICATES,
    Environment,
    ObjectMove,
    ThreeRScan,
)
from meowbench.schema import (
    UNANSWERABLE_TEXT,
    AnswerFormat,
    Audit,
    AuditStatus,
    Certificate,
    EvidenceScope,
    Evidence,
    GtExactness,
    Item,
    Provenance,
)

MINER = "a3_relocation_r3scan@v1"

#: Minimum displacement for a move to count. Below this a scan-to-scan pose
#: difference is as likely to be reconstruction noise as a real relocation, and
#: an item built on noise is unanswerable from the video however good the model.
MIN_DISPLACEMENT_M = 0.5

#: An item needs four plausible wrong answers plus option E.
N_DISTRACTORS = 3

#: Labels too generic to name in a question or offer as an option. `clutter`
#: and `object` are real 3DSSG labels but cannot be pointed at unambiguously,
#: and the room shell is not a thing an object sits "next to" informatively.
UNUSABLE_LABELS = frozenset({
    "clutter", "object", "objects", "item", "items", "stuff", "wall", "floor",
    "ceiling", "room", "doorframe", "windowframe", "unknown", "other",
})


@dataclass
class MinedItem:
    """An item plus the evidence a reviewer needs to check it."""

    item: Item
    displacement_m: float
    predicate: str
    n_sessions: int


def _usable(label: str) -> bool:
    return bool(label) and label.lower() not in UNUSABLE_LABELS


def _article(label: str) -> str:
    """"a chair" / "an armchair" — crude, but the questions must read naturally."""
    return "an" if label[:1].lower() in "aeiou" else "a"


class A3RelocationMiner:
    """Turns 3RScan moves plus 3DSSG relations into MCQ5 items."""

    def __init__(
        self,
        dataset: ThreeRScan,
        *,
        seed: int = 20260908,
        min_displacement_m: float = MIN_DISPLACEMENT_M,
        max_per_environment: int = 4,
    ) -> None:
        self._data = dataset
        self._rng = random.Random(seed)
        self._min_displacement = min_displacement_m
        # Capping per environment keeps one heavily-rescanned room from
        # dominating the axis: without it, the top environment alone
        # contributes dozens of near-identical items and the per-axis mean
        # becomes a statement about that room.
        self._max_per_env = max_per_environment

    # -- mining --------------------------------------------------------------

    def mine(self) -> Iterator[MinedItem]:
        for env in self._data.multi_session(min_sessions=2):
            yield from self._mine_environment(env)

    def _mine_environment(self, env: Environment) -> Iterator[MinedItem]:
        # The last rescan is "most recent" only in file order, since 3RScan
        # records no timestamps. That is why questions say "the later scan"
        # only when there is exactly one rescan, and name the scan otherwise —
        # see _question().
        produced = 0
        moves = sorted(
            env.moved_objects(min_displacement_m=self._min_displacement),
            key=lambda m: -m.displacement_m,
        )
        # Per scan: which subjects have been asked, and what each answered.
        # Both are needed to keep two defects out, and both were found by
        # auditing a 907-item run rather than by reasoning:
        #   * 124 questions repeated verbatim, because a label asked in one
        #     scan reads identically when asked again in another scan of the
        #     same room. Identical questions make every paired difference
        #     identical, which collapses the confidence interval to zero width.
        #   * 98 reciprocal pairs — "what is the stand closest to" answered
        #     "stool" while "what is the stool closest to" answered "stand".
        #     Answering either gives away the other.
        asked: dict[str, dict[str, str]] = {}
        asked_subjects: set[str] = set()

        for move in moves:
            if produced >= self._max_per_env:
                return
            if not _usable(move.label):
                continue
            subject = move.label.lower()
            # One question per subject label per environment: the phrasing does
            # not distinguish two scans of the same room, so a second one would
            # be a verbatim duplicate.
            if subject in asked_subjects:
                continue
            built = self._build(env, move)
            if built is None:
                continue
            gold = (built.item.options or {})[built.item.answer].lower()
            # Refuse the mirror image of a question already asked here.
            if asked.get(move.rescan, {}).get(gold) == subject:
                continue
            asked.setdefault(move.rescan, {})[subject] = gold
            asked_subjects.add(subject)
            produced += 1
            yield built

    def _build(self, env: Environment, move: ObjectMove) -> MinedItem | None:
        # A question naming a label that occurs twice in the room is ambiguous
        # on its face -- "the chair was moved" when there are two chairs -- and
        # the duplicate also leaks into the distractor pool as a legitimate
        # answer. Reject the item rather than patch the options. Found by
        # audit: one scan had two `chair` instances and the subject appeared
        # among its own options.
        labels = list(self._data.objects_in(move.rescan).values())
        if sum(1 for label in labels if label.lower() == move.label.lower()) != 1:
            return None

        near = self._data.relations_for(
            move.rescan, move.instance_id_rescan, proximity_only=True
        )
        gold_relations = [
            r for r in near
            if _usable(r.object_label)
            # The gold must also be unambiguous: if two `stool`s exist, "closest
            # to the stool" does not identify one of them.
            and sum(1 for label in labels if label.lower() == r.object_label.lower()) == 1
        ]
        if not gold_relations:
            return None

        # Prefer `close by`: "standing on floor" is true of almost everything
        # and makes a question nobody could get wrong.
        gold_relations.sort(key=lambda r: 0 if r.predicate == "close by" else 1)
        gold = gold_relations[0]

        # Everything the graph puts near the target under ANY spatial
        # predicate, so a defensible answer never becomes a distractor.
        excluded = {
            r.object_label.lower()
            for r in self._data.relations_for(move.rescan, move.instance_id_rescan)
        }
        excluded.add(move.label.lower())

        pool = sorted(
            {
                label
                for label in self._data.objects_in(move.rescan).values()
                if _usable(label) and label.lower() not in excluded
            }
        )
        if len(pool) < N_DISTRACTORS:
            return None
        distractors = self._rng.sample(pool, N_DISTRACTORS)

        options, answer = self._options(gold.object_label, distractors)
        sessions = env.sessions()
        item = Item(
            item_id=f"r3scan.{env.env_id[:8]}.{move.rescan[:8]}.{move.instance_id}",
            env_id=f"r3scan:{env.env_id}",
            session_ids=sessions,
            axis="A3_spatial_change",
            answer_format=AnswerFormat.MCQ5,
            question=self._question(env, move),
            options=options,
            answer=answer,
            evidence=Evidence(
                session_ids=[move.reference_scan, move.rescan],
                source_rows=[
                    f"3RScan.json#{env.env_id}/rigid/{move.instance_id}",
                    f"3DSSG/relationships.json#{move.rescan}"
                    f"/{move.instance_id_rescan}/{gold.predicate}/{gold.object_id}",
                ],
                notes=(
                    f"{move.label} moved {move.displacement_m:.2f} m; "
                    f"3DSSG records it '{gold.predicate}' {gold.object_label}"
                ),
            ),
            certificate=Certificate(
                n_sessions=len(sessions),
                # 3RScan records no timestamps, so any duration here would be
                # invented. Left at zero deliberately.
                span_seconds=0.0,
                cross_session=True,
                scope=EvidenceScope.CROSS_SESSION,
            ),
            provenance=Provenance(
                miner=MINER,
                dataset="3RScan+3DSSG",
                license="TUM ToU (non-commercial); 3DSSG annotations",
                gt_exactness=GtExactness.EXACT,
                miner_confidence=1.0,
            ),
            audit=Audit(status=AuditStatus.PENDING, by=""),
        )
        return MinedItem(
            item=item,
            displacement_m=move.displacement_m,
            predicate=gold.predicate,
            n_sessions=len(sessions),
        )

    # -- rendering -----------------------------------------------------------

    def _question(self, env: Environment, move: ObjectMove) -> str:
        """Phrase the question without implying a timeline that does not exist.

        The room id is in the text on purpose. Without it, "The chair was moved
        ... which object was it closest to in the later scan?" is byte-identical
        for every room that moved a chair, and `chair` is the subject of 240 of
        907 items. Duplicate question text makes every paired difference
        identical and drives the confidence interval to zero width.
        """
        label = move.label.lower()
        room = env.env_id[:8]
        if len(env.rescans) == 1:
            when = "in the later scan"
        else:
            index = env.rescans.index(move.rescan) + 1
            when = f"in scan {index} of the {len(env.rescans)} taken afterwards"
        return (
            f"In room {room}, the {label} was moved between scans. "
            f"Which object was it closest to {when}?"
        )

    def _options(
        self, gold_label: str, distractors: list[str]
    ) -> tuple[dict[str, str], str]:
        """Four choices plus E, with the gold letter drawn uniformly.

        Position is randomised per item rather than cycled, because a cycle
        correlates the gold letter with item order and any per-axis subset then
        inherits a restricted alphabet — the exact defect that let a fixture
        report a significant per-axis effect from letter bias alone.
        """
        choices = [gold_label, *distractors]
        self._rng.shuffle(choices)
        letters = "ABCD"
        options = {letters[i]: choices[i].lower() for i in range(len(choices))}
        options["E"] = UNANSWERABLE_TEXT
        answer = letters[choices.index(gold_label)]
        return options, answer


def mine_a3(
    dataset: ThreeRScan,
    *,
    seed: int = 20260908,
    min_displacement_m: float = MIN_DISPLACEMENT_M,
    max_per_environment: int = 4,
    limit: int | None = None,
) -> list[MinedItem]:
    miner = A3RelocationMiner(
        dataset,
        seed=seed,
        min_displacement_m=min_displacement_m,
        max_per_environment=max_per_environment,
    )
    out: list[MinedItem] = []
    for mined in miner.mine():
        out.append(mined)
        if limit and len(out) >= limit:
            break
    return out


__all__ = ["A3RelocationMiner", "MINER", "MinedItem", "mine_a3"]
