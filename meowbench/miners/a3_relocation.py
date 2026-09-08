"""A3 object relocation: mine "what did it end up next to" from 3RScan.

An item asks which object a moved thing ended up nearest to, in a later scan of
the same room. The move is 3RScan's own rigid-transform ground truth and the
answer is computed from object centroids, so nothing is inferred from text and
no human labels an item.

WHY THE GOLD IS GEOMETRY AND NOT 3DSSG's `close by`

The obvious shortcut is 3DSSG's hand-annotated `close by` predicate — 26,500 of
them, 23,998 attached to an object that moved, and no coordinates needed. It is
the wrong answer key, and measurement is what settled it: on one bathroom scan
the `close by` partner was the true nearest object in **2 of 20** cases, with
one pair ranking 16th. `bathtub close by bath cabinet` holds while the actual
nearest object is the toilet. `close by` is a loose human proximity judgement,
so using it as the gold for a "closest" question marks a large share of items
wrong AND can leave a nearer object sitting in the distractors.

So the gold comes from `semseg.v2.json` OBB centroids (see
`scripts/fetch_3rscan_obbs.py`, ~7 MiB for 1,380 scans, no usage agreement
needed to fetch). Centroid distance is itself an approximation — two large
touching objects can measure metres apart — which is why an item is only kept
when the nearest object beats the runner-up by `MIN_MARGIN_M`. A near-tie has
no defensible answer and would punish a model for being right.

HOW THE DISTRACTORS ARE BUILT

Distractors are the next-nearest objects in the same room, not a random sample.
That matters twice over: they share the room's vocabulary, so a language prior
has nothing to grip (the failure TemporalBench documented), and they share its
spatial scale, so a model cannot win by noticing one option is implausibly far
away. Objects sharing the gold's label are skipped, since "closest to the
chair" does not identify one of two chairs.

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

#: How much nearer the gold must be than the runner-up. Centroid distance is an
#: approximation -- two large touching objects can measure metres apart -- so a
#: near-tie has no defensible answer and would punish a model for being right.
MIN_MARGIN_M = 0.2

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
    #: How much nearer the gold is than the runner-up. Small margins mean
    #: "closest" is not well defined, so items below MIN_MARGIN_M are rejected.
    margin_m: float
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
        min_margin_m: float = MIN_MARGIN_M,
        max_per_environment: int = 4,
    ) -> None:
        self._data = dataset
        self._rng = random.Random(seed)
        self._min_displacement = min_displacement_m
        self._min_margin = min_margin_m
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

        near = self._data.nearest_to(
            move.rescan, move.instance_id_rescan, exclude=UNUSABLE_LABELS
        )
        if len(near) < N_DISTRACTORS + 1:
            return None

        gold_id, gold_label, gold_distance = near[0]
        runner_up_distance = near[1][2]
        # A clear margin is required, not just the top rank. Distances are
        # between OBB centroids, so two large touching objects can measure
        # metres apart; when the top two are nearly tied, "closest" has no
        # defensible answer and the item would punish a model for being right.
        if runner_up_distance - gold_distance < self._min_margin:
            return None

        # The gold label must be unique in the room, or "closest to the chair"
        # does not identify one object.
        labels = [label for _, label, _ in near]
        if labels.count(gold_label) != 1:
            return None

        # Distractors are the next-nearest objects rather than a random sample:
        # they share the room's vocabulary AND its spatial scale, so a model
        # cannot win by noticing that one option is implausibly far away.
        # Objects sharing the gold's label are skipped for the same
        # disambiguation reason.
        distractors: list[str] = []
        for _, label, _ in near[1:]:
            if label == gold_label or label in distractors:
                continue
            distractors.append(label)
            if len(distractors) == N_DISTRACTORS:
                break
        if len(distractors) < N_DISTRACTORS:
            return None

        options, answer = self._options(gold_label, distractors)
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
                    f"semseg.v2.json#{move.rescan}/{move.instance_id_rescan}",
                    f"semseg.v2.json#{move.rescan}/{gold_id}",
                ],
                notes=(
                    f"{move.label} moved {move.displacement_m:.2f} m; nearest "
                    f"object afterwards is {gold_label} at {gold_distance:.2f} m, "
                    f"next is {near[1][1]} at {runner_up_distance:.2f} m "
                    f"(margin {runner_up_distance - gold_distance:.2f} m)"
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
            margin_m=runner_up_distance - gold_distance,
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
    min_margin_m: float = MIN_MARGIN_M,
    max_per_environment: int = 4,
    limit: int | None = None,
) -> list[MinedItem]:
    miner = A3RelocationMiner(
        dataset,
        seed=seed,
        min_displacement_m=min_displacement_m,
        min_margin_m=min_margin_m,
        max_per_environment=max_per_environment,
    )
    out: list[MinedItem] = []
    for mined in miner.mine():
        out.append(mined)
        if limit and len(out) >= limit:
            break
    return out


__all__ = ["A3RelocationMiner", "MINER", "MIN_MARGIN_M", "MinedItem", "mine_a3"]
