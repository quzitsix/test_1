"""Data contracts for MEOWBench.

Two families live here:

1. **Corpus schema** — `Item` / `EnvManifest`, the frozen on-disk benchmark
   (`items.jsonl` / `envs.jsonl`).
2. **Wire schema** — the JSONL messages exchanged with a black-box system
   under test over stdin/stdout.

A hard rule enforced by `Item.to_query()`: audit-only fields (`evidence`,
`certificate`, `bias_score`, `audit`, `answer*`) are *never* handed to the
system under test. Leaking `evidence` would tell it exactly where to look;
leaking `answer` would be the whole benchmark.
"""

from __future__ import annotations

from enum import Enum
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

# --------------------------------------------------------------------------
# enums
# --------------------------------------------------------------------------


class AnswerFormat(str, Enum):
    MCQ5 = "mcq5"
    OPEN = "open"
    NUMERIC = "numeric"


class ContextMode(str, Enum):
    """How much the harness lets the system see at query time.

    The three modes are the same adapter code under different staging
    policies, which is what makes Memory Gain measurable.
    """

    BLIND = "blind"
    MEMORY = "memory"
    ORACLE = "oracle"


class EvidenceScope(str, Enum):
    SINGLE_SCENE = "single_scene"
    SINGLE_SESSION = "single_session"
    CROSS_SESSION = "cross_session"
    WHOLE_ENV = "whole_env"


class AuditStatus(str, Enum):
    PENDING = "pending"
    ACCEPTED = "accepted"
    EDITED = "edited"
    REJECTED = "rejected"


class GtExactness(str, Enum):
    EXACT = "exact"
    DERIVED = "derived"
    WEAK = "weak"


class PredictionStatus(str, Enum):
    OK = "ok"
    ERROR = "error"
    TIMEOUT = "timeout"
    MALFORMED = "malformed"
    SKIPPED = "skipped"


#: The canonical option-E text. Kept identical corpus-wide so that models
#: cannot detect the unanswerable class from wording alone.
UNANSWERABLE_TEXT = "The information is not available based on the given context"

MCQ_LETTERS = ("A", "B", "C", "D", "E")

# --------------------------------------------------------------------------
# corpus schema
# --------------------------------------------------------------------------


class Span(BaseModel):
    """A time window inside one session, in seconds on that session's timeline."""

    model_config = ConfigDict(extra="forbid")

    session_id: str
    start_sec: float = Field(ge=0.0)
    end_sec: float = Field(ge=0.0)

    @model_validator(mode="after")
    def _ordered(self) -> Span:
        if self.end_sec < self.start_sec:
            raise ValueError(f"end_sec {self.end_sec} < start_sec {self.start_sec}")
        return self

    @property
    def duration_sec(self) -> float:
        return self.end_sec - self.start_sec


class Evidence(BaseModel):
    """Where the answer comes from. Audit-only — never sent to the system.

    `source_rows` is the load-bearing field: every released item must be
    traceable to concrete annotation rows, otherwise the ground truth is
    unfalsifiable.
    """

    model_config = ConfigDict(extra="forbid")

    session_ids: list[str] = Field(default_factory=list)
    spans: list[Span] = Field(default_factory=list)
    source_rows: list[str] = Field(default_factory=list)
    notes: str = ""


class Certificate(BaseModel):
    """Minimum evidence a human needs, after EgoSchema's temporal certificate.

    Reported as a distribution to show the corpus genuinely demands memory
    rather than single-scene perception.
    """

    model_config = ConfigDict(extra="forbid")

    n_sessions: int = Field(ge=0)
    span_seconds: float = Field(ge=0.0)
    cross_session: bool = False
    scope: EvidenceScope = EvidenceScope.SINGLE_SESSION

    @model_validator(mode="after")
    def _consistent(self) -> Certificate:
        if self.cross_session and self.n_sessions < 2:
            raise ValueError("cross_session=True requires n_sessions >= 2")
        if self.scope is EvidenceScope.CROSS_SESSION and not self.cross_session:
            raise ValueError("scope=cross_session requires cross_session=True")
        return self


class Provenance(BaseModel):
    """Which miner produced this, from which dataset, under which licence."""

    model_config = ConfigDict(extra="forbid")

    miner: str
    dataset: str
    license: str = ""
    gt_exactness: GtExactness = GtExactness.DERIVED
    miner_confidence: float = Field(default=1.0, ge=0.0, le=1.0)


class Audit(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: AuditStatus = AuditStatus.PENDING
    by: str = ""
    notes: str = ""


class NumericAnswer(BaseModel):
    """Numeric ground truth, scored with MRA rather than exact match."""

    model_config = ConfigDict(extra="forbid")

    value: float
    unit: str

    @model_validator(mode="after")
    def _mra_defined(self) -> NumericAnswer:
        # MRA divides by the target; a zero target makes relative error
        # undefined. The upstream VSI-Bench implementation does not guard
        # this, so we reject such items at authoring time instead.
        if self.value == 0.0:
            raise ValueError(
                "numeric answer of exactly 0 makes relative error undefined; "
                "rephrase the question or use a different unit"
            )
        return self


class Item(BaseModel):
    """One benchmark question."""

    model_config = ConfigDict(extra="forbid")

    item_id: str
    env_id: str
    session_ids: list[str] = Field(default_factory=list)
    axis: str
    answer_format: AnswerFormat

    question: str
    options: dict[str, str] | None = None

    answer: str | None = None
    answer_text: str | None = None
    answer_numeric: NumericAnswer | None = None
    aliases: list[str] = Field(default_factory=list)

    evidence: Evidence = Field(default_factory=Evidence)
    certificate: Certificate | None = None
    provenance: Provenance | None = None
    audit: Audit = Field(default_factory=Audit)
    bias_score: float | None = Field(default=None, ge=0.0, le=1.0)

    is_unanswerable: bool = False
    source_env_id: str | None = None  # donor env, for unanswerable controls

    @field_validator("item_id", "env_id", "axis")
    @classmethod
    def _non_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("must not be blank")
        return v

    @model_validator(mode="after")
    def _answer_matches_format(self) -> Item:
        fmt = self.answer_format
        if fmt is AnswerFormat.MCQ5:
            if not self.options:
                raise ValueError("mcq5 requires options")
            if tuple(self.options) != MCQ_LETTERS:
                raise ValueError(
                    f"mcq5 options must be exactly {MCQ_LETTERS}, got {tuple(self.options)}"
                )
            if self.options["E"] != UNANSWERABLE_TEXT:
                raise ValueError("option E text must be the canonical UNANSWERABLE_TEXT")
            if self.answer not in MCQ_LETTERS:
                raise ValueError(f"mcq5 answer must be one of {MCQ_LETTERS}, got {self.answer!r}")
            if self.is_unanswerable and self.answer != "E":
                raise ValueError("unanswerable items must have answer 'E'")
            if not self.is_unanswerable and self.answer == "E":
                raise ValueError("answer 'E' requires is_unanswerable=True")
        elif fmt is AnswerFormat.OPEN:
            if not (self.answer_text or "").strip():
                raise ValueError("open items require answer_text")
            if self.options:
                raise ValueError("open items must not carry options")
        elif fmt is AnswerFormat.NUMERIC:
            if self.answer_numeric is None:
                raise ValueError("numeric items require answer_numeric")
            if self.options:
                raise ValueError("numeric items must not carry options")
        return self

    @model_validator(mode="after")
    def _sessions_cover_evidence(self) -> Item:
        missing = set(self.evidence.session_ids) - set(self.session_ids)
        if missing:
            raise ValueError(f"evidence cites sessions outside session_ids: {sorted(missing)}")
        for span in self.evidence.spans:
            if span.session_id not in self.session_ids:
                raise ValueError(f"evidence span cites unknown session {span.session_id!r}")
        return self

    # -- the leak barrier ---------------------------------------------------

    def to_query(self) -> QueryMsg:
        """Project to the wire message, dropping every audit-only field.

        This is the *only* sanctioned path from corpus to system under test.
        """
        return QueryMsg(
            item_id=self.item_id,
            question=self.question,
            answer_format=self.answer_format,
            options=dict(self.options) if self.options else None,
            unit=self.answer_numeric.unit if self.answer_numeric else None,
        )


class SessionRef(BaseModel):
    """One recording session, plus whatever side-channels it ships with."""

    model_config = ConfigDict(extra="forbid")

    session_id: str
    order: int = Field(ge=0)
    video_path: str | None = None
    duration_sec: float | None = Field(default=None, ge=0.0)
    asr_path: str | None = None
    caption_path: str | None = None


class EnvManifest(BaseModel):
    """One environment (a home, a kitchen, a scanned room) and its sessions.

    `order` defines ingest order and must be a permutation of 0..n-1 so that
    "chronological" is unambiguous for the system under test.
    """

    model_config = ConfigDict(extra="forbid")

    env_id: str
    dataset: str = ""
    sessions: list[SessionRef] = Field(min_length=1)

    @model_validator(mode="after")
    def _orders_are_a_permutation(self) -> EnvManifest:
        ids = [s.session_id for s in self.sessions]
        if len(set(ids)) != len(ids):
            raise ValueError("duplicate session_id within env")
        orders = sorted(s.order for s in self.sessions)
        if orders != list(range(len(self.sessions))):
            raise ValueError(f"session orders must be 0..{len(self.sessions) - 1}, got {orders}")
        return self

    def ordered(self) -> list[SessionRef]:
        return sorted(self.sessions, key=lambda s: s.order)


# --------------------------------------------------------------------------
# wire schema
# --------------------------------------------------------------------------


class Capabilities(BaseModel):
    """What a system declares about itself during the handshake."""

    model_config = ConfigDict(extra="allow")

    context_mode: ContextMode
    accepts: list[str] = Field(default_factory=list)
    max_query_seconds: float | None = None


class HelloMsg(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal["hello"] = "hello"
    protocol: str


class ReadyMsg(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal["ready"] = "ready"
    system_id: str
    capabilities: Capabilities


class EnvBeginMsg(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal["env_begin"] = "env_begin"
    env_id: str
    n_sessions: int = Field(ge=1)


class IngestMsg(BaseModel):
    """One session handed over for ingestion.

    In `blind` mode every payload path is None — the system is told a session
    happened but shown nothing.
    """

    model_config = ConfigDict(extra="forbid")
    type: Literal["ingest"] = "ingest"
    session_id: str
    order: int = Field(ge=0)
    video_path: str | None = None
    duration_sec: float | None = None
    asr_path: str | None = None
    caption_path: str | None = None


class IngestDoneMsg(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal["ingest_done"] = "ingest_done"
    session_id: str
    stats: dict[str, Any] = Field(default_factory=dict)


class IngestEndMsg(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal["ingest_end"] = "ingest_end"
    env_id: str


class IngestEndAckMsg(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal["ingest_end_ack"] = "ingest_end_ack"
    memory_bytes: int | None = None
    n_records: int | None = None
    stats: dict[str, Any] = Field(default_factory=dict)


class QueryMsg(BaseModel):
    """A question, stripped of everything that would give the answer away."""

    model_config = ConfigDict(extra="forbid")
    type: Literal["query"] = "query"
    item_id: str
    question: str
    answer_format: AnswerFormat
    options: dict[str, str] | None = None
    unit: str | None = None


class AnswerMsg(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal["answer"] = "answer"
    item_id: str
    answer: str | None = None
    answer_text: str | None = None
    raw: str | None = None
    latency_ms: float | None = None
    tokens: dict[str, int] = Field(default_factory=dict)


class EnvEndMsg(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal["env_end"] = "env_end"


class ByeMsg(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal["bye"] = "bye"


class ErrorMsg(BaseModel):
    """A system may report a per-item failure without dying."""

    model_config = ConfigDict(extra="forbid")
    type: Literal["error"] = "error"
    message: str
    item_id: str | None = None
    fatal: bool = False


HarnessMsg = Annotated[
    HelloMsg | EnvBeginMsg | IngestMsg | IngestEndMsg | QueryMsg | EnvEndMsg | ByeMsg,
    Field(discriminator="type"),
]

SystemMsg = Annotated[
    ReadyMsg | IngestDoneMsg | IngestEndAckMsg | AnswerMsg | ErrorMsg,
    Field(discriminator="type"),
]
