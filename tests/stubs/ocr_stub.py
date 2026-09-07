#!/usr/bin/env python3
"""A stub that recovers answers from video *pixels*, for the `probe` suite.

`perceiving_stub.py` reads the answer key out of the container's `comment` tag.
That is enough to prove the three tracks are wired up, but it cannot exercise
`fixtures/probe`, whose container metadata is deliberately empty: there, the
answers exist only as rendered text in the frames. A stub that decodes nothing
would score chance on every track and the suite's whole point — that a
*collapsed* Memory Gain means the measurement chain is broken — would be
untestable in CI.

So this stub actually looks at the pixels:

    blind    - never given a path       -> fixed guess, lands at chance
    memory   - reads during ingest, then the media is revoked
               -> can only be right if it *remembered*
    oracle   - may re-read at query time
               -> upper bound

WHY NOT A REAL OCR ENGINE

pytesseract needs a system binary and easyocr downloads weights; both would
make this test unrunnable in the CI matrix that is exactly where it earns its
keep. Instead we exploit an asymmetry that only holds for a synthetic fixture:
the *renderer* is known. Pillow's `load_default(size=...)` is deterministic, so
each candidate glyph can be re-rendered and matched against the frame. Measured
on the committed videos, char-by-char rendering is bit-identical to
full-string rendering (max abs difference 0 across the line), so glyph
templates need no kerning model.

Matching is per-glyph rather than per-line, and that choice is load-bearing.
The obvious approach — re-render each candidate `(object, placement)` line and
correlate whole crops — requires a *candidate list*, i.e. knowing the object
vocabulary in advance. Then "the object was never shown" would be a lookup in a
hardcoded list rather than an observation, and the unanswerable controls would
be fake: the stub would score them right without ever having watched. Reading
glyph by glyph means the recovered vocabulary comes only from what was on
screen, so `E` is genuinely derived from *not having seen* the object.

Robustness to the fixture's H.264 crf=30 comes from three things:

* a binarising ink threshold, not exact pixel equality — the text is
  near-black or white on a flat background, so thresholding survives ringing
  that would wreck a raw difference;
* per-glyph normalised correlation on a fixed 20x20 grid, which is invariant to
  the brightness and contrast drift compression introduces;
* a geometry gate (advance width, ink height, baseline offset) that discards
  most candidates before correlating, so the score only has to separate glyphs
  of near-identical shape.

Measured on all six committed probe videos, three sampled frames each: every
body line and every session number is recovered exactly, with a worst-case
top-vs-runner-up correlation margin of 0.033 on body glyphs and 0.126 on the
session digits.

The geometry constants below mirror `render_frame` in
`scripts/make_probe_fixture.py`. They are duplicated rather than imported
because that module exposes its layout only as literals inside the drawing
call, and because what is needed here is glyph templates rather than whole
frames. If the generator's layout changes, this stub stops recovering anything
and `tests/test_probe_fixture.py` fails on a collapsed gain — which is the
intended failure mode, not a silent one.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from meowbench.media import MediaError, sample_frames  # noqa: E402
from meowbench.schema import UNANSWERABLE_TEXT  # noqa: E402

#: Canvas and text layout, mirroring the fixture generator's `render_frame`.
WIDTH, HEIGHT = 640, 480
HEADER_X, HEADER_Y = 14, 14
HEADER_SIZE = 34
BODY_X, BODY_TOP = 26, 96
BODY_SIZE = 30
SLOT_STEP = 104
PLACE_DY = 38
LINE_H = 38
FOOTER_H = 34

#: Body text is near-black on a light background; the header is white on dark
#: blue. One threshold serves both, with the comparison inverted for the header.
INK_LEVEL = 170

#: Body lines are rendered upper-case only. Restricting the alphabet is not a
#: convenience: in this font `I` and `l` rasterise to the same bitmap, so
#: offering both makes the correlation genuinely ambiguous and the margin
#: meaningless.
UPPER = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
DIGITS = "0123456789"

#: Side of the square grid every glyph is resampled onto before correlating.
GLYPH_GRID = 20

#: Tolerances for the geometry gate, in source pixels.
MAX_WIDTH_SLACK = 2
MAX_HEIGHT_SLACK = 2
MAX_BASELINE_SLACK = 3

#: Frames sampled per session. The generator draws identical content in every
#: frame bar a clock, so one would do; three lets a majority vote absorb a
#: single badly-compressed frame.
FRAMES_PER_SESSION = 3

#: What to say with nothing to go on. Fixed rather than random so the blind
#: track is a clean, reproducible chance baseline.
DEFAULT_GUESS = "A"

_SESSION_RE = re.compile(r"session\s+(\d+)\s+of", re.IGNORECASE)
_OBJECT_RE = re.compile(r"where was the\s+(.+?)\s*\?", re.IGNORECASE)
_RECENT_RE = re.compile(r"\s*in the most recent session$", re.IGNORECASE)


def send(obj: dict) -> None:
    sys.stdout.write(json.dumps(obj) + "\n")
    sys.stdout.flush()


def _norm(text: str) -> str:
    """Collapse to bare upper-case letters.

    Word spacing is dropped on purpose: glyph segmentation recovers ink runs,
    and inter-word gaps are not reliably distinguishable from wide intra-word
    ones at this size. Comparing on letters alone sidesteps the question.
    """
    return re.sub(r"[^A-Z]", "", text.upper())


#: The generator writes "<OBJECT> IS ON"; this is that suffix, normalised.
OBJECT_SUFFIX = _norm(" IS ON")
#: Placements are rendered as "THE <PLACE>" but the options say just "<place>".
PLACE_PREFIX = _norm("THE")


# -- glyph machinery --------------------------------------------------------

_fonts: dict[int, object] = {}
_templates: dict[tuple[int, str], dict] = {}


def _font(size: int):
    if size not in _fonts:
        from PIL import ImageFont

        _fonts[size] = ImageFont.load_default(size=size)
    return _fonts[size]


def _ink_runs(mask) -> list[tuple[int, int]]:
    """Contiguous column ranges containing ink, i.e. candidate glyph boxes."""
    columns = mask.any(axis=0)
    runs: list[tuple[int, int]] = []
    start = None
    for x, filled in enumerate(columns):
        if filled and start is None:
            start = x
        elif not filled and start is not None:
            runs.append((start, x))
            start = None
    if start is not None:
        runs.append((start, len(columns)))
    return runs


def _glyph_vector(block):
    """Mean-centred unit vector for one glyph box, plus its ink extent.

    Cropping to the ink rows before resampling is what makes the descriptor
    independent of where in the band the glyph sits; the row offset is returned
    separately so the caller can still gate on baseline position.

    Returns `solid=True` for a glyph whose box is uniformly inked — `I` is a
    bare vertical bar, so once cropped to its own extent it has zero variance
    and the correlation is undefined (0/0). Treating that as unreadable silently
    deleted every `I`, turning "RED MUG IS ON" into "REDMUGSON" and matching
    nothing. Such a glyph is identifiable only by its dimensions, so the flag
    tells the caller to decide on geometry instead.
    """
    import numpy as np
    from PIL import Image

    rows = np.nonzero(block.any(axis=1))[0]
    if not len(rows):
        return None
    top, bottom = int(rows[0]), int(rows[-1]) + 1
    scaled = Image.fromarray((block[top:bottom] * 255).astype(np.uint8)).resize(
        (GLYPH_GRID, GLYPH_GRID), Image.BILINEAR
    )
    values = np.asarray(scaled, dtype=np.float32)
    values -= values.mean()
    norm = float(np.sqrt((values * values).sum()))
    if norm <= 1e-6:
        return values, top, bottom, True
    return values / norm, top, bottom, False


def _glyph_templates(size: int, alphabet: str) -> dict:
    """Per-character descriptors, rendered once and cached.

    Each glyph is drawn at a fixed origin inside a generous canvas so that the
    recorded baseline offset is comparable with one measured in a real frame.
    """
    key = (size, alphabet)
    if key in _templates:
        return _templates[key]

    import numpy as np
    from PIL import Image, ImageDraw

    origin = size * 2
    built: dict[str, tuple] = {}
    for char in alphabet:
        canvas = Image.new("L", (size * 3, size * 3), 255)
        ImageDraw.Draw(canvas).text((origin, origin), char, font=_font(size), fill=0)
        mask = np.asarray(canvas) < 128
        runs = _ink_runs(mask)
        if not runs:
            continue
        left, right = runs[0][0], runs[-1][1]
        descriptor = _glyph_vector(mask[:, left:right])
        if descriptor is None:
            continue
        vector, top, bottom, solid = descriptor
        built[char] = (vector, right - left, bottom - top, top - origin, solid)
    _templates[key] = built
    return built


def read_line(mask, size: int, alphabet: str) -> str:
    """Transcribe one binarised text band, or "" if nothing is legible.

    An unclassifiable run yields "?" so the caller can reject the whole line
    rather than record a plausible-looking half-read fact.
    """
    out: list[str] = []
    templates = _glyph_templates(size, alphabet)
    for left, right in _ink_runs(mask):
        descriptor = _glyph_vector(mask[:, left:right])
        if descriptor is None:
            continue
        vector, top, bottom, solid = descriptor
        width = right - left
        height = bottom - top
        best_score, best_char = -2.0, "?"
        for char, (t_vector, t_width, t_height, t_top, t_solid) in templates.items():
            if (
                abs(t_width - width) > MAX_WIDTH_SLACK
                or abs(t_height - height) > MAX_HEIGHT_SLACK
                or abs(t_top - top) > MAX_BASELINE_SLACK
            ):
                continue
            if solid or t_solid:
                # No shape to correlate. Geometry already gated width, height
                # and baseline, which is all that distinguishes a solid bar.
                if solid and t_solid and best_score < 0.0:
                    best_score, best_char = 0.0, char
                continue
            score = float((vector * t_vector).sum())
            if score > best_score:
                best_score, best_char = score, char
        out.append(best_char)
    return "".join(out)


# -- frame reading ----------------------------------------------------------


def _read_frame(image, env_id: str) -> tuple[int | None, dict[str, str]]:
    """Recover (session number, {object: placement}) from one frame.

    `env_id` is needed only to locate the session digits: the generator renders
    the header as "<env_id>   SESSION NN", so the digits begin one measured
    prefix-width in. Cropping there rather than transcribing the whole header
    avoids having to model the lower-case environment name, whose glyphs include
    the `I`/`l` collision this matcher deliberately sidesteps.
    """
    import numpy as np

    if image.size != (WIDTH, HEIGHT):
        # Absolute layout constants cannot survive a rescale. Recovering
        # nothing surfaces as n_records=0 plus a report note, which is a
        # visible failure rather than a quietly wrong answer.
        return None, {}

    gray = np.asarray(image.convert("L"))

    session = None
    digits_x = HEADER_X + int(_font(HEADER_SIZE).getlength(f"{env_id}   SESSION "))
    if 0 <= digits_x < WIDTH:
        # The header is white on dark blue, so ink is *above* the threshold
        # here. The band must start exactly at the text origin: shifting it up
        # even a few pixels moves every measured baseline and the geometry gate
        # then rejects every candidate digit.
        digits = read_line(
            gray[HEADER_Y : HEADER_Y + LINE_H + 6, digits_x:WIDTH] > INK_LEVEL,
            HEADER_SIZE,
            DIGITS,
        )
        if digits.isdigit():
            session = int(digits)

    facts: dict[str, str] = {}
    slot = 0
    while True:
        top = BODY_TOP + SLOT_STEP * slot
        if top + PLACE_DY + LINE_H > HEIGHT - FOOTER_H:
            break
        slot += 1
        object_line = read_line(
            gray[top : top + LINE_H, :] < INK_LEVEL, BODY_SIZE, UPPER
        )
        place_line = read_line(
            gray[top + PLACE_DY : top + PLACE_DY + LINE_H, :] < INK_LEVEL,
            BODY_SIZE,
            UPPER,
        )
        if "?" in object_line or "?" in place_line:
            continue
        if not object_line.endswith(OBJECT_SUFFIX):
            continue
        name = object_line[: -len(OBJECT_SUFFIX)]
        place = place_line
        if place.startswith(PLACE_PREFIX):
            place = place[len(PLACE_PREFIX) :]
        if name and place:
            facts[name] = place
    return session, facts


def read_session(path: str, env_id: str, fallback_session: int | None = None):
    """Decode a session video into (session number, facts, frames decoded).

    Facts are majority-voted across frames: a single frame whose compression
    happened to break one glyph should not be able to install a wrong location.
    """
    try:
        frames = sample_frames(path, n_frames=FRAMES_PER_SESSION, max_side=None)
    except (MediaError, OSError):
        # A revoked (truncated) file lands here. That is the memory track
        # working as designed, not an error worth reporting.
        return None, {}, 0

    votes: dict[str, dict[str, int]] = {}
    session_votes: dict[int, int] = {}
    for frame in frames:
        try:
            session, facts = _read_frame(frame.image, env_id)
        except Exception:  # noqa: BLE001 - one bad frame must not kill ingest
            continue
        if session is not None:
            session_votes[session] = session_votes.get(session, 0) + 1
        for name, place in facts.items():
            votes.setdefault(name, {})
            votes[name][place] = votes[name].get(place, 0) + 1

    agreed = {
        name: max(places.items(), key=lambda kv: kv[1])[0] for name, places in votes.items()
    }
    number = (
        max(session_votes.items(), key=lambda kv: kv[1])[0]
        if session_votes
        else fallback_session
    )
    return number, agreed, len(frames)


# -- answering --------------------------------------------------------------


def _abstain_letter(options: dict[str, str] | None) -> str:
    """The option that says the answer is not available, read off the options."""
    wanted = _norm(UNANSWERABLE_TEXT)
    for letter, text in (options or {}).items():
        if _norm(text) == wanted:
            return letter
    return DEFAULT_GUESS


def _place_letter(place: str, options: dict[str, str] | None) -> str | None:
    """The option naming `place`. Never hardcoded: always read from the query."""
    for letter, text in (options or {}).items():
        if _norm(text) == place:
            return letter
    for letter, text in (options or {}).items():
        normalised = _norm(text)
        if normalised and (place.endswith(normalised) or normalised.endswith(place)):
            return letter
    return None


def parse_question(question: str) -> tuple[str, int | None]:
    """Extract (normalised object, explicit session number or None)."""
    match = _OBJECT_RE.search(question)
    phrase = match.group(1) if match else ""
    phrase = _RECENT_RE.sub("", phrase)
    session = _SESSION_RE.search(question)
    return _norm(phrase), (int(session.group(1)) if session else None)


class Memory:
    """What was recovered during ingest, keyed by session number."""

    def __init__(self) -> None:
        self.sessions: dict[int, dict[str, str]] = {}
        self.paths: list[str] = []

    def clear_recovered(self) -> None:
        self.sessions.clear()

    def add(self, session: int | None, facts: dict[str, str]) -> None:
        if session is None or not facts:
            return
        self.sessions.setdefault(session, {}).update(facts)

    @property
    def n_records(self) -> int:
        return sum(len(facts) for facts in self.sessions.values())

    def memory_bytes(self) -> int:
        return len(
            json.dumps({str(k): v for k, v in self.sessions.items()}).encode("utf-8")
        )

    def lookup(self, name: str, session: int | None) -> str | None:
        """The recorded placement, or None if this object was never observed.

        None is the honest "unanswerable" signal, and it is reachable only by
        having watched every session and not found the object.
        """
        if not self.sessions:
            return None
        target = session if session is not None else max(self.sessions)
        found = self.sessions.get(target, {}).get(name)
        if found is not None:
            return found
        # Asked about a session whose read failed: fall back to the most recent
        # session that did mention the object, rather than claiming absence.
        for number in sorted(self.sessions, reverse=True):
            if name in self.sessions[number]:
                return self.sessions[number][name]
        return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--context-mode", default="memory", choices=("blind", "memory", "oracle"))
    parser.add_argument(
        "--forget",
        action="store_true",
        help="drop what was read during ingest, to test that revocation bites",
    )
    args = parser.parse_args(argv)

    memory = Memory()
    reread = False
    env_id = ""

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        msg = json.loads(line)
        kind = msg["type"]

        if kind == "hello":
            send(
                {
                    "type": "ready",
                    "system_id": f"ocr_stub-{args.context_mode}",
                    "capabilities": {"context_mode": args.context_mode},
                }
            )
        elif kind == "env_begin":
            memory = Memory()
            reread = False
            env_id = msg.get("env_id", "")
        elif kind == "ingest":
            path = msg.get("video_path")
            frames = 0
            if path:
                memory.paths.append(path)
                order = msg.get("order")
                session, facts, frames = read_session(
                    path, env_id, None if order is None else int(order) + 1
                )
                memory.add(session, facts)
            send(
                {
                    "type": "ingest_done",
                    "session_id": msg["session_id"],
                    "stats": {"frames": frames},
                }
            )
        elif kind == "ingest_end":
            if args.forget:
                memory.clear_recovered()
            send(
                {
                    "type": "ingest_end_ack",
                    "n_records": memory.n_records,
                    "memory_bytes": memory.memory_bytes(),
                }
            )
        elif kind == "query":
            options = msg.get("options") or {}
            name, session = parse_question(msg.get("question", ""))
            if not memory.sessions and memory.paths and not reread:
                # Oracle: the payload was never revoked, so reading it now is
                # legitimate. Done once per environment, not once per question.
                reread = True
                for index, path in enumerate(memory.paths):
                    late_session, late_facts, _ = read_session(path, env_id, index + 1)
                    memory.add(late_session, late_facts)
            place = memory.lookup(name, session) if name else None
            if place is None:
                answer = _abstain_letter(options) if memory.sessions else DEFAULT_GUESS
            else:
                answer = _place_letter(place, options) or DEFAULT_GUESS
            send({"type": "answer", "item_id": msg["item_id"], "answer": answer})
        elif kind == "env_end":
            pass
        elif kind == "bye":
            return 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
