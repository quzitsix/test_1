"""Guard the declared minimum Python version.

`pyproject.toml` declares `requires-python = ">=3.11"`, but development happens on
3.12. Python 3.12 rewrote f-string parsing (PEP 701), so a backslash inside an
f-string *expression* is a hard SyntaxError on 3.11 and perfectly legal on 3.12 —
a file can import cleanly on the dev box and fail to parse on the server.

That happened: `adapters/hf_vlm.py` shipped with

    f"hf_vlm:{os.path.basename(model_path.rstrip('/\\\\'))}"

and only broke when it reached a 3.11 environment.

Two checks, because they catch different things:

1. `ast.parse(..., feature_version=(3, 11))` catches *grammar* newer than 3.11
   (match statements, the `type` keyword, PEP 695 generics, and so on).
   **It does not catch the f-string case** — verified: the 3.12 tokenizer accepts
   the backslash regardless of `feature_version`, and only a genuinely old
   interpreter rejects it. So this check is necessary but not sufficient.
2. A targeted scan for backslashes inside f-string expression slots, which is the
   specific footgun that bit us and which check 1 provably misses.
"""

from __future__ import annotations

import ast
import io
import sys
import tokenize
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
MIN_VERSION = (3, 11)


def python_files() -> list[Path]:
    roots = [REPO / "meowbench", REPO / "tests", REPO / "scripts"]
    out: list[Path] = []
    for root in roots:
        if root.is_dir():
            out.extend(p for p in root.rglob("*.py") if "__pycache__" not in p.parts)
    return sorted(out)


FILES = python_files()


def test_there_are_files_to_check() -> None:
    """A silent zero-file glob would make this suite vacuously green."""
    assert len(FILES) > 10


@pytest.mark.parametrize("path", FILES, ids=lambda p: str(p.relative_to(REPO)))
def test_grammar_is_not_newer_than_minimum(path: Path) -> None:
    """No syntax newer than the oldest Python we claim to support."""
    source = path.read_text(encoding="utf-8")
    try:
        ast.parse(source, filename=str(path), feature_version=MIN_VERSION)
    except SyntaxError as exc:
        pytest.fail(
            f"{path.relative_to(REPO)}:{exc.lineno} uses syntax newer than Python "
            f"{MIN_VERSION[0]}.{MIN_VERSION[1]} (you are on "
            f"{sys.version_info.major}.{sys.version_info.minor}): {exc.msg}"
        )


def backslashes_in_fstring_expressions(source: str) -> list[int]:
    """Line numbers where an f-string's `{...}` slot contains a backslash.

    Illegal before 3.12, and this scanner has to work on *both* tokenizers,
    which disagree about what an f-string even is:

    * **3.12+** (PEP 701) emits ``FSTRING_START``, ``FSTRING_MIDDLE`` for the
      literal runs, then the ordinary tokens of each replacement field, then
      ``FSTRING_END``. So the backslash lives in a separate token and we track
      brace depth across tokens.
    * **3.11 and earlier** emit the whole f-string as a single ``STRING`` token,
      so the backslash is inside that token's text and we scan the text.

    Supporting only one of them is how this check silently passed on the machine
    it was written on and failed on the machine it was meant to protect.
    """
    if hasattr(tokenize, "FSTRING_START"):
        return _scan_pep701(source)
    return _scan_single_string_token(source)


def _tokens(source: str) -> list[tokenize.TokenInfo]:
    try:
        return list(tokenize.generate_tokens(io.StringIO(source).readline))
    except (tokenize.TokenError, IndentationError, SyntaxError):
        return []  # unparseable for another reason; the ast check reports it


def _scan_pep701(source: str) -> list[int]:
    """3.12+: f-strings are structured tokens; inspect the replacement fields."""
    hits: list[int] = []
    fstring_depth = 0
    brace_depth: list[int] = []

    for token in _tokens(source):
        name = tokenize.tok_name[token.type]

        if name == "FSTRING_START":
            fstring_depth += 1
            brace_depth.append(0)
            continue
        if name == "FSTRING_END":
            if fstring_depth:
                fstring_depth -= 1
                brace_depth.pop()
            continue
        if not fstring_depth:
            continue

        if token.type == tokenize.OP and token.string == "{":
            brace_depth[-1] += 1
            continue
        if token.type == tokenize.OP and token.string == "}":
            brace_depth[-1] = max(brace_depth[-1] - 1, 0)
            continue

        if brace_depth[-1] > 0 and "\\" in token.string:
            hits.append(token.start[0])

    return sorted(set(hits))


def _scan_string_text(text: str, line: int) -> list[int]:
    """Does this f-string literal's text carry a backslash inside a `{...}`?

    Split out from the tokenizer walk so it can be unit-tested on any
    interpreter: on 3.12 the pre-3.12 branch can never be reached from source,
    because that tokenizer does not produce the single STRING token it expects.
    """
    quote_at = min((i for i, ch in enumerate(text) if ch in "\"'"), default=len(text))
    if "f" not in text[:quote_at].lower():
        return []

    depth = 0
    index = 0
    while index < len(text):
        char = text[index]
        if char == "{":
            if index + 1 < len(text) and text[index + 1] == "{":
                index += 2  # `{{` is a literal brace, not a field
                continue
            depth += 1
        elif char == "}":
            depth = max(depth - 1, 0)
        elif char == "\\" and depth > 0:
            return [line]
        index += 1
    return []


def _scan_single_string_token(source: str) -> list[int]:
    """3.11 and earlier: the f-string is one STRING token; scan its text."""
    hits: list[int] = []
    for token in _tokens(source):
        if token.type == tokenize.STRING:
            hits.extend(_scan_string_text(token.string, token.start[0]))
    return sorted(set(hits))


@pytest.mark.parametrize("path", FILES, ids=lambda p: str(p.relative_to(REPO)))
def test_no_backslash_inside_fstring_expressions(path: Path) -> None:
    """The specific 3.11 footgun that `feature_version` cannot see."""
    hits = backslashes_in_fstring_expressions(path.read_text(encoding="utf-8"))
    assert not hits, (
        f"{path.relative_to(REPO)} has a backslash inside an f-string expression at "
        f"line(s) {hits}. That is a SyntaxError before Python 3.12 — compute the "
        f"value into a variable first."
    )


def test_the_scanner_actually_detects_the_original_bug() -> None:
    """Without this, the check above could silently match nothing forever.

    Exercises the *dispatching* entry point, i.e. whichever scanner this
    interpreter actually uses.
    """
    backslash = chr(92)
    offender = 'x = f"a:{p.rstrip(' + "'/" + backslash + backslash + "'" + ')}"'
    assert backslashes_in_fstring_expressions(offender) == [1]

    # And must not fire on the legitimate neighbours.
    assert backslashes_in_fstring_expressions('x = f"a:{p}" + "b\\\\c"') == []
    assert backslashes_in_fstring_expressions('x = "plain \\\\ string"') == []
    assert backslashes_in_fstring_expressions('x = f"{{literal braces}}"') == []
    assert backslashes_in_fstring_expressions("x = f'{a}{b}'") == []


def test_the_pre_312_scanner_handles_a_single_string_token() -> None:
    """Exercise the 3.11-shaped path using a synthetic token stream.

    The scanner cannot be driven from source on 3.12, because this tokenizer
    never emits an f-string as one STRING token — that is the whole reason there
    are two implementations. So feed it the token shape a 3.11 tokenizer *would*
    produce and check the text-scanning logic directly.

    Without this the 3.11 branch would only ever be exercised on 3.11, which is
    exactly how the first version of this file passed on 3.12 while the check it
    was protecting was broken on the server.
    """
    backslash = chr(92)
    offender = 'f"a:{p.rstrip(' + "'/" + backslash + backslash + "'" + ')}"'

    assert _scan_string_text(offender, line=1) == [1]
    assert _scan_string_text('f"a:{p}"', line=1) == []
    assert _scan_string_text('"plain ' + backslash + backslash + ' string"', line=1) == []
    assert _scan_string_text('f"{{literal}}"', line=1) == []
    # A backslash outside any replacement field is legal even in an f-string.
    assert _scan_string_text('f"tab' + backslash + backslash + 't {p}"', line=1) == []


@pytest.mark.skipif(
    not hasattr(tokenize, "FSTRING_START"),
    reason="PEP 701 tokenizer only exists on 3.12+",
)
def test_the_pep701_scanner_is_exercised_where_available() -> None:
    backslash = chr(92)
    offender = 'x = f"a:{p.rstrip(' + "'/" + backslash + backslash + "'" + ')}"'
    assert _scan_pep701(offender) == [1]
    assert _scan_pep701('x = f"a:{p}"') == []
    assert _scan_pep701('x = "plain ' + backslash + backslash + ' string"') == []
