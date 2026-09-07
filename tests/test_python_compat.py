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

    Illegal before 3.12. Implemented against the 3.12+ tokenizer, which (per
    PEP 701) no longer emits an f-string as one STRING token: it emits
    FSTRING_START, then FSTRING_MIDDLE for the literal runs, then the ordinary
    tokens of each replacement field, then FSTRING_END. So we track f-string
    nesting and brace depth and look at the tokens *inside* the braces — which is
    also why a naive scan of the whole string literal found nothing.
    """
    hits: list[int] = []
    try:
        tokens = list(tokenize.generate_tokens(io.StringIO(source).readline))
    except (tokenize.TokenError, IndentationError, SyntaxError):
        return hits  # unparseable for another reason; the ast check reports it

    fstring_depth = 0
    brace_depth: list[int] = []  # brace nesting per active f-string

    for token in tokens:
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

        # Inside a replacement field: any token carrying a backslash is fatal
        # before 3.12, whether it is a nested string or a line continuation.
        if brace_depth[-1] > 0 and "\\" in token.string:
            hits.append(token.start[0])

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
    """Without this, the check above could silently match nothing forever."""
    backslash = chr(92)
    offender = 'x = f"a:{p.rstrip(' + "'/" + backslash + backslash + "'" + ')}"'
    assert backslashes_in_fstring_expressions(offender) == [1]

    # And must not fire on the legitimate neighbours.
    assert backslashes_in_fstring_expressions('x = f"a:{p}" + "b\\\\c"') == []
    assert backslashes_in_fstring_expressions('x = "plain \\\\ string"') == []
    assert backslashes_in_fstring_expressions('x = f"{{literal braces}}"') == []
    assert backslashes_in_fstring_expressions("x = f'{a}{b}'") == []
