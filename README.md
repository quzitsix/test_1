# MEOWBench

A model-agnostic benchmark for **household long-term spatial memory**: can a system
watch a home over many sessions, then answer questions about where things are,
how they changed, and what the people who live there habitually do?

The system under test is a **black box**. MEOWBench hands it video, takes the
video away, and asks questions. It never inspects the system's memory, retrieved
evidence, or citations — so a long-context VLM, an external-memory research
codebase, and a test-time-training model all compete under one contract.

> **Status: M2.** Harness, protocol, scoring, reporting, and two real adapters
> (`hf_vlm` for local HuggingFace weights, `openai_compat` for any
> OpenAI-compatible endpoint including vLLM) are implemented and tested. Real
> dataset mining (`mine`, `audit`) and the LLM judge (`judge`, `debias`) are next;
> those subcommands explain what they will do and exit non-zero.
>
> To run a real model on a cluster, follow
> [`docs/M2_SERVER_RUNBOOK.md`](docs/M2_SERVER_RUNBOOK.md).

---

## The measurement

The point is not "this system scores X". It is **how much the memory buys** over
the same model with no video at all. So every suite is run in up to three
tracks, which differ *only* in what the harness stages:

| track | what the system gets | role |
|---|---|---|
| `blind` | no video, no captions — only "a session happened" | language-prior baseline |
| `memory` | video during ingestion; **revoked before any question** | the real contest |
| `oracle` | video kept for the whole environment | long-context ceiling |

**Memory Gain** = `memory` − `blind`, per axis, as a *paired* difference with a
95% interval. Paired because both tracks answer the same questions, so
differencing per item removes question difficulty from the variance; treating
them as independent samples would inflate the interval and hide real effects.

```
$ meowbench compare --run runs/memory --baseline runs/blind

axis                              n    gain  95% CI            sig
----------------------------------------------------------------------
overall                          16  +0.750  [+0.531, +0.969]  *
A3_spatial_change                 8  +0.500  [+0.130, +0.870]  *
A8_routine                        8  +1.000  [+1.000, +1.000]  *

* the paired 95% interval excludes zero
```

---

## Quick start

```bash
pip install -e .

# 1. look at a suite
meowbench suite --suite fixtures/demo

# 2. run the reference adapter in two tracks
meowbench run --suite fixtures/demo --run-id blind  --context-mode blind \
  --system "python tests/stubs/perceiving_stub.py --context-mode blind"
meowbench run --suite fixtures/demo --run-id memory --context-mode memory \
  --system "python tests/stubs/perceiving_stub.py --context-mode memory"

# 3. score and compare
meowbench report  --run runs/memory
meowbench compare --run runs/memory --baseline runs/blind

# ...or a real model (pip install -e ".[hf]")
meowbench run --suite fixtures/demo --run-id qwen-memory --context-mode memory   --handshake-timeout 900   --system "python -m meowbench.adapters.hf_vlm --model-path /path/to/Qwen2.5-VL-7B-Instruct --context-mode memory"
```

`fixtures/demo` is synthetic: a real 7 KiB H.264 file whose answer key sits in the
container metadata. A stub that reads it scores 1.0 while a blind one sits at
chance, so the whole pipeline runs in CI with no GPU, no API key, and no dataset
licence — and because the video is genuinely decodable, real VLM adapters can be
smoke-tested on it too. Their accuracy there will be near chance by design: the
fixture validates plumbing, not capability.

---

## Integrating your system

Implement one stdin/stdout loop. Your process is started **once** (a 30 GB
checkpoint should not reload per question) and driven per environment:

```
→ {"type":"hello","protocol":"meowbench/1"}
← {"type":"ready","system_id":"my-system-0.4.2",
   "capabilities":{"context_mode":"memory","accepts":["video_path","asr"]}}

→ {"type":"env_begin","env_id":"demo:home1","n_sessions":8}
→ {"type":"ingest","session_id":"s1","order":0,
   "video_path":"/staged/.../s1.mp4","duration_sec":1234.5}
← {"type":"ingest_done","session_id":"s1","stats":{}}
   … one per session, in `order` …
→ {"type":"ingest_end","env_id":"demo:home1"}
← {"type":"ingest_end_ack","memory_bytes":…,"n_records":…}
                                        ⟵ staged video is revoked here
→ {"type":"query","item_id":"it00","question":"…",
   "answer_format":"mcq5","options":{"A":"…","B":"…","C":"…","D":"…","E":"…"}}
← {"type":"answer","item_id":"it00","answer":"C","latency_ms":123}

→ {"type":"env_end"}   → {"type":"bye"}
```

Start from **`meowbench/adapters/echo_stub.py`** — under 200 lines, with three
`TODO` markers for the only parts that are yours: ingest a session, finish
ingestion, answer a question. Plus one flag: your `context_mode`.

Then self-check before spending GPU hours:

```bash
meowbench verify-adapter --system "python my_adapter.py"
```

```
[PASS] handshake
[PASS] ingest_end acknowledged
[PASS] releases media handles at ingest_end
[PASS] echoes the item_id
…
14/14 checks passed
```

### Rules that actually matter

- **Answer the question you were asked.** The `item_id` you return must match the
  one you were sent. Reordering or batching is rejected, because a mismatched
  answer would silently scramble every score after it.
- **Close your file handles before acknowledging `ingest_end`.** In `memory` mode
  the video is revoked immediately afterwards. Holding it open is detected and
  latched onto the run as `revocation_contested`, which invalidates the result.
- **Declare `oracle` if you need the video at query time.** That is a legitimate
  configuration — it measures the long-context ceiling. Declaring `memory` and
  then peeking is not.
- **Non-JSON stdout is fine.** Progress bars are tolerated and skipped.
- **Ignore message types you do not recognise**, so protocol additions do not
  break you.
- Per-item trouble? Send `{"type":"error","item_id":…,"message":…}` and stay
  alive; only that item fails. Add `"fatal":true` if you truly cannot continue.

---

## How the two-phase boundary is enforced

Most eval harnesses enforce nothing here: lmms-eval, VLMEvalKit, HELM and
OpenEQA hand over full paths and trust the wrapper, and streaming benchmarks
enforce their timestamp discipline purely by convention while the whole video
sits decoded in memory. MEOWBench does better than that, and is explicit about
where the guarantee stops.

Video is never exposed at its dataset path. It is staged into per-run scratch,
and in `memory` mode revoked at `ingest_end` by **truncating then unlinking**.
The ordering was chosen from measurement, not intuition:

| revocation strategy | with a live reader holding the file |
|---|---|
| `shutil.rmtree` | `PermissionError`; file survives and is still reopenable |
| `os.rename` of the directory | same |
| **`truncate(0)` per file** | **succeeds; a fresh `open()` yields 0 bytes** |

Two consequences worth knowing:

- **Revocable staging always copies, never hardlinks.** A hardlink shares its
  inode with the dataset original, so truncating it would zero the source video —
  verified, it really does. A regression test pins this.
- **A failed unlink is a signal, not noise.** It means the system held the video
  across the phase boundary, recorded as `revocation_contested`. On Linux a
  `/proc/<pid>/fd` audit runs as well.

Every run records an `enforcement` tier (`declared` / `revoked` / `isolated`), so
a reported number always carries the strength of the guarantee behind it.

**Residual risk, stated plainly:** a handle opened before revocation can still
return data already in its buffer (~8 KB measured). This is a strong, auditable
honesty constraint, not cryptographic isolation. Hard isolation needs a container
with mount-lifecycle control (`isolated`, not yet implemented).

---

## Scoring

| format | metric |
|---|---|
| `mcq5` | exact letter match, with tolerant extraction of real model phrasing |
| `numeric` | **MRA** (Mean Relative Accuracy), ported to match VSI-Bench exactly |
| `open` | LLM-as-a-judge (M4); reported as *pending*, never silently scored 0 |

MRA averages an indicator over ten tolerance thresholds, `linspace(0.5, 0.95, 10)`:

```
MRA = mean over θ of  1[ |pred − target| / target ≤ 1 − θ ]
```

Upstream details are preserved deliberately: the comparison is `≤`; an
unparseable prediction scores 0.0 rather than being dropped from the
denominator. One divergence — a gold value of exactly 0 makes relative error
undefined, so such items are rejected when authored instead of dividing by zero.

Three outcomes a single accuracy number would blur are kept apart:

- **committed and wrong** — the system chose, and chose badly;
- **abstained** — it picked option `E`, correct only on unanswerable controls;
- **errored** — the harness never got an answer (timeout, crash, malformed).

Errors stay in the denominator by default. Dropping them would let a flaky system
raise its own score by failing selectively; `--exclude-errors` exists for
diagnosis and the report says which convention was used.

Every MCQ item carries the same option `E` text corpus-wide, so the unanswerable
class cannot be spotted from wording, and unanswerable controls are built by
transplanting a question from a *different* environment.

---

## Artifacts

```
runs/<run_id>/
  predictions.jsonl   one self-contained record per item
  results.sqlite      operational DB: resume, dedup, cost accounting
```

`predictions.jsonl` **denormalises the question, gold answer, and evidence into
each row**. That costs a few KB per item and buys re-judging, re-aggregating, or
handing results to a third party with nothing else. OpenEQA stores only
`{question_id, answer}` and must re-key its dataset to score, so editing the
dataset makes old results unjudgeable — the failure this avoids.

Judgments likewise record the 1–5 → [0,1] normalisation **inline**, plus the
rubric's `sha256`. A bare stored `4` is meaningless without the source that
rescales it, and a quietly edited rubric is the most common cause of
irreproducible LLM-judge numbers.

Runs are resumable by construction: predictions commit per item, so re-invoking
the same `run_id` skips what already succeeded, and a fully-answered environment
skips ingestion entirely rather than re-reading hours of video.

---

## Question axes

Names deliberately reuse existing benchmarks' terms so results can be positioned
against prior work.

| id | axis | aligned naming |
|---|---|---|
| `A1` | static object location | OpenEQA *object localization* |
| `A2` | metric space (direction, distance, size, room area) | VSI-Bench's eight tasks |
| `A3` | object relocation / last-known location | Ego4D VQ2D–VQ3D framing |
| `A4` | state change | OpenEQA *object state recognition* |
| `A5` | appearance attributes | OpenEQA *attribute recognition* |
| `A6` | temporal order | VSI-Bench *appearance order* |
| `A7` | event recall | EgoLifeQA *EventRecall* |
| `A8` | routine / habit | EgoLifeQA *HabitInsight*; MEMORA `SHABIT`/`SROUTINE` |
| `A9` | preference | MEMORA `SPREF` |
| `A10` | person relations | EgoLifeQA *RelationMap* |
| `A11` | counting | VSI-Bench *object counting* |
| `A12` | unanswerable controls | MEMORA *E-correct*; LongMemEval abstention |

Two cross-cutting labels are carried per item rather than being axes:
**evidence scope** (`single_scene` … `whole_env`) and a **memory certificate**
(minimum sessions and time span needed), whose distribution is reported as
evidence that the suite genuinely demands long-horizon memory. `A8`/`A9`
additionally require `cross_session` with at least two sessions.

---

## Layout

```
meowbench/
  schema.py         corpus + wire contracts; Item.to_query() is the leak barrier
  suite.py          freeze / load / checksum-verify a release
  runner.py         the run loop, per environment
  store.py          SQLite: resume, re-judging, cost
  artifacts.py      self-contained JSONL records
  conformance.py    verify-adapter checks
  cli.py            command line
  adapters/
    protocol.py     long-lived subprocess driver, timeouts, crash isolation
    staging.py      staging + revocation, enforcement tiers
    echo_stub.py    reference adapter — copy this
    hf_vlm.py       local HuggingFace VLM, weights loaded in-process
    openai_compat.py  any /v1/chat/completions endpoint, incl. local vLLM
  scoring/
    deterministic.py  MCQ + MRA
    aggregate.py      per-axis cells, Wilson intervals, Memory Gain
  media.py          frame sampling via PyAV (no ffmpeg binary needed)
tests/              196 tests, no GPU / API key / dataset required
fixtures/demo/      synthetic suite for CI
docs/               M2 server runbook
```

`Item.to_query()` is the only sanctioned path from corpus to system, and it drops
every audit-only field. Leaking `evidence` would tell a system exactly where to
look; leaking `answer` would void the benchmark. A test asserts the projection
contains nothing else.

---

## Data and licensing

MEOWBench mines questions from **existing annotations of public datasets** rather
than commissioning large-scale labelling, and pairs that with human review. None
of the candidate sources permits redistributing frames, so a release ships:

> `items.jsonl` + `envs.jsonl` (text, keyed by `(dataset, video_id, time range)`)
> plus the mining and evaluation code — **no frames, no clips, no copies of
> upstream annotation files.**

Users obtain the media themselves from each dataset's official channel. Because
several sources are non-commercial, **a benchmark assembled this way inherits a
non-commercial constraint**.

Contamination is treated as a first-class problem, not a footnote: several strong
candidate datasets appear verbatim in public video instruction-tuning mixes, and
some of the models under test do not disclose their training data at all. The
planned mitigations are low-contamination secondary splits, a preference for
cross-session questions (clip-level tuning data does not contain cross-session
conclusions), an exclusion list for splits already consumed by published
benchmarks, and shortcut pruning with blind and fine-tuned-blind diagnostics.

---

## Development

```bash
pip install -e ".[dev]"
pytest -q
```

The suite spawns real subprocesses and asserts on adversarial behaviour — held
handles, mismatched item ids, timeouts, crashes mid-run — because those are the
failures that quietly corrupt results rather than announcing themselves.
