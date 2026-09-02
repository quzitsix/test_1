"""A stub that actually uses the video: reads a gold letter embedded in it.

Stands in for a model that genuinely perceives. In blind mode it sees no path
and must guess; in memory mode it can read during ingest and remember; in
oracle mode it can read at query time. This is what makes the three-track
table produce a non-zero Memory Gain.
"""
import argparse, json, os, sys

ap = argparse.ArgumentParser()
ap.add_argument("--context-mode", default="memory")
ap.add_argument("--forget", action="store_true", help="do not remember across ingest_end")
a = ap.parse_args()

def send(o):
    sys.stdout.write(json.dumps(o) + "\n"); sys.stdout.flush()

remembered = {}
last_path = None

for line in sys.stdin:
    line = line.strip()
    if not line:
        continue
    m = json.loads(line)
    t = m["type"]
    if t == "hello":
        send({"type": "ready", "system_id": f"oracle_stub-{a.context_mode}",
              "capabilities": {"context_mode": a.context_mode}})
    elif t == "ingest":
        p = m.get("video_path")
        last_path = p
        if p and not a.forget:
            try:
                with open(p, "rb") as fh:
                    body = fh.read().decode("utf-8", "replace")
                for chunk in body.split("|"):
                    if chunk.startswith("ANS:"):
                        k, v = chunk[4:].split("=")
                        remembered[k] = v
            except OSError:
                pass
        send({"type": "ingest_done", "session_id": m["session_id"]})
    elif t == "ingest_end":
        send({"type": "ingest_end_ack", "n_records": len(remembered)})
    elif t == "query":
        key = m["item_id"]
        ans = remembered.get(key)
        if ans is None and last_path:          # oracle: re-read at query time
            try:
                with open(last_path, "rb") as fh:
                    body = fh.read().decode("utf-8", "replace")
                for chunk in body.split("|"):
                    if chunk.startswith("ANS:"):
                        k, v = chunk[4:].split("=")
                        if k == key:
                            ans = v
            except OSError:
                pass
        send({"type": "answer", "item_id": key, "answer": ans or "A"})
    elif t == "bye":
        break
