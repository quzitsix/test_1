#!/usr/bin/env python3
"""Read HF diagnostic JSONL without loading models or media."""
import argparse
from collections import Counter
import json
from pathlib import Path


def summarize(path: Path) -> dict:
    rows = []
    incomplete = 0
    for line in path.read_text(encoding='utf-8').splitlines():
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            incomplete += 1
    generations = [r for r in rows if r.get('event') == 'generate_done']
    return {
        'file': str(path),
        'events': len(rows), 'unreadable_lines': incomplete,
        'last_event': {k: rows[-1].get(k) for k in ('event', 'env_id', 'phase', 'session_id', 'item_id')} if rows else None,
        'ready': [r for r in rows if r.get('event') == 'ready'],
        'completed_generations': len(generations),
        'completed_by_phase': dict(Counter(r.get('phase') for r in generations)),
        'notes_at_token_limit': sum(r.get('phase') == 'note' and r.get('at_token_limit', False) for r in generations),
        'seconds_completed_calls': {k: round(sum(r.get(k, 0) for r in generations), 3)
            for k in ('encode_seconds', 'transfer_seconds', 'generate_seconds', 'decode_seconds')},
        'sample_seconds': round(sum(r.get('seconds', 0) for r in rows if r.get('event') == 'sample'), 3),
        'note_chars': sum(r.get('note_chars', 0) for r in rows if r.get('event') == 'note'),
        'max_prompt_tokens': max((r.get('prompt_tokens', 0) for r in generations), default=0),
        'notice': 'Totals include all completed attempts/PIDs in this file, exclude unfinished calls, and are not full job wall time. Notes are model output, not gold.',
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('files', nargs='+', type=Path)
    args = parser.parse_args()
    for path in args.files:
        print(json.dumps(summarize(path), ensure_ascii=False, indent=2))
