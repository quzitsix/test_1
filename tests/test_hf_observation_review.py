"""Replay diagnostics must not present the wrong media or another attempt's notes."""
import json
from pathlib import Path

import pytest

from meowbench.artifacts import PredictionRow, SystemInfo
from meowbench.datasets.supermemory import make_plan, prepare_suite
from meowbench.media import sample_frames
from meowbench.suite import file_sha256
from test_supermemory import V1, load_script, row, video, write_rows


@pytest.fixture
def review_case(tmp_path):
    source = tmp_path/(V1+'.mp4')
    video(source)
    source_row = row()
    source_row['question'] = 'Where? <script>alert(1)</script>'
    plan = make_plan(write_rows(tmp_path, [source_row]))
    suite = prepare_suite(plan, tmp_path, tmp_path/'suite', chunk_seconds=1, max_side=128)
    item = suite.items[0]
    session = suite.envs[item.env_id].sessions[0]
    frames = sample_frames(session.video_path, n_frames=2, max_side=128)
    run = tmp_path/'run'
    run.mkdir()
    execution = {'suite_sha': suite.suite_sha+'+filtered', 'mode': 'memory',
                 'model_config': {'n_frames': 2, 'max_side': 128}, 'commit': 'test-only',
                 'media_sha256': suite.manifest['media_sha256']}
    (run/'execution.json').write_text(json.dumps(execution))
    pred = PredictionRow.from_item(item, run_id='run',
        system=SystemInfo(system_id='fake', context_mode='memory'), answer='A', raw='A')
    (run/'predictions.jsonl').write_text(pred.model_dump_json(by_alias=True), encoding='utf-8')
    common = {'pid': 10, 'env_id': item.env_id}
    trace = [{**common, 'event': 'env_begin'},
             {**common, 'event': 'sample', 'session_id': session.session_id,
              'timestamps_sec': [f.timestamp_sec for f in frames],
              'image_sizes': [list(f.image.size) for f in frames]},
             {**common, 'event': 'note', 'session_id': session.session_id,
              'text': 'Model note </pre><script>alert(2)</script>'},
             {**common, 'event': 'answer', 'item_id': item.item_id, 'text': 'A'}]
    (run/'adapter_trace.jsonl').write_text('\n'.join(json.dumps(r) for r in trace))
    return suite, run, item, session, trace


def test_replay_creates_frames_and_video_without_changing_input(review_case, tmp_path):
    suite, run, item, session, trace = review_case
    viewer = load_script('make_hf_observation_review')
    output = tmp_path/'review'/'q9.html'
    original = file_sha256(Path(session.video_path))
    viewer.render_review(suite.path, run, item.item_id, session.session_id, output)
    text = output.read_text(encoding='utf-8')
    assert '<script>alert' not in text
    assert '&lt;script&gt;alert' in text
    assert text.count('data:image/png;base64,') == 2
    assert 'context.mp4' in text and '<video controls' in text
    assert file_sha256(output.parent/'q9_media'/'context.mp4') == original
    assert file_sha256(Path(session.video_path)) == original


@pytest.mark.parametrize('mismatch', ['media', 'timestamp', 'answer', 'protected_output', 'index'])
def test_replay_rejects_wrong_evidence_before_creating_output(review_case, tmp_path, mismatch):
    suite, run, item, session, trace = review_case
    viewer = load_script('make_hf_observation_review')
    output = tmp_path/'bad'/'q9.html'
    if mismatch == 'media':
        with Path(session.video_path).open('ab') as f:
            f.write(b'changed')
    elif mismatch == 'timestamp':
        trace[1]['timestamps_sec'][0] += .1
    elif mismatch == 'answer':
        trace[-1]['text'] = 'B'
    elif mismatch == 'protected_output':
        output = suite.path/'q9.html'
    elif mismatch == 'index':
        index = json.loads((suite.path/'media_index.json').read_text())
        index[item.env_id][0]['start_sec'] += .1
        (suite.path/'media_index.json').write_text(json.dumps(index))
    (run/'adapter_trace.jsonl').write_text('\n'.join(json.dumps(r) for r in trace))
    with pytest.raises(ValueError):
        viewer.render_review(suite.path, run, item.item_id, session.session_id, output)
    assert not output.exists()


def test_replay_selects_latest_completed_attempt_not_later_partial_data():
    viewer = load_script('make_hf_observation_review')
    old = [{'pid': 1, 'env_id': 'e', 'event': 'env_begin'},
           {'pid': 1, 'env_id': 'e', 'event': 'answer', 'item_id': 'q', 'text': 'B'}]
    good = [{'pid': 2, 'env_id': 'e', 'event': 'env_begin'},
            {'pid': 2, 'env_id': 'e', 'event': 'note', 'text': 'current'},
            {'pid': 2, 'env_id': 'e', 'event': 'answer', 'item_id': 'q', 'text': 'A'}]
    partial = [{'pid': 3, 'env_id': 'e', 'event': 'env_begin'}]
    assert viewer.completed_attempt(old+good+partial, 'q', 'e') == good
