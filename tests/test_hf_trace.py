"""Diagnostic output must not change prompts, media access or answers."""
import json
from types import SimpleNamespace

import pytest
from PIL import Image

from meowbench.adapters.hf_vlm import HFVLMAdapter
from meowbench.media import Frame


def adapter(trace_file=None, mode='memory'):
    torch = pytest.importorskip('torch')
    a = HFVLMAdapter.__new__(HFVLMAdapter)  # No weights/downloads/devices.
    a.context_mode = mode
    a._torch = torch
    a._input_device = torch.device('cpu')
    a._trace_file = trace_file
    a._trace_env_id = None
    a._trace_context = {}
    a._notes = []
    a._session_paths = []
    a._oracle_frames = None
    a._n_frames = 2
    a._max_side = 64
    a._max_oracle_frames = 8
    a._note_max_new_tokens = 2
    a._max_new_tokens = 2
    a._temperature = 0
    a.messages = []

    def encode(messages, images):
        a.messages.append(messages)
        return {'input_ids': torch.tensor([[1, 2, 3]]),
                'pixel_values': torch.ones((max(1, len(images)), 2))}

    def generate(input_ids, **kwargs):
        return torch.cat([input_ids, torch.tensor([[7, 8]])], dim=1)

    a._encode = encode
    a._model = SimpleNamespace(dtype=torch.float32,
        config=SimpleNamespace(is_encoder_decoder=False), generate=generate)
    a._processor = SimpleNamespace(batch_decode=lambda *a, **k: ['A'])
    return a


def query():
    return {'item_id': 'i1', 'answer_format': 'mcq', 'question': 'Where?',
            'options': {'A': 'table', 'B': 'shelf'}}


def test_trace_is_observation_only_and_flushes_notes(tmp_path, monkeypatch):
    frames = [Frame(7.5, Image.new('RGB', (32, 24))),
              Frame(22.5, Image.new('RGB', (32, 24)))]
    monkeypatch.setattr('meowbench.adapters.hf_vlm.sample_frames', lambda *a, **k: frames)
    path = tmp_path/'trace.jsonl'
    plain, traced = adapter(), adapter(path)
    for a in (plain, traced):
        a.on_env_begin('e1', 1)
        assert a.ingest({'session_id': 's1', 'order': 0, 'video_path': 'video.mp4'}) == {
            'frames': 2, 'note_chars': 1}
        assert a.answer(query()) == {'answer': 'A', 'raw': 'A'}
    assert plain.messages == traced.messages
    rows = [json.loads(x) for x in path.read_text().splitlines()]
    sample = next(r for r in rows if r['event'] == 'sample')
    assert sample['timestamps_sec'] == [7.5, 22.5]
    assert sample['image_sizes'] == [[32, 24], [32, 24]]
    assert sample['session_id'] == 's1'
    assert next(r for r in rows if r['event'] == 'note')['text'] == 'A'
    generations = [r for r in rows if r['event'] == 'generate_done']
    assert [r['n_images'] for r in generations] == [2, 0]
    assert all(r['output_tokens'] == 2 and r['at_token_limit'] for r in generations)
    assert all(r['prompt_tokens'] == 3 for r in generations)
    assert all(r['generate_seconds'] >= 0 and r['encode_seconds'] >= 0 for r in generations)
    assert not any('gold_answer' in r or 'pixel_values' in r for r in rows)

    before = path.read_bytes()
    traced.on_env_begin('e2', 0)
    traced.answer(query())
    assert path.read_bytes().startswith(before)  # appends, never overwrites
    assert 'Here are your own notes' not in str(traced.messages[-1])
    assert json.loads(path.read_text().splitlines()[-1])['env_id'] == 'e2'


def test_oracle_trace_does_not_decode_again_or_keep_old_environment(tmp_path, monkeypatch):
    calls = []

    def sample(path, **kwargs):
        calls.append(path)
        return [Frame(4.0, Image.new('RGB', (24, 24)))]

    monkeypatch.setattr('meowbench.adapters.hf_vlm.sample_frames', sample)
    path = tmp_path/'oracle.jsonl'
    a = adapter(path, 'oracle')
    a.on_env_begin('first', 1)
    a.ingest({'video_path': '/stage/clip-one.mp4'})
    assert a.on_ingest_end()['frames'] == 1
    a.answer(query())
    a.answer(query())
    assert len(calls) == 1
    a.on_env_begin('second', 1)
    a.ingest({'video_path': '/stage/clip-two.mp4'})
    a.on_ingest_end()
    rows = [json.loads(x) for x in path.read_text().splitlines()]
    samples = [r for r in rows if r['event'] == 'sample']
    assert [(r['env_id'], r['session_id']) for r in samples] == [
        ('first', 'clip-one'), ('second', 'clip-two')]
