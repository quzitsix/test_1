"""Protocol wiring only; the learner is owned/tested by the sibling TTT_frame repo."""

from unittest.mock import Mock

import pytest

pytest.importorskip("ttt_frame")

from meowbench.adapters.base import build_prompt
from meowbench.adapters.ttt_lact import LoRAVideoAdapter
from meowbench.schema import IngestEndAckMsg


def test_bridge_passes_only_media_then_only_question_and_resets():
    engine = Mock()
    engine.ingest_video.return_value = {"frames": 4}
    engine.finish_ingest.return_value = {"memory_bytes": 64, "n_records": 0, "stats": {}}
    engine.answer.return_value = "B"
    adapter = LoRAVideoAdapter(None, engine=engine)
    adapter.on_env_begin("home1", 1)
    assert adapter.ingest({"video_path": "/staged/s1.mp4", "session_id": "s1"}) == {"frames": 4}
    engine.ingest_video.assert_called_once_with("/staged/s1.mp4")
    IngestEndAckMsg.model_validate({"type": "ingest_end_ack", **adapter.on_ingest_end()})
    query = dict(question="Where is mug?", answer_format="mcq5", options={"A": "sink", "B": "shelf"})
    assert adapter.answer(query)["answer"] == "B"
    engine.answer.assert_called_once_with(build_prompt(query), use_memory=True)
    adapter.on_env_end()
    adapter.on_env_begin("home2", 1)
    assert engine.reset.call_count == 3


def test_blind_never_ingests_and_base_read_control_disables_parameters():
    engine = Mock()
    engine.answer.return_value = "shelf"
    adapter = LoRAVideoAdapter(None, engine=engine, context_mode="blind", read_base=True)
    adapter.ingest({"video_path": "/even-if-supplied.mp4"})
    engine.ingest_video.assert_not_called()
    query = dict(question="Where is mug?", answer_format="open")
    assert adapter.answer(query)["answer_text"] == "shelf"
    engine.answer.assert_called_with(build_prompt(query), use_memory=False)


def test_missing_video_and_oracle_are_not_silently_accepted():
    with pytest.raises(ValueError, match="oracle"):
        LoRAVideoAdapter(None, engine=Mock(), context_mode="oracle")
    adapter = LoRAVideoAdapter(None, engine=Mock())
    with pytest.raises(ValueError, match="video_path"):
        adapter.ingest({"video_path": None})
