"""Native QA contract and real-data preparation regression checks (no model downloads)."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import pytest
from meowbench.adapters.base import build_prompt
from meowbench.artifacts import PredictionRow, SystemInfo
from meowbench.datasets.supermemory import make_plan, prepare_suite, read_plan
from meowbench.datasets.video_windows import render_window
from meowbench.media import sample_frames
from meowbench.schema import Item
from meowbench.scoring.aggregate import score_prediction
from meowbench.scoring.deterministic import extract_mcq_letter

V1 = 'Person_1_session_1_01012026_glasses_1'
V2 = 'Person_1_session_2_01022026_glasses_1'
V3 = 'Person_1_session_3_01032026_glasses_1'
UNKNOWN = 'This question can not be answered.'


def native_item(**kwargs):
    params = dict(item_id='native-1',env_id='env',axis='visual',answer_format='mcq',
        question='Where is it?',options={'A':'sink','B':UNKNOWN,'C':'table','D':'shelf'},
        answer='C',abstention_option='B')
    params.update(kwargs)
    return Item(**params)


def row(qid=1, video=V1, stamp=1000, *, evidence_video=None, evidence_stamp=None):
    return {'question_id':qid,'question':'Where was the cup put?',
        'choices':['table', UNKNOWN, 'sink', 'cupboard'], 'correct_option_index':0,
        'correct_answer':'table','choice_types':['correct','incorrect','incorrect','vague'],
        'subject':1,'is_answerable':True,'video_ids':[V1,V2,V3],
        'metadata':{'skill':'object_location_memory','primary_video_id':video,'primary_video_start_time':stamp},
        'question_evidence':{'video_id':video,'modalities':['Video'],
            'time_spans':[{'start_time':2,'end_time':3,'video_id':video}]},
        'answer_evidence':{'is_answerable':True,'text':'table','evidence_list':[
            {'video_id':evidence_video or video,'start_time':evidence_stamp or stamp,
             'modalities':['Video'],'time_span':{'start_time':.2,'end_time':1.5}}]}}


def write_rows(tmp_path, rows):
    p=tmp_path/'all_qa.json';p.write_text(json.dumps(rows),encoding='utf-8');return p


def test_native_mcq_preserves_query_and_abstention_position():
    item=native_item()
    q=item.to_query().model_dump(mode='json')
    assert q['options']==item.options and q['answer_format']=='mcq'
    assert not set(q)&{'answer','abstention_option','evidence','correct_option_index'}
    assert 'D. shelf' in build_prompt(q) and '\nE.' not in build_prompt(q)
    info=SystemInfo(system_id='stub',context_mode='blind')
    pred=PredictionRow.from_item(item,run_id='run',system=info,answer='B')
    assert score_prediction(pred).abstained and score_prediction(pred).score==0
    pred.answer='C'
    assert score_prediction(pred).score==1 and not score_prediction(pred).abstained
    unans=native_item(answer='B',is_unanswerable=True)
    assert score_prediction(PredictionRow.from_item(unans,run_id='run',system=info,answer='B')).score==1
    assert extract_mcq_letter('E',options=item.options) is None


def test_native_e_is_not_automatically_abstention():
    item=native_item(options={'A':'one','B':'two','C':'three','D':'four','E':'five'},
                     answer='E',abstention_option=None)
    pred=PredictionRow.from_item(item,run_id='run',system=SystemInfo(system_id='s',context_mode='blind'),answer='E')
    assert score_prediction(pred).score==1 and not score_prediction(pred).abstained
    assert extract_mcq_letter('insufficient information',options=item.options) is None


@pytest.mark.parametrize('change',[{'answer':'E'},{'abstention_option':'E'},
    {'is_unanswerable':True},{'options':{'A':'a','C':'c'}},
    {'options':{'A':'','B':'b'}}, {'answer_format':'mcq5'}])
def test_invalid_native_contract(change):
    with pytest.raises(ValueError): native_item(**change)


def test_plan_preserves_original_and_filters_modality_future_and_sessions(tmp_path):
    good=row()
    audio=row(2);audio['answer_evidence']['evidence_list'][0]['modalities']=['Audio','Video']
    future=row(3);future['answer_evidence']['evidence_list'][0]['time_span']['end_time']=3.5
    cross=row(4,V2,2000,evidence_video=V1,evidence_stamp=1000)
    plan=make_plan(write_rows(tmp_path,[good,audio,future,cross]),limit=20,max_videos=20)
    assert [e['source']['question_id'] for e in plan['examples']]==[1]
    assert plan['examples'][0]['source']==good
    assert plan['examples'][0]['recordings'][0]['end_sec']==2
    assert 'requires_nonvisual_or_unknown_modality' in plan['counts']['excluded']
    assert 'answer_evidence_after_query_boundary' in plan['counts']['excluded']


def test_history_uses_all_prior_recordings_not_only_answer_evidence(tmp_path):
    rows=[row(1),row(2,V2,2000,evidence_video=V1,evidence_stamp=1000),row(3,V3,3000)]
    plan=make_plan(write_rows(tmp_path,rows),context='history',limit=20,max_videos=20)
    second=next(x for x in plan['examples'] if x['source']['question_id']==2)
    assert [r['video_id'] for r in second['recordings']]==[V1,V2]
    assert second['recordings'][0]['end_sec']==1002
    assert second['recordings'][1]['end_sec']==2


def test_unknown_history_rejected_instead_of_silently_dropped(tmp_path):
    plan=make_plan(write_rows(tmp_path,[row()]),context='history')
    assert plan['examples']==[]
    assert plan['counts']['excluded']['history_has_unknown_recording_times']==1


def test_duplicate_or_modified_plan_is_rejected(tmp_path):
    with pytest.raises(ValueError,match='duplicated'):
        make_plan(write_rows(tmp_path,[row(),row()]))
    plan=make_plan(write_rows(tmp_path,[row()]))
    path=tmp_path/'plan.json'; path.write_text(json.dumps(plan),encoding='utf-8')
    assert read_plan(path)['plan_sha256']==plan['plan_sha256']
    plan['examples'][0]['source']['correct_answer']='changed'
    path.write_text(json.dumps(plan),encoding='utf-8')
    with pytest.raises(ValueError,match='hash'): read_plan(path)


def video(path):
    import av
    import numpy as np
    with av.open(str(path),'w') as c:
        s=c.add_stream('libx264',rate=10);s.width=160;s.height=120;s.pix_fmt='yuv420p'
        for i in range(40):
            # Red before 2 seconds, blue afterwards: future leakage is visible.
            pixels=np.zeros((120,160,3),dtype=np.uint8);pixels[:,:,0 if i<20 else 2]=240
            f=av.VideoFrame.from_ndarray(pixels,format='rgb24')
            for pkt in s.encode(f): c.mux(pkt)
        for pkt in s.encode(): c.mux(pkt)


def test_preparation_physically_removes_future_and_preserves_raw(tmp_path, capsys):
    import hashlib
    import numpy as np
    src=tmp_path/(V1+'.mp4');video(src)
    original=hashlib.sha256(src.read_bytes()).hexdigest()
    plan=make_plan(write_rows(tmp_path,[row()]))
    out=tmp_path/'suite'
    suite=prepare_suite(plan,tmp_path,out,chunk_seconds=1,sample_fps=2,max_side=128)
    output=capsys.readouterr().out
    assert 'clip 1/2' in output and 'clip 2/2' in output
    assert output.count('START')==2 and output.count('DONE')==2
    assert len(suite.items)==1
    item=suite.items[0]
    assert item.question==row()['question'] and list(item.options.values())==row()['choices']
    assert item.certificate.n_sessions==1 and not item.certificate.cross_session
    assert len(suite.envs[item.env_id].sessions)==2
    for s in suite.envs[item.env_id].sessions:
        assert Path(s.video_path).is_relative_to(out)
        for frame in sample_frames(s.video_path,n_frames=2):
            a=np.array(frame.image).mean(axis=(0,1))
            assert a[0]>200 and a[2]<40  # NEVER see the blue future
    assert hashlib.sha256(src.read_bytes()).hexdigest()==original
    assert suite.manifest['media_sha256']
    with pytest.raises(FileExistsError): prepare_suite(plan,tmp_path,out)


def test_missing_video_never_freezes_a_fake_visual_suite(tmp_path):
    plan=make_plan(write_rows(tmp_path,[row()]))
    with pytest.raises(FileNotFoundError,match='Missing videos'):
        prepare_suite(plan,tmp_path,tmp_path/'suite')
    assert not (tmp_path/'suite').exists()


def test_truncated_window_is_not_silently_padded_to_look_complete(tmp_path):
    src=tmp_path/'short.mp4';video(src)
    with pytest.raises(ValueError,match='ended before'):
        render_window(src,tmp_path/'bad.mp4',start=0,end=8,fps=2,max_side=128)
    assert not (tmp_path/'bad.mp4').exists()


@pytest.mark.parametrize('start,end,fps,channel', [
    (0, 1.9, 2, 0), (2.1, 3.9, 2, 2), (0, .05, 30, 0),
])
def test_render_sampling_budget_and_short_or_late_windows(tmp_path, start, end, fps, channel):
    import math
    import av
    import numpy as np
    src=tmp_path/'source.mp4';video(src)
    target=tmp_path/'bounded.mp4'
    stats=render_window(src,target,start=start,end=end,fps=fps,max_side=128,decode_threads=2)
    expected=math.ceil((end-start)*fps)
    # Conversion work must scale with output FPS, not input FPS. Decoder
    # reference frames remain necessary and must not all become RGB buffers.
    assert 0 < stats['converted_frames'] <= expected
    assert stats['output_frames'] == expected
    with av.open(str(target)) as c:
        frames=list(c.decode(video=0))
    assert len(frames)==expected
    for frame in frames:
        colors=np.asarray(frame.to_image()).mean(axis=(0,1))
        assert colors[channel]>200 and colors[2-channel]<40


def test_render_progress_covers_seek_preroll(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from itertools import count
    from meowbench.datasets import video_windows
    src=tmp_path/'source.mp4';video(src)
    ticks=count(0,6)
    monkeypatch.setattr(video_windows,'time',SimpleNamespace(monotonic=lambda:next(ticks)))
    updates=[]
    render_window(src,tmp_path/'late.mp4',start=2.5,end=3.5,fps=2,max_side=128,
                  progress=lambda position,encoded:updates.append((position,encoded)))
    assert any(t<2.5 for t,_ in updates), 'seek preroll must not look like a hang'
    assert any(2.5<=t<3.5 for t,_ in updates)


def load_script(name):
    path=Path(__file__).resolve().parents[1]/'scripts'/f'{name}.py'
    sys.path.insert(0,str(path.parent))
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def test_viewer_escapes_html_and_supports_plan_and_prepared_video(tmp_path):
    viewer=load_script('make_real_review')
    r=row();r['question']='Where? </script><img src=x onerror=alert(1)>'
    plan=make_plan(write_rows(tmp_path,[r]))
    page=tmp_path/'review.html'
    viewer.render(viewer.review_data(plan),page)
    text=page.read_text(encoding='utf-8')
    assert '</script><img' not in text and '\\u003c/script' in text
    src=tmp_path/(V1+'.mp4');video(src)
    suite=prepare_suite(plan,tmp_path,tmp_path/'suite',chunk_seconds=1,max_side=128)
    data=viewer.review_data(plan,suite,out=page,copy_media=True)
    assert all(m['poster'].startswith('data:image/jpeg') and m['url'] for m in data['media'].values())
    viewer.render(data,page)
    assert len(list((tmp_path/'review_media').glob('*.mp4')))==2


def test_different_model_backends_share_protocol_and_safe_argv():
    launcher=load_script('run_real')
    hf=launcher.adapter_command({'id':'q','model_path':'/my models/q','gpu':'0'},'memory','python')
    assert '/my models/q' in hf and '--context-mode' in hf
    api=launcher.adapter_command({'id':'api','backend':'openai','model':'m','base_url':'http://localhost:8000/v1'},'oracle','python')
    assert 'meowbench.adapters.openai_compat' in api
    ext=launcher.adapter_command({'backend':'external','command':['/other/python','a.py','--mode','{context_mode}']},'blind','python')
    assert ext[-1]=='blind'
    with pytest.raises(ValueError): launcher.adapter_command({'backend':'external','command':'python a.py'},'blind','python')


def test_hf_diagnostics_are_explicit_and_command_records_settings():
    launcher=load_script('run_real')
    model={'id':'q','model_path':'/model','max_new_tokens':32,
           'note_max_new_tokens':128,'torch_num_threads':8}
    cmd=launcher.adapter_command(model,'memory','python',trace_file='/run with spaces/trace.jsonl')
    for flag, value in [('--trace-file','/run with spaces/trace.jsonl'),
                        ('--max-new-tokens','32'), ('--note-max-new-tokens','128'),
                        ('--torch-num-threads','8')]:
        assert cmd[cmd.index(flag)+1] == value
    with pytest.raises(ValueError,match='HF adapter only'):
        launcher.adapter_command({'backend':'external'},'oracle','python',trace_file='x')
    for invalid in (0, -1, True, '8'):
        with pytest.raises(ValueError,match='positive integer'):
            launcher.adapter_command({**model,'torch_num_threads':invalid},'memory','python')


def test_prepared_real_profile_runs_three_tracks_and_exports_predictions(tmp_path):
    from test_adapters import FakeServer, run_with
    from meowbench.schema import ContextMode
    src=tmp_path/(V1+'.mp4');video(src)
    plan=make_plan(write_rows(tmp_path,[row()]))
    suite=prepare_suite(plan,tmp_path,tmp_path/'suite',chunk_seconds=1,max_side=128)
    run_dirs=[]
    for mode in ContextMode:
        with FakeServer(responder=lambda payload:'A') as server:
            predictions=run_with(tmp_path,suite.envs,suite.items,server,mode,n_frames=2)
        assert len(predictions)==1 and score_prediction(predictions[0]).score==1
        if mode is not ContextMode.BLIND:
            assert predictions[0].env_run.total_frames==4
        run_dirs.append(tmp_path/'art'/mode.value)
    viewer=load_script('make_real_review')
    data=viewer.review_data(plan,suite,runs=run_dirs,out=tmp_path/'results.html')
    assert len(data['examples'][0]['predictions'])==3
    viewer.render(data,tmp_path/'results.html')
