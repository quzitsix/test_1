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


def test_preparation_physically_removes_future_and_preserves_raw(tmp_path, capsys, monkeypatch):
    import hashlib
    import numpy as np
    from meowbench.datasets import supermemory
    def unexpected_disk_check(path):
        pytest.fail('Default preparation must not add a disk-space requirement')
    monkeypatch.setattr(supermemory.shutil, 'disk_usage', unexpected_disk_check)
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
    assert 'min_free_bytes' not in suite.manifest['preparation']
    with pytest.raises(FileExistsError): prepare_suite(plan,tmp_path,out)


@pytest.mark.parametrize('phase', ['before_output', 'before_clip'])
def test_preparation_reserves_clip_bytes_before_creating_or_rendering(tmp_path, monkeypatch, phase):
    from types import SimpleNamespace
    from meowbench.datasets import supermemory, video_windows
    src=tmp_path/(V1+'.mp4');video(src)
    original=src.read_bytes()
    plan=make_plan(write_rows(tmp_path,[row()]))
    floor=100
    required=2 * 128**2 * 3 + 1024**2  # One second, 2 FPS, bounded RGB + overhead.
    values=iter(([floor+required-1] if phase=='before_output' else
                 [floor+required, floor+required-1]))
    monkeypatch.setattr(supermemory.shutil, 'disk_usage',
                        lambda path: SimpleNamespace(free=next(values)))
    monkeypatch.setattr(video_windows, 'render_window',
                        lambda *a, **kw: pytest.fail('Insufficient space must stop before render'))
    out=tmp_path/'suite'
    with pytest.raises(ValueError, match='reserved disk floor'):
        prepare_suite(plan,tmp_path,out,chunk_seconds=1,sample_fps=2,max_side=128,
                      min_free_bytes=floor)
    assert out.exists() == (phase=='before_clip')
    assert not (out/'media').exists() and src.read_bytes()==original


def test_preparation_progress_floor_preserves_completed_clips_and_cleans_partial(tmp_path, monkeypatch):
    from itertools import count
    from types import SimpleNamespace
    from meowbench.datasets import supermemory, video_windows
    src=tmp_path/(V1+'.mp4');video(src)
    original=src.read_bytes()
    plan=make_plan(write_rows(tmp_path,[row()]))
    ticks=count(0,6)
    monkeypatch.setattr(video_windows, 'time', SimpleNamespace(monotonic=lambda:next(ticks)))
    space=SimpleNamespace(free=10**9)
    monkeypatch.setattr(supermemory.shutil, 'disk_usage', lambda path:space)
    real_render=video_windows.render_window
    calls=[]
    def render(source,target,**kwargs):
        calls.append(target)
        progress=kwargs.pop('progress')
        def guarded(position,encoded):
            if len(calls)==2 and encoded>0:
                space.free=99
            progress(position,encoded)
        return real_render(source,target,progress=guarded,**kwargs)
    monkeypatch.setattr(video_windows,'render_window',render)
    out=tmp_path/'suite'
    with pytest.raises(ValueError,match='reserved disk floor'):
        prepare_suite(plan,tmp_path,out,chunk_seconds=1,sample_fps=2,max_side=128,
                      min_free_bytes=100)
    assert len(calls)==2 and calls[0].is_file() and not calls[1].exists()
    assert not list(out.rglob('*.partial.mp4')) and not (out/'manifest.json').exists()
    assert src.read_bytes()==original


@pytest.mark.parametrize('floor', [-1, True, 1.5, '1', float('nan'), float('inf')])
def test_preparation_rejects_invalid_floor_before_any_output(tmp_path, floor):
    with pytest.raises(ValueError,match='preparation settings'):
        prepare_suite({},tmp_path,tmp_path/'suite',min_free_bytes=floor)
    assert not (tmp_path/'suite').exists()


def test_preparation_records_nonzero_floor_and_cli_validates_gib(tmp_path,monkeypatch):
    from types import SimpleNamespace
    from meowbench.datasets import supermemory
    src=tmp_path/(V1+'.mp4');video(src)
    plan=make_plan(write_rows(tmp_path,[row()]))
    monkeypatch.setattr(supermemory.shutil,'disk_usage',lambda path:SimpleNamespace(free=10**9))
    suite=prepare_suite(plan,tmp_path,tmp_path/'suite',chunk_seconds=1,max_side=128,
                        min_free_bytes=100)
    assert suite.manifest['preparation']['min_free_bytes']==100
    script=load_script('prepare_supermemory')
    monkeypatch.setattr(script,'read_plan',lambda path:plan)
    received=[]
    def prepare(*args,**kwargs):
        received.append(kwargs['min_free_bytes'])
        return SimpleNamespace(describe=lambda:'synthetic suite')
    monkeypatch.setattr(script,'prepare_suite',prepare)
    command=['prepare_supermemory.py','prepare','--plan','unused','--video-root',str(tmp_path),
             '--out',str(tmp_path/'cli-suite'),'--min-free-gib']
    monkeypatch.setattr(script.sys,'argv',command+['1.5'])
    assert script.main()==0 and received==[int(1.5*1024**3)]
    for value in ('-1','nan','inf'):
        monkeypatch.setattr(script.sys,'argv',command+[value])
        with pytest.raises(SystemExit) as exc:
            script.main()
        assert exc.value.code==2
    assert len(received)==1


def test_parallel_preparation_deduplicates_and_matches_single_worker_media(tmp_path,monkeypatch):
    import av
    from threading import Barrier, Lock
    from meowbench.datasets import video_windows
    src=tmp_path/(V1+'.mp4');video(src)
    original=src.read_bytes()
    plan=make_plan(write_rows(tmp_path,[row(1),row(2)]))
    sequential=prepare_suite(plan,tmp_path,tmp_path/'sequential',chunk_seconds=.5,max_side=128)
    real_render=video_windows.render_window
    barrier,lock=Barrier(4),Lock()
    calls=[]
    active=peak=0
    def concurrent(source,target,**kwargs):
        nonlocal active,peak
        with lock:
            calls.append(target.name)
            active+=1
            peak=max(peak,active)
        try:
            barrier.wait(timeout=5)
            assert kwargs['decode_threads']==4
            return real_render(source,target,**kwargs)
        finally:
            with lock:
                active-=1
    monkeypatch.setattr(video_windows,'render_window',concurrent)
    parallel=prepare_suite(plan,tmp_path,tmp_path/'parallel',chunk_seconds=.5,max_side=128,workers=4)
    assert peak==4 and active==0 and len(calls)==len(set(calls))==4
    assert parallel.manifest['preparation']['workers']==4
    assert 'workers' not in sequential.manifest['preparation']
    assert [item.model_dump() for item in parallel.items]==[item.model_dump() for item in sequential.items]
    assert parallel.manifest['media_sha256'].keys()==sequential.manifest['media_sha256'].keys()
    def pixels(path):
        with av.open(str(path)) as reader:
            return [(frame.pts,frame.width,frame.height,frame.to_ndarray(format='rgb24').tobytes())
                    for frame in reader.decode(video=0)]
    for relative in parallel.manifest['media_sha256']:
        assert pixels(tmp_path/'parallel'/relative)==pixels(tmp_path/'sequential'/relative)
    assert src.read_bytes()==original


@pytest.mark.parametrize('workers',[2,4])
def test_parallel_preparation_reserves_all_concurrent_clip_budgets(tmp_path,monkeypatch,workers):
    from types import SimpleNamespace
    from meowbench.datasets import supermemory,video_windows
    src=tmp_path/(V1+'.mp4');video(src)
    plan=make_plan(write_rows(tmp_path,[row()]))
    per_clip=128**2*3+1024**2  # .5-second clips at 2 FPS.
    monkeypatch.setattr(supermemory.shutil,'disk_usage',
                        lambda path:SimpleNamespace(free=100+workers*per_clip-1))
    monkeypatch.setattr(video_windows,'render_window',
                        lambda *a,**kw:pytest.fail('All worker reservations must pass before rendering'))
    out=tmp_path/'suite'
    with pytest.raises(ValueError,match='reserved disk floor'):
        prepare_suite(plan,tmp_path,out,chunk_seconds=.5,max_side=128,
                      min_free_bytes=100,workers=workers)
    assert not out.exists()


def test_parallel_failure_cancels_pending_jobs_and_waits_for_active_cleanup(tmp_path,monkeypatch):
    from threading import Barrier,Lock
    import time
    from meowbench.datasets import video_windows
    src=tmp_path/(V1+'.mp4');video(src)
    original=src.read_bytes()
    plan=make_plan(write_rows(tmp_path,[row()]))
    barrier,lock=Barrier(2),Lock()
    started,finished=[],[]
    def failing(source,target,*,start,progress,**kwargs):
        target.parent.mkdir(parents=True,exist_ok=True)
        partial=target.with_suffix('.partial.mp4')
        partial.write_bytes(b'temporary encoded bytes')
        with lock:
            started.append(start)
        try:
            barrier.wait(timeout=5)
            if start==0:
                raise ValueError('synthetic primary render failure')
            deadline=time.monotonic()+5
            while time.monotonic()<deadline:
                progress(start,0)
                time.sleep(.001)
            pytest.fail('Active worker did not receive cancellation')
        finally:
            time.sleep(.02)  # Caller must wait for this cleanup before returning.
            partial.unlink()
            with lock:
                finished.append(start)
    monkeypatch.setattr(video_windows,'render_window',failing)
    out=tmp_path/'suite'
    with pytest.raises(ValueError,match='synthetic primary render failure'):
        prepare_suite(plan,tmp_path,out,chunk_seconds=.25,max_side=128,workers=2)
    assert sorted(started)==sorted(finished)==[0,.25]  # Six queued jobs never render.
    assert not list(out.rglob('*.partial.mp4')) and not (out/'manifest.json').exists()
    assert src.read_bytes()==original


@pytest.mark.parametrize('workers',[0,5,True,1.5])
def test_preparation_and_cli_reject_invalid_worker_counts(tmp_path,monkeypatch,workers):
    with pytest.raises(ValueError,match='preparation settings'):
        prepare_suite({},tmp_path,tmp_path/'suite',workers=workers)
    script=load_script('prepare_supermemory')
    monkeypatch.setattr(script.sys,'argv',['prepare_supermemory.py','prepare','--plan','unused',
        '--video-root',str(tmp_path),'--out',str(tmp_path/'suite'),'--workers',str(workers)])
    with pytest.raises(SystemExit) as exc:
        script.main()
    assert exc.value.code==2 and not (tmp_path/'suite').exists()


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
