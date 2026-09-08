import json, os, collections, sys

BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ann')

def load(fp):
    s = open(fp, encoding='utf-8').read()
    try:
        return json.loads(s), 'ok'
    except Exception as e:
        objs = []
        depth = 0
        start = None
        instr = False
        esc = False
        for i, ch in enumerate(s):
            if instr:
                if esc:
                    esc = False
                elif ch == '\\':
                    esc = True
                elif ch == '"':
                    instr = False
                continue
            if ch == '"':
                instr = True
            elif ch == '{':
                if depth == 0:
                    start = i
                depth += 1
            elif ch == '}':
                depth -= 1
                if depth == 0 and start is not None:
                    try:
                        objs.append(json.loads(s[start:i + 1]))
                    except Exception:
                        pass
                    start = None
        return objs, 'RECOVERED:' + str(e)[:35]

parts = sorted(os.listdir(BASE))
print('part | vids_place | place_ids | pid>=2vid | obj_ids | oid>=2vid | notes')
TP = TO = 0
multi_pid_parts = []
multi_oid_parts = []
allplace = {}
allobj = {}
for p in parts:
    pl, s1 = load(os.path.join(BASE, p, 'place.json'))
    ob, s2 = load(os.path.join(BASE, p, 'object.json'))
    allplace[p] = pl
    allobj[p] = ob
    pv = collections.defaultdict(set)
    ov = collections.defaultdict(set)
    for r in pl:
        pid = str(r.get('place_id') or '').strip()
        vid = str(r.get('video_id') or '').strip() or str(r.get('video_path') or '').strip()
        if pid:
            pv[pid].add(vid)
    for r in ob:
        oid = str(r.get('object_id') or '').strip()
        vid = str(r.get('video_id') or '').strip() or str(r.get('video_path') or '').strip()
        if oid:
            ov[oid].add(vid)
    plm = [k for k, v in pv.items() if len(v) >= 2]
    obm = [k for k, v in ov.items() if len(v) >= 2]
    TP += len(plm)
    TO += len(obm)
    if plm:
        multi_pid_parts.append(p)
    if obm:
        multi_oid_parts.append(p)
    nv = len(set(str(r.get('video_id') or '').strip() for r in pl))
    note = '' if (s1 == 'ok' and s2 == 'ok') else 'place:' + s1 + ' obj:' + s2
    print('%s | %d | %d | %d | %d | %d | %s' % (p, nv, len(pv), len(plm), len(ov), len(obm), note))

print()
print('TOTAL place_ids in >=2 videos:', TP, ' participants with any:', len(multi_pid_parts), multi_pid_parts)
print('TOTAL object_ids in >=2 videos:', TO, ' participants with any:', len(multi_oid_parts), multi_oid_parts)

print()
print('=== created_at / any date-like fields across all record types ===')
keys = collections.Counter()
created = collections.Counter()
for p in parts:
    for f in ['QA', 'event', 'object', 'place']:
        d, st = load(os.path.join(BASE, p, f + '.json'))
        for r in d:
            for k in r.keys():
                keys[(f, k)] += 1
            m = r.get('meta') or {}
            if isinstance(m, dict):
                for k in m.keys():
                    keys[(f, 'meta.' + k)] += 1
                if m.get('created_at'):
                    created[str(m['created_at'])] += 1
print('field inventory:')
for k, v in sorted(keys.items()):
    print('  ', k, v)
print()
print('distinct meta.created_at values:', len(created))
for k, v in created.most_common(40):
    print('  ', repr(k), v)
