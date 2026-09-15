import json,math
from pathlib import Path
import pyarrow as pa
import pyarrow.parquet as pq
from preanchor.labels.recompute import recompute_sentence
from preanchor.labels.threshold import train_threshold
from preanchor.io import write_json,write_parquet
ROOT=Path('/lambda/nfs/preanchor/preanchor'); run='20260914T190027Z-stage4fallback-6e78475'; b=ROOT/'artifacts/runs'/run/'stages/stage4_fallback'; out=ROOT/'data/processed/stage4'; out.mkdir(parents=True,exist_ok=True)
m=json.loads((b/'source_manifest.json').read_text()); paths={}
for r in m['files']: paths.setdefault((r['problem_id'],r['trace_kind']),{})[int(Path(r['path']).parent.name.removeprefix('chunk_'))]=Path(r['local_path'])
sent=pq.read_table(ROOT/'data/processed/stage3/sentences.parquet').to_pylist(); rows=[]
for s in sent:
 rr=recompute_sentence(s['sentence_idx'],paths[(s['problem_id'],s['trace_kind'])]); rows.append({**s,**rr,'importance_released':s.get('importance_released'),'importance_legacy':rr.get('importance_canonical')})
train={2238,1591,4682,6596,3448,2189,3550,4605,2137,4019,3360,3935}; eligible=[r for r in rows if r.get('eligible') and math.isfinite(r['importance_primary']) and r['importance_primary']>=0]; q90=train_threshold(eligible,train,.9)
for r in rows: r['anchor_threshold']=q90; r['anchor']=bool(r.get('eligible') and r['importance_primary']>=q90)
write_parquet(out/'labels.parquet',pa.Table.from_pylist(rows)); write_json(out/'labels_summary.json',{'schema_version':1,'rows':len(rows),'eligible':len(eligible),'primary_source':'canonical_fallback','train_q90':q90,'finite_nonnegative':True,'threshold_problem_ids':sorted(train)})
print(json.dumps({'rows':len(rows),'eligible':len(eligible),'train_q90':q90}))
