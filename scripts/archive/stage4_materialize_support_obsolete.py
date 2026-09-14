import json,time,math
from pathlib import Path
import pyarrow as pa, pyarrow.parquet as pq
from sentence_transformers import SentenceTransformer,util
from preanchor.io import write_parquet,write_json
ROOT=Path('/lambda/nfs/preanchor/preanchor'); run='20260914T190027Z-stage4fallback-6e78475'; b=ROOT/'artifacts/runs'/run/'stages/stage4_fallback'; out=ROOT/'data/processed/stage4'; out.mkdir(exist_ok=True)
model_id='sentence-transformers/all-MiniLM-L6-v2'; rev='1110a243fdf4706b3f48f1d95db1a4f5529b4d41'; threshold=.8
m=json.loads((b/'source_manifest.json').read_text()); paths={}
for r in m['files']: paths.setdefault((r['problem_id'],r['trace_kind']),{})[int(Path(r['path']).parent.name.removeprefix('chunk_'))]=Path(r['local_path'])
sents=pq.read_table(ROOT/'data/processed/stage3/sentences.parquet').to_pylist(); model=SentenceTransformer(model_id,revision=rev,device='cpu')
rows=[]; t0=time.time()
for j,s in enumerate(sents):
 key=(s['problem_id'],s['trace_kind']); i=s['sentence_idx']; p=paths[key].get(i); target=s['text']; raw=json.loads(p.read_text()) if p else []
 texts=[x.get('chunk_removed','') for x in raw]; em=model.encode([target]+texts,batch_size=32,normalize_embeddings=True,convert_to_tensor=True,show_progress_bar=False); sims=util.cos_sim(em[0:1],em[1:]).cpu().numpy()[0].tolist() if texts else []
 dis=[k for k,v in enumerate(sims) if v < threshold]; answers=[raw[k].get('answer','') for k in dis]; no=sum(1 for a in answers if not a)
 rows.append({'sentence_id':s['sentence_id'],'trajectory_id':s['trajectory_id'],'problem_id':s['problem_id'],'trace_kind':s['trace_kind'],'sentence_idx':i,'source_fork_path':str(p),'sample_count':len(raw),'dissimilar_count':len(dis),'similar_count':len(raw)-len(dis),'without_no_answer_count':no,'without_answer_categories':json.dumps(sorted(set(answers))),'model_id':model_id,'model_revision':rev,'cosine_threshold':threshold,'source_manifest_sha256':'caf3ce503ca0260db47301b6e98bae9fa61860fd3a461d325718dfff86'})
 if (j+1)%100==0: print(j+1,flush=True)
write_parquet(out/'support_metadata.parquet',pa.Table.from_pylist(rows)); write_json(out/'support_metadata.json',{'schema_version':1,'rows':len(rows),'model_id':model_id,'model_revision':rev,'cosine_dissimilarity_rule':'similarity < 0.8','source_manifest_sha256':'caf3ce503ca0260db47301b6e98bae9faaf327819bff503cf7f13d68314578ffed07940f5','elapsed_seconds':time.time()-t0})
