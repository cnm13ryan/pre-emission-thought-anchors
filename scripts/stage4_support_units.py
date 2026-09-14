"""Resumable Stage 4 support units and CPU/GPU benchmark/materialization."""
import hashlib,json,time,os
from pathlib import Path
import pyarrow.parquet as pq
from sentence_transformers import SentenceTransformer
from preanchor.io import UnitStore,write_json
ROOT=Path('/lambda/nfs/preanchor/preanchor'); RUN='20260914T190027Z-stage4fallback-6e78475'; REV='1110a243fdf4706b3f48f1d95db1a4f5529b4d41'; MODEL='sentence-transformers/all-MiniLM-L6-v2'; TH=.8

def rows(n=100): return pq.read_table(ROOT/'data/processed/stage3/sentences.parquet').to_pylist()[:n]
def raw_paths():
 m=json.loads((ROOT/'artifacts/runs'/RUN/'stages/stage4_fallback/source_manifest.json').read_text()); out={}
 for r in m['files']: out[(r['problem_id'],r['trace_kind'],int(Path(r['path']).parent.name.removeprefix('chunk_')))] = Path(r['local_path'])
 return out,hashlib.sha256((ROOT/'artifacts/runs'/RUN/'stages/stage4_fallback/source_manifest.json').read_bytes()).hexdigest()
def compute(device,n=100):
 import torch
 model=SentenceTransformer(MODEL,revision=REV,device=device); paths,mh=raw_paths(); ss=rows(n); t0=time.time(); parse=embed=0; out=[]
 for s in ss:
  raw=json.loads(paths[(s['problem_id'],s['trace_kind'],s['sentence_idx'])].read_text()); texts=[s['text']]+[x.get('chunk_removed','') for x in raw]; a=time.time(); emb=model.encode(texts,batch_size=64,normalize_embeddings=True,convert_to_numpy=True,show_progress_bar=False); embed+=time.time()-a; parse+=time.time()-a
  sims=emb[1:]@emb[0]; dis=sims<TH; ans=[raw[i].get('answer','') for i,v in enumerate(dis) if v]; out.append({'sentence_id':s['sentence_id'],'source_manifest_sha256':mh,'model_id':MODEL,'model_revision':REV,'cosine_threshold':TH,'sample_count':len(raw),'dissimilar_count':int(dis.sum()),'similar_count':int((~dis).sum()),'no_answer_count':sum(not x for x in ans),'similarities':sims.tolist()})
 return out,time.time()-t0,embed
def materialize(device='cuda'):
    paths,mh=raw_paths(); ss=rows(6474); root=ROOT/'data/processed/stage4/support_units'; store=UnitStore(root); model=SentenceTransformer(MODEL,revision=REV,device=device); done=[]
    for s in ss:
        key=hashlib.sha256(f"{s['sentence_id']}:{mh}:{MODEL}:{REV}:{TH}".encode()).hexdigest()
        def writer(d,s=s):
            raw=json.loads(paths[(s['problem_id'],s['trace_kind'],s['sentence_idx'])].read_text()); texts=[s['text']]+[x.get('chunk_removed','') for x in raw]; emb=model.encode(texts,batch_size=64,normalize_embeddings=True,convert_to_numpy=True,show_progress_bar=False); sims=emb[1:]@emb[0]; dis=sims<TH; ans=[raw[i].get('answer','') for i,v in enumerate(dis) if v]; write_json(d/'result.json',{'sentence_id':s['sentence_id'],'source_manifest_sha256':mh,'model_id':MODEL,'model_revision':REV,'cosine_threshold':TH,'sample_count':len(raw),'dissimilar_count':int(dis.sum()),'similar_count':int((~dis).sum()),'no_answer_count':sum(not x for x in ans)})
        store.produce(key,writer); done.append(key)
    return done

def materialize_batched(device='cuda', group_size=100):
    """Batch targets/replacements across sentences; sentence units remain atomic."""
    import torch
    paths,mh=raw_paths(); ss=rows(6474); root=ROOT/'data/processed/stage4/support_units'; store=UnitStore(root); model=SentenceTransformer(MODEL,revision=REV,device=device); done=[]
    for start in range(0,len(ss),group_size):
        group=ss[start:start+group_size]; payload=[]; spans=[]
        for s in group:
            raw=json.loads(paths[(s['problem_id'],s['trace_kind'],s['sentence_idx'])].read_text()); begin=len(payload); payload.append(s['text']); payload.extend(x.get('chunk_removed','') for x in raw); spans.append((s,raw,begin,len(raw)))
        emb=model.encode(payload,batch_size=2048,normalize_embeddings=True,convert_to_numpy=True,show_progress_bar=False)
        for s,raw,begin,n in spans:
            key=hashlib.sha256(f"{s['sentence_id']}:{mh}:{MODEL}:{REV}:{TH}".encode()).hexdigest(); sims=emb[begin+1:begin+1+n]@emb[begin]; dis=sims<TH; ans=[raw[i].get('answer','') for i,v in enumerate(dis) if v]
            result={'sentence_id':s['sentence_id'],'source_manifest_sha256':mh,'model_id':MODEL,'model_revision':REV,'cosine_threshold':TH,'sample_count':n,'dissimilar_count':int(dis.sum()),'similar_count':int((~dis).sum()),'no_answer_count':sum(not x for x in ans)}
            store.produce(key,lambda d,result=result: write_json(d/'result.json',result)); done.append(key)
    return done
if __name__=='__main__':
 import sys
 dev=sys.argv[1] if len(sys.argv)>1 else 'cpu'; o,w,e=compute(dev,100); print(json.dumps({'device':dev,'sentences':100,'samples':sum(x['sample_count'] for x in o),'sentences_per_sec':100/w,'samples_per_sec':sum(x['sample_count'] for x in o)/w,'embedding_seconds':e,'wall_seconds':w,'rows':o},separators=(',',':')))
