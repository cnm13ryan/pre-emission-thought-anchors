"""Materialize compact raw-label-derived Stage 4 sensitivity inputs."""
import hashlib, json, math
from pathlib import Path
import pyarrow as pa
import pyarrow.parquet as pq
from preanchor.io import write_json, write_parquet
from preanchor.labels.convergence import convergence_idx

ROOT = Path('/lambda/nfs/preanchor/preanchor')
RUN = '20260914T190027Z-stage4fallback-6e78475'
STAGE = ROOT/'artifacts/runs'/RUN/'stages/stage4_fallback'
MANIFEST = STAGE/'source_manifest.json'
MODEL = 'sentence-transformers/all-MiniLM-L6-v2'
MODEL_REV = '1110a243fdf4706b3f48f1d95db1a4f5529b4d41'
OUT = ROOT/'data/processed/stage4'
TRAIN = {2238,1591,4682,6596,3448,2189,3550,4605,2137,4019,3360,3935}

def main():
    manifest_hash = hashlib.sha256(MANIFEST.read_bytes()).hexdigest()
    labels = {r['sentence_id']: r for r in pq.read_table(OUT/'labels.parquet').to_pylist()}
    support = {}
    for d in (OUT/'support_units').iterdir():
        if not d.is_dir() or d.name == '.locks': continue
        result = json.loads((d/'result.json').read_text())
        support[result['sentence_id']] = result
    if len(support) != 6474: raise RuntimeError(f'support units: {len(support)}')
    # The verified source units are used only to derive accuracy/convergence and
    # answer/no-answer support metadata; embedding decisions come from support units.
    source = {}
    manifest = json.loads(MANIFEST.read_text())
    for x in manifest['files']:
        source[(x['problem_id'], x['trace_kind'], int(Path(x['path']).parent.name.removeprefix('chunk_')))] = Path(x['local_path'])
    sentences = {r['sentence_id']: r for r in pq.read_table(ROOT/'data/processed/stage3/sentences.parquet').to_pylist()}
    by_traj = {}
    rows = []
    for sid, lab in labels.items():
        s = support[sid]
        sr = sentences[sid]
        raw = json.loads(source[(sr['problem_id'], sr['trace_kind'], sr['sentence_idx'])].read_text())
        # Build convergence from the released source metadata where available.
        acc = (sum(bool(x.get('is_correct')) for x in raw)/len(raw)) if raw else None
        row = dict(sentence_id=sid, problem_id=lab['problem_id'],
                   trace_kind=lab['trace_kind'], sentence_idx=lab['sentence_idx'],
                   source_manifest_sha256=manifest_hash, model_id=MODEL,
                   model_revision=MODEL_REV, cosine_dissimilarity_threshold=0.8,
                   sample_count=s['sample_count'],
                   similar_count=s['similar_count'],
                   dissimilar_count=s['dissimilar_count'],
                   no_answer_dissimilar_count=s['no_answer_count'],
                   importance_canonical=lab['importance_canonical'],
                   importance_primary=lab['importance_primary'],
                   eligible=lab['eligible'], train_problem=lab['problem_id'] in TRAIN,
                   trace_correct=(lab['trace_kind'] == 'correct'))
        by_traj.setdefault((lab['problem_id'], lab['trace_kind']), []).append((lab['sentence_idx'], acc))
        rows.append(row)
    conv = []
    for key, vals in by_traj.items():
        vals.sort()
        vals = [x for x in vals if x[1] is not None]
        a5 = convergence_idx([x[1] for x in vals], 5) if vals else None
        a4 = convergence_idx([x[1] for x in vals], 4) if vals else None
        for idx, _ in vals: conv.append((key[0], key[1], idx, a4, a5))
    cmap = {(p,t,i):(a4,a5) for p,t,i,a4,a5 in conv}
    for r in rows:
        a4,a5 = cmap.get((r['problem_id'],r['trace_kind'],r['sentence_idx']),(None,None))
        r['convergence_k4_idx']=a4; r['convergence_k5_idx']=a5
        r['eligible_k4']=r['eligible'] and (a4 is None or r['sentence_idx'] < a4)
        r['eligible_k5']=r['eligible'] and (a5 is None or r['sentence_idx'] < a5)
    out = OUT/'raw_sensitivity_inputs.parquet'; write_parquet(out, pa.Table.from_pylist(rows))
    vals=[r['importance_primary'] for r in rows if r['eligible'] and r['train_problem'] and math.isfinite(r['importance_primary'])]
    thresholds={f'q{q}': float(__import__('numpy').quantile(vals,q/100,method='linear')) for q in (80,90,95)}
    meta={'schema_version':1,'rows':len(rows),'source_manifest_sha256':manifest_hash,
          'model_id':MODEL,'model_revision':MODEL_REV,'cosine_rule':'similarity < 0.8',
          'thresholds_train_only':thresholds,'train_problem_ids':sorted(TRAIN),
          'support_units':len(support),'provenance':'stage4_support_units'}
    write_json(OUT/'raw_sensitivity_inputs.json',meta)
    print(json.dumps(meta, sort_keys=True))
if __name__ == '__main__': main()
