"""Download all approved raw resamples for the Stage 4 canonical-label fallback."""
from __future__ import annotations
import hashlib, json, os, time
from pathlib import Path, PurePosixPath
from huggingface_hub import HfApi, snapshot_download
from preanchor.config import load_config, config_hash
from preanchor.io import UnitStore, write_json, write_artifact

PREFIX = "deepseek-r1-distill-qwen-14b/temperature_0.6_top_p_0.95"

def main():
    root=Path(os.environ["PREANCHOR_ROOT"]); repo=Path(__file__).resolve().parents[1]
    cfg=load_config(repo/"configs/mvp.yaml"); run_id=os.environ["STAGE4_RUN_ID"]
    run=root/"artifacts/runs"/run_id; stage=run/"stages/stage4_fallback"; stage.mkdir(parents=True,exist_ok=True)
    api=HfApi(); rows=[]
    for kind in ("correct", "incorrect"):
      for p in cfg.dataset.problem_ids:
        d=f"{PREFIX}/{kind}_base_solution/problem_{p}"
        for e in api.list_repo_tree(repo_id=cfg.dataset.id,repo_type="dataset",revision=cfg.dataset.revision,path_in_repo=d,recursive=True):
            if getattr(e,"path","").endswith("/solutions.json") and "/chunk_" in e.path:
                rows.append({"path":e.path,"size":e.size,"git_blob_sha1":e.blob_id,"lfs_sha256":e.lfs.sha256 if e.lfs else None,"problem_id":p,"trace_kind":kind,"role":"audit_resamples"})
    rows.sort(key=lambda x:x["path"]); assert len(rows)==6474
    discovery={"schema_version":1,"dataset_id":cfg.dataset.id,"revision":cfg.dataset.revision,"files":rows}
    write_json(stage/"discovery.json",discovery)
    store=UnitStore(stage/"units"); cache=root/"cache/huggingface/hub"; manifest=[]; started=time.monotonic()
    for i,r in enumerate(rows,1):
        key=config_hash({"stage":"stage4_fallback","record":r,"discovery":config_hash(discovery),"run_id":run_id})
        def writer(directory,r=r):
            snap=snapshot_download(repo_id=cfg.dataset.id,repo_type="dataset",revision=cfg.dataset.revision,allow_patterns=[r["path"]],cache_dir=str(cache),max_workers=1)
            src=Path(snap)/r["path"]; data=src.read_bytes(); assert len(data)==r["size"]
            digest=hashlib.sha256(data).hexdigest(); assert (not r["lfs_sha256"] or digest==r["lfs_sha256"])
            write_artifact(directory/PurePosixPath(r["path"]).name,data)
        directory,created=store.produce(key,writer); path=directory/PurePosixPath(r["path"]).name
        manifest.append({**r,"unit_key":key,"local_path":str(path),"sha256":hashlib.sha256(path.read_bytes()).hexdigest()})
        if i%10==0:
            from preanchor.io import atomic_bytes, json_bytes
            atomic_bytes(stage/"progress.json", json_bytes({"completed":i,"total":len(rows),"elapsed_seconds":time.monotonic()-started}), replace=True)
    write_json(stage/"source_manifest.json",{"schema_version":1,"discovery_sha256":config_hash(discovery),"files":manifest})
    write_json(stage/"SUCCESS.json",{"schema_version":1,"files":len(manifest),"bytes":sum(r["size"] for r in rows)})
if __name__=='__main__': main()
