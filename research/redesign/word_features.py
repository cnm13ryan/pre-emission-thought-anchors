"""Phase 1 (redesign): word-level boundary features from the stored trace and layer-48 states.

For every sentence row in Stage 7 ``features.parquet``:
- ``prefix_ids``: the last 64 token ids at or before ``boundary_token_idx`` (pre-emission);
- ``prev_token``: the id at ``boundary_token_idx``;
- ``top_ids`` / ``top_probs``: top-20 of softmax(lm_head(final_norm(h48))) in float32,
  computed in the same bf16 path as ``preanchor.model.logits.boundary_logits``.

Gates: re-tokenisation must reproduce Stage 7 ``token_ids_sha256`` for every trajectory,
and the recomputed entropy must match the stored ``next_token_entropy``; the observed
error is reported, not used to change anything.

Usage: python research/redesign/word_features.py --out FILE [--device mps]
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from safetensors import safe_open
from transformers import AutoTokenizer

ROOT = Path.home() / "preanchor-root"
SNAP = (ROOT / "cache/hf/hub/models--deepseek-ai--DeepSeek-R1-Distill-Qwen-14B/snapshots/"
        "1df8507178afcc1bef68cd8c393f61a886323761")
ACT_RUN = ROOT / "artifacts/runs/20260915T182306Z-3ab408d80a-7452f1c2/activations"
K_PREFIX, TOP_K, EPS = 64, 20, 1e-5


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="mps")
    args = parser.parse_args()
    feats = pd.read_parquet(ROOT / "data/processed/stage7/features.parquet")
    trajs = pd.read_parquet(ROOT / "data/processed/stage3/trajectories.parquet")
    tok = AutoTokenizer.from_pretrained(SNAP)
    with safe_open(SNAP / "model-00004-of-000004.safetensors", "pt") as f:
        head = f.get_tensor("lm_head.weight").to(args.device)          # bf16 [V, d]
        norm_w = f.get_tensor("model.norm.weight").to(args.device)  # same shard (index)
    rows, audit = [], {"token_hash_mismatch": [], "entropy_abs_err": []}
    for tid, group in feats.groupby("trajectory_id", sort=True):
        text = trajs.loc[trajs.trajectory_id == tid, "full_cot"].item()
        ids = tok(text, add_special_tokens=False)["input_ids"]
        digest = hashlib.sha256(",".join(map(str, ids)).encode()).hexdigest()
        if digest != group.token_ids_sha256.iloc[0]:
            audit["token_hash_mismatch"].append(tid)
            continue
        unit = ACT_RUN / group.shard_path.iloc[0]
        with safe_open(unit, "pt") as f:
            act = f.get_tensor("resid")                                  # [n, 49, d] fp16
        h = act[group.row_idx.to_numpy(), 48].to(args.device).to(torch.bfloat16)
        with torch.inference_mode():
            hf = h.float()
            normed = hf * torch.rsqrt(hf.pow(2).mean(-1, keepdim=True) + EPS)
            normed = norm_w * normed.to(torch.bfloat16)
            logits = (normed @ head.T).float()
            logp = torch.log_softmax(logits, -1)
            probs = logp.exp()
            entropy = -(probs * logp).sum(-1).cpu().numpy()
            top = torch.topk(probs, TOP_K, -1)
        top_ids, top_p = top.indices.cpu().numpy(), top.values.cpu().numpy()
        for j, r in enumerate(group.itertuples()):
            b = int(r.boundary_token_idx)
            rows.append({"sentence_id": r.sentence_id, "prev_token": int(ids[b]),
                         "prefix_ids": [int(x) for x in ids[max(0, b - K_PREFIX + 1): b + 1]],
                         "top_ids": top_ids[j].tolist(), "top_probs": top_p[j].tolist(),
                         "entropy_recomputed": float(entropy[j])})
            audit["entropy_abs_err"].append(abs(float(entropy[j]) - r.next_token_entropy))
        print(tid, len(group), flush=True)
    out = pd.DataFrame(rows)
    out.to_parquet(args.out, index=False)
    err = np.asarray(audit["entropy_abs_err"])
    summary = {"rows": len(out), "token_hash_mismatch": audit["token_hash_mismatch"],
               "entropy_abs_err_max": float(err.max()) if len(err) else None,
               "entropy_abs_err_p99": float(np.quantile(err, .99)) if len(err) else None,
               "entropy_abs_err_median": float(np.median(err)) if len(err) else None,
               "device": args.device}
    Path(str(args.out) + ".audit.json").write_text(json.dumps(summary, indent=1))
    print(summary)


if __name__ == "__main__":
    main()
