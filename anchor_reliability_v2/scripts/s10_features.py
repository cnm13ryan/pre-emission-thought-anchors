"""Boundary features for the prediction baselines (plan §8). Pre-emission only.

From the stored trace and the MVP layer-48 boundary states (read in place, shard hashes
checked against inputs/MANIFEST.json):
- prev_token, prefix_ids (last 64 tokens at or before the boundary);
- top-20 next-token ids/probs, entropy, from softmax(lm_head(RMSNorm(h48))) in bf16 → fp32;
- marker masses: next-token probability on reasoning-marker tokens, grouped.
Adapted from research/redesign/word_features.py (this repository, uncommitted at f11823f).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from safetensors import safe_open
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from arv2.paths import DERIVED, INPUTS, RESULTS, sha256_file, write_json  # noqa: E402

SNAP = Path(os.environ.get("ARV2_MODEL_SNAPSHOT", Path.home() / "preanchor-root/cache/hf/hub/"
            "models--deepseek-ai--DeepSeek-R1-Distill-Qwen-14B/snapshots/"
            "1df8507178afcc1bef68cd8c393f61a886323761"))
MARKERS = {
    "backtrack": ["Wait", "But", "However", "Hmm", "Actually", "Hold", "Alternatively", "No"],
    "plan": ["Let", "So", "Now", "First", "Next", "Then", "To", "I"],
    "conclude": ["Therefore", "Thus", "Hence", "Final", "**"],
    "check": ["Check", "Verify", "Double", "Let's"],
}
K_PREFIX, TOP_K, EPS = 64, 20, 1e-5


def marker_ids(tok) -> dict[str, list[int]]:
    out = {}
    for group, words in MARKERS.items():
        ids = set()
        for w in words:
            for variant in (w, " " + w, "\n" + w, "\n\n" + w):
                enc = tok(variant, add_special_tokens=False)["input_ids"]
                if len(enc) == 1:
                    ids.add(enc[0])
        out[group] = sorted(ids)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="mps")
    args = ap.parse_args()
    manifest = json.loads((INPUTS / "MANIFEST.json").read_text())
    act_root = Path(manifest["referenced_in_place"]["activation_root"])
    feats = pd.read_parquet(INPUTS / "mvp_features.parquet")
    trajs = pd.read_parquet(INPUTS / "trajectories.parquet")
    tok = AutoTokenizer.from_pretrained(SNAP)
    mids = marker_ids(tok)
    with safe_open(SNAP / "model-00004-of-000004.safetensors", "pt") as f:
        head = f.get_tensor("lm_head.weight").to(args.device)
        norm_w = f.get_tensor("model.norm.weight").to(args.device)
    rows, audit = [], {"token_hash_mismatch": [], "shard_hash_mismatch": [], "entropy_err": []}
    for tid, g in feats.groupby("trajectory_id", sort=True):
        text = trajs.loc[trajs.trajectory_id == tid, "full_cot"].item()
        ids = tok(text, add_special_tokens=False)["input_ids"]
        if hashlib.sha256(",".join(map(str, ids)).encode()).hexdigest() != g.token_ids_sha256.iloc[0]:
            audit["token_hash_mismatch"].append(tid)
            continue
        rel = g.shard_path.iloc[0]
        unit = act_root / rel
        if sha256_file(unit) != manifest["referenced_in_place"]["shards"][rel]:
            audit["shard_hash_mismatch"].append(tid)
            continue
        with safe_open(unit, "pt") as f:
            h = f.get_tensor("resid")[g.row_idx.to_numpy(), 48]
        h = h.to(args.device).to(torch.bfloat16)
        with torch.inference_mode():
            hf = h.float()
            x = (norm_w * (hf * torch.rsqrt(hf.pow(2).mean(-1, keepdim=True) + EPS))
                 .to(torch.bfloat16))
            logp = torch.log_softmax((x @ head.T).float(), -1)
            p = logp.exp()
            ent = -(p * logp).sum(-1).cpu().numpy()
            top = torch.topk(p, TOP_K, -1)
            masses = {k: p[:, v].sum(-1).cpu().numpy() for k, v in mids.items()}
        ti, tp = top.indices.cpu().numpy(), top.values.cpu().numpy()
        for j, r in enumerate(g.itertuples()):
            b = int(r.boundary_token_idx)
            rec = {"sentence_id": r.sentence_id, "prev_token": int(ids[b]),
                   "prefix_ids": [int(t) for t in ids[max(0, b - K_PREFIX + 1): b + 1]],
                   "top_ids": ti[j].tolist(), "top_probs": tp[j].tolist(),
                   "entropy_v2": float(ent[j])}
            rec.update({f"marker_{k}": float(v[j]) for k, v in masses.items()})
            rows.append(rec)
            audit["entropy_err"].append(abs(float(ent[j]) - r.next_token_entropy))
    pd.DataFrame(rows).to_parquet(DERIVED / "boundary_features.parquet", index=False)
    err = np.asarray(audit["entropy_err"])
    summary = {"rows": len(rows), "token_hash_mismatch": audit["token_hash_mismatch"],
               "shard_hash_mismatch": audit["shard_hash_mismatch"],
               "entropy_abs_err_median": float(np.median(err)),
               "entropy_abs_err_max": float(err.max()), "marker_token_ids": mids,
               "device": args.device}
    write_json(RESULTS / "s10_features_audit.json", summary)
    print({k: v for k, v in summary.items() if k != "marker_token_ids"})


if __name__ == "__main__":
    main()
