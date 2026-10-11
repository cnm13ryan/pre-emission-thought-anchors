"""Phase 2 (redesign) scoring: labels per replicate and run-to-run reliability. Exploratory.

Implements ``phase2_plan.md`` §1-4 exactly. Inputs: ``completions_index.parquet`` (from
``phase2_generate.py index``), ``inputs.parquet``, the Phase 1 per-rollout table (as
replicate 0) and the Phase 1 similarity table. Runs on CPU/MPS.

Usage: python research/redesign/phase2_score.py --run-dir DIR --inputs FILE [--device mps]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
from preanchor.labels.fresh import answer_category  # noqa: E402

ROOT = Path.home() / "preanchor-root"
P1 = ROOT / "data/processed/redesign_phase1"
NA = "<NO_ANSWER>"
THRESH = {"a_kl_1e-9": 1.3549810893213545, "b_kl_kt": 0.24619239326587636,
          "c_fork_tv": 0.17}
SPEARMAN_MIN, ICC_MIN, KAPPA_MIN = 0.80, 0.75, 0.60
BOOT = 2000


def kl(without, with_, alpha):
    p, q = Counter(without), Counter(with_)
    cats = set(p) | set(q) | {NA}
    k = len(cats)
    v = 0.0
    for a in cats:
        pp = (p[a] + alpha) / (len(without) + alpha * k)
        qq = (q[a] + alpha) / (len(with_) + alpha * k)
        v += pp * math.log(pp / qq)
    return max(v, 0.0)


def tv(a, b):
    ca, cb = Counter(a), Counter(b)
    return 0.5 * sum(abs(ca[x] / len(a) - cb[x] / len(b)) for x in set(ca) | set(cb))


def icc_2_1(x: np.ndarray, y: np.ndarray) -> float:
    """Shrout & Fleiss ICC(2,1): two-way random, absolute agreement, single rater."""
    data = np.column_stack([x, y]).astype(float)
    n, k = data.shape
    grand = data.mean()
    msr = k * ((data.mean(1) - grand) ** 2).sum() / (n - 1)
    msc = n * ((data.mean(0) - grand) ** 2).sum() / (k - 1)
    sse = ((data - data.mean(1, keepdims=True) - data.mean(0, keepdims=True) + grand) ** 2).sum()
    mse = sse / ((n - 1) * (k - 1))
    return float((msr - mse) / (msr + (k - 1) * mse + k * (msc - mse) / n))


def kappa(x, y):
    x, y = np.asarray(x, bool), np.asarray(y, bool)
    po = np.mean(x == y)
    pe = x.mean() * y.mean() + (1 - x.mean()) * (1 - y.mean())
    return float((po - pe) / (1 - pe)) if pe < 1 else float("nan")


def labels_for(samples: pd.DataFrame, plan_inputs: pd.DataFrame) -> pd.DataFrame:
    """samples: trajectory_id, fork_idx, category, similarity(to S_fork) for one replicate."""
    by = {k: g for k, g in samples.groupby(["trajectory_id", "fork_idx"])}
    out = []
    for r in plan_inputs[plan_inputs.is_label_fork].itertuples():
        F, G = by.get((r.trajectory_id, r.fork_idx)), by.get((r.trajectory_id, r.fork_idx + 1))
        rec = {"trajectory_id": r.trajectory_id, "problem_id": r.problem_id,
               "sentence_idx": r.fork_idx, "status": "OK"}
        if F is None or G is None or not len(F) or not len(G):
            rec["status"] = "MISSING_FORK"
            out.append(rec)
            continue
        rec["n_fork_i"], rec["n_fork_next"] = len(F), len(G)
        rec["c_fork_tv"] = tv(F.category.tolist(), G.category.tolist())
        valid = F.similarity.notna()
        without = F.loc[valid & (F.similarity < 0.8), "category"].tolist()
        rec["n_without"] = len(without)
        rec["n_no_similarity"] = int((~valid).sum())
        if not without:
            rec["status"] = "NO_SUPPORT"
        else:
            w = G.category.tolist()
            rec["a_kl_1e-9"] = kl(without, w, 1e-9)
            rec["b_kl_kt"] = kl(without, w, 0.5)
        out.append(rec)
    return pd.DataFrame(out)


def reliability(x: pd.DataFrame, y: pd.DataFrame, seed_material: str) -> list[dict]:
    m = x.merge(y, on=["trajectory_id", "problem_id", "sentence_idx"], suffixes=("_1", "_2"))
    forms = [("a_kl_1e-9", "continuous_log1p"), ("a_kl_1e-9", "flag"),
             ("b_kl_kt", "continuous_log1p"), ("b_kl_kt", "flag"),
             ("c_fork_tv", "continuous"), ("c_fork_tv", "flag")]
    rng = np.random.default_rng(int(hashlib.sha256(seed_material.encode()).hexdigest()[:8], 16))
    problems = np.array(sorted(m.problem_id.unique()))
    boots = [rng.choice(problems, len(problems), replace=True) for _ in range(BOOT)]
    results = []
    for label, form in forms:
        c1, c2 = f"{label}_1", f"{label}_2"
        d = m[m[c1].notna() & m[c2].notna()]
        rec = {"label": label, "form": form, "rows": len(d),
               "excluded": int(len(m) - len(d))}

        def stats(frame):
            a, b = frame[c1].to_numpy(float), frame[c2].to_numpy(float)
            if form == "flag":
                t = THRESH[label]
                return {"kappa": kappa(a >= t, b >= t)}
            if form == "continuous_log1p":
                a, b = np.log1p(a), np.log1p(b)
            return {"spearman": float(spearmanr(a, b).statistic), "icc_2_1": icc_2_1(a, b)}

        point = stats(d)
        rec.update(point)
        draws = {k: [] for k in point}
        for sample in boots:
            frame = pd.concat([d[d.problem_id == p] for p in sample])
            if len(frame) < 3:
                continue
            for k, v in stats(frame).items():
                draws[k].append(v)
        for k, v in draws.items():
            v = np.asarray([z for z in v if np.isfinite(z)])
            rec[f"{k}_lo"], rec[f"{k}_hi"] = (float(np.quantile(v, .025)),
                                              float(np.quantile(v, .975))) if len(v) else (None, None)
            rec[f"{k}_nonfinite_draws"] = int(BOOT - len(v))
        if form == "flag":
            rec["pass"] = bool(rec["kappa"] >= KAPPA_MIN)
            rec["robust"] = bool(rec["pass"] and (rec["kappa_lo"] or -1) >= KAPPA_MIN)
            if label in THRESH:
                rec["flag_rate_1"] = float((d[c1] >= THRESH[label]).mean())
                rec["flag_rate_2"] = float((d[c2] >= THRESH[label]).mean())
        else:
            rec["pass"] = bool(rec["spearman"] >= SPEARMAN_MIN and rec["icc_2_1"] >= ICC_MIN)
            rec["robust"] = bool(rec["pass"] and (rec["spearman_lo"] or -1) >= SPEARMAN_MIN
                                 and (rec["icc_2_1_lo"] or -1) >= ICC_MIN)
        rec["verdict"] = ("pass" if rec["robust"] else "pass, not robust") if rec["pass"] \
            else "below threshold: dropped"
        results.append(rec)
    return results


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--inputs", type=Path, required=True)
    ap.add_argument("--device", default="mps")
    args = ap.parse_args()
    inputs = pd.read_parquet(args.inputs)
    idx = pd.read_parquet(args.run_dir / "completions_index.parquet")
    status = idx.groupby(["replicate", "status"]).size().to_dict()
    ok = idx[idx.status == "OK"].copy()
    # largest common N per fork across both replicates (plan §3)
    n = ok.groupby(["trajectory_id", "fork_idx", "replicate"]).size().unstack()
    common = n.min(axis=1).rename("common_n")
    ok = ok.merge(common, left_on=["trajectory_id", "fork_idx"], right_index=True)
    ok = ok[ok.sample_idx < ok.common_n]
    ok["category"] = [answer_category(t or "") for t in ok.raw_answer]
    ok["truncated"] = ok.finish_reason == "length"
    target = inputs.set_index(["trajectory_id", "fork_idx"]).target_text
    ok["target_text"] = [target[(t, f)] for t, f in zip(ok.trajectory_id, ok.fork_idx,
                                                        strict=True)]
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2",
                                revision="1110a243fdf4706b3f48f1d95db1a4f5529b4d41",
                                device=args.device)
    has = ok.first_replacement_sentence.notna()
    texts = sorted(set(ok.loc[has, "first_replacement_sentence"]) | set(ok.target_text))
    vec = model.encode(texts, batch_size=256, normalize_embeddings=True,
                       convert_to_numpy=True, show_progress_bar=False).astype(np.float32)
    pos = {t: i for i, t in enumerate(texts)}
    sim = np.full(len(ok), np.nan)
    a = ok.loc[has, "first_replacement_sentence"].map(pos).to_numpy()
    b = ok.loc[has, "target_text"].map(pos).to_numpy()
    sim[has.to_numpy()] = np.einsum("ij,ij->i", vec[a].astype(np.float64),
                                    vec[b].astype(np.float64))
    ok["similarity"] = sim
    reps = {r: labels_for(ok[ok.replicate == r], inputs) for r in (1, 2)}
    # replicate 0: released samples for the same forks (Phase 1 tables)
    import glob

    files = [f for f in glob.glob(str(P1 / "rollouts/*/fork_*.parquet"))]
    keys = set(zip(inputs.trajectory_id, inputs.fork_idx))
    rel = pd.concat([pd.read_parquet(f, columns=["trajectory_id", "fork_idx", "sample_idx",
                                                 "category"]) for f in files])
    rel = rel[[k in keys for k in zip(rel.trajectory_id, rel.fork_idx)]]
    rel = rel.merge(pd.read_parquet(P1 / "similarity.parquet"),
                    on=["trajectory_id", "fork_idx", "sample_idx"])
    reps[0] = labels_for(rel, inputs)
    primary = reliability(reps[1], reps[2], "20260914:phase2_bootstrap")
    secondary = {f"{i}_vs_0": reliability(reps[i], reps[0], f"20260914:phase2_bootstrap:{i}v0")
                 for i in (1, 2)}
    out_dir = REPO / "research/redesign/tables"
    for r, frame in reps.items():
        frame.to_csv(out_dir / f"phase2_labels_replicate{r}.csv", index=False)
    pd.DataFrame(primary).to_csv(out_dir / "phase2_reliability_primary.csv", index=False)
    no_answer = ok[ok.category == NA].assign(
        cls=lambda d: np.where(d.truncated, "length_truncated",
                               np.where(d.raw_answer.str.contains(r"\\boxed\{", regex=True),
                                        "parse_failure", "other_non_answer")))
    summary = {"stage": "exploratory", "preregistered": False,
               "status_counts": {f"{k[0]}:{k[1]}": int(v) for k, v in status.items()},
               "common_n_min": int(common.min()), "common_n_median": float(common.median()),
               "label_status": {r: f.status.value_counts().to_dict() for r, f in reps.items()},
               "truncation_rate": float(ok.truncated.mean()),
               "no_answer_classes": no_answer.groupby(["replicate", "cls"]).size()
               .rename(lambda k: f"{k[0]}:{k[1]}" if isinstance(k, tuple) else k).to_dict(),
               "similarity_missing": int(np.isnan(sim).sum()),
               "primary": primary, "secondary": secondary,
               "any_form_passes": any(r["pass"] for r in primary)}
    (out_dir / "phase2_summary.json").write_text(json.dumps(summary, indent=1, default=str))
    print(json.dumps(summary, indent=1, default=str))


if __name__ == "__main__":
    main()
