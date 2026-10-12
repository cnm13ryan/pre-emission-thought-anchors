"""EXPLORATORY root-cause analysis: why do the stored internal representations add little
when predicting thought anchors, and which representations or targets work better?

Data: the 40 MVP traces (20 problems), 100 released resamples per fork, and the stored
residual stream at the boundary token BEFORE each sentence (all 49 layers).

Analyses:
A. Target ceilings: split-half (50/50) reliability of each target, Spearman–Brown
   projected to n = 100, overall and within trajectory.
B. Predictability: leave-one-problem-out (20 folds) PCA-128 + RidgeCV, for each
   representation × layer × target.
   - Representations: pre = h_i (before S_i); post = h_{i+1} (after S_i);
     delta = post − pre; pre+post; text = MiniLM embedding of S_i; position.
   - Targets: acc_level p_i, answer entropy H_i, noise floor, fork_tv, tv_debiased,
     signed Δacc, |Δacc|.
C. What the pre-emission fork_tv prediction carries: its correlation with the floor,
   entropy and accuracy, and its partial correlation with fork_tv given the floor.
D. Value-probe route: p̂ from h at every boundary; anchor = Δp̂ = p̂_{i+1} − p̂_i.
E. Clean-label check on the 190 Phase 2 sentences (300 resamples per fork, from s13):
   Spearman with pooled labels, and AP for the 7 real effects.

Output: results/s14_representation_rootcause.json and results/s14_oof_predictions.parquet.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from safetensors import safe_open
from scipy.stats import rankdata, spearmanr
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeCV
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from arv2 import reliability as R  # noqa: E402
from arv2.paths import DERIVED, INPUTS, RESULTS, sha256_file, write_json  # noqa: E402

LAYERS = [8, 16, 24, 32, 40, 48]
PCA_DIM = 128
ALPHAS = np.logspace(-1, 5, 13)
TARGETS = ["acc_level", "entropy", "floor", "fork_tv", "tv_debiased", "signed_dacc",
           "abs_dacc"]


# ---------------------------------------------------------------- targets
def fork_stats(roll: pd.DataFrame) -> dict:
    """Per (trajectory, fork): category codes and correctness, full and half A/B."""
    codes = {c: i for i, c in enumerate(sorted(roll.cat.unique()))}
    roll = roll.assign(code=roll.cat.map(codes))
    out = {}
    for key, g in roll.groupby(["trajectory_id", "fork_idx"]):
        g = g.sort_values("sample_idx")
        c, y = g.code.to_numpy(), g.correct_v2.to_numpy(float)
        h = g.sample_idx.to_numpy() % 2 == 0
        out[key] = {"all": (c, y), "A": (c[h], y[h]), "B": (c[~h], y[~h])}
    return out, len(codes)


def entropy(c, k):
    p = np.bincount(c, minlength=k) / len(c)
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


def tv(a, b, k):
    return 0.5 * float(np.abs(np.bincount(a, minlength=k) / len(a)
                              - np.bincount(b, minlength=k) / len(b)).sum())


def floor(a, b, k):
    p = np.bincount(np.concatenate([a, b]), minlength=k) / (len(a) + len(b))
    return 0.5 * float((np.sqrt(p * (1 - p) * (1 / len(a) + 1 / len(b))) * np.sqrt(2 / np.pi)).sum())


def targets_for(fs, k, tid, i, part):
    (ca, ya), (cb, yb) = fs[(tid, i)][part], fs[(tid, i + 1)][part]
    t, f = tv(ca, cb, k), floor(ca, cb, k)
    return {"acc_level": ya.mean(), "entropy": entropy(ca, k), "floor": f, "fork_tv": t,
            "tv_debiased": t - f, "signed_dacc": yb.mean() - ya.mean(),
            "abs_dacc": abs(yb.mean() - ya.mean())}


def perm_p(fs, k, tid, i, perms, rng):
    (ca, _), (cb, _) = fs[(tid, i)]["all"], fs[(tid, i + 1)]["all"]
    obs, cc, n = tv(ca, cb, k), np.concatenate([ca, cb]), len(ca)
    ge = sum(tv(cc[p[:n]], cc[p[n:]], k) >= obs for p in (rng.permutation(len(cc))
                                                         for _ in range(perms)))
    return (ge + 1) / (perms + 1)


# ---------------------------------------------------------------- representations
def load_acts(frame: pd.DataFrame) -> dict[int, np.ndarray]:
    manifest = json.loads((INPUTS / "MANIFEST.json").read_text())
    root = Path(manifest["referenced_in_place"]["activation_root"])
    feats = pd.read_parquet(INPUTS / "mvp_features.parquet").set_index("sentence_id")
    pos = {s: j for j, s in enumerate(frame.sentence_id)}
    acts = {L: np.full((len(frame), 5120), np.nan, np.float32) for L in LAYERS}
    for rel, digest in manifest["referenced_in_place"]["shards"].items():
        if sha256_file(root / rel) != digest:
            raise SystemExit(f"activation shard changed: {rel}")
        with safe_open(root / rel, "pt") as f:
            arr = f.get_tensor("resid")[:, LAYERS].float().numpy()
        for sid, r in feats[feats.shard_path == rel].iterrows():
            if sid in pos:
                for j, L in enumerate(LAYERS):
                    acts[L][pos[sid]] = arr[int(r.row_idx), j]
    return acts


def text_embeddings(texts: list[str], device: str) -> np.ndarray:
    from sentence_transformers import SentenceTransformer

    m = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2",
                            revision="1110a243fdf4706b3f48f1d95db1a4f5529b4d41", device=device)
    return m.encode(texts, batch_size=256, normalize_embeddings=True, convert_to_numpy=True,
                    show_progress_bar=False).astype(np.float32)


# ---------------------------------------------------------------- modelling
def lopo_predict(blocks: list[np.ndarray], Y: np.ndarray, groups: np.ndarray,
                 pca: bool) -> np.ndarray:
    """Leave-one-problem-out; each block is scaled and (optionally) PCA'd on train only."""
    oof = np.full(Y.shape, np.nan)
    for g in np.unique(groups):
        tr, te = groups != g, groups == g
        Xtr, Xte = [], []
        for B in blocks:
            sc = StandardScaler().fit(B[tr])
            btr, bte = sc.transform(B[tr]), sc.transform(B[te])
            if pca and B.shape[1] > PCA_DIM:
                p = PCA(PCA_DIM, svd_solver="randomized", random_state=0).fit(btr)
                btr, bte = p.transform(btr), p.transform(bte)
            Xtr.append(btr)
            Xte.append(bte)
        Xtr, Xte = np.hstack(Xtr), np.hstack(Xte)
        ys = StandardScaler().fit(Y[tr])
        m = RidgeCV(alphas=ALPHAS, alpha_per_target=True).fit(Xtr, ys.transform(Y[tr]))
        oof[te] = ys.inverse_transform(m.predict(Xte))
    return oof


def within(y, s, traj):
    vals = [spearmanr(y[traj == t], s[traj == t])[0] for t in np.unique(traj)]
    vals = [v for v in vals if np.isfinite(v)]
    return float(np.mean(vals)) if vals else np.nan


def boot_ci(y, s, groups, fn, reps=1000, seed=0):
    draws = R.cluster_bootstrap(groups, lambda r: fn(y[r], s[r]), reps, seed)
    lo, hi, _ = R.interval(draws)
    return lo, hi


def partial_spearman(y, s, z):
    """Spearman of y and s after linear removal of z (all on ranks)."""
    ry, rs, rz = rankdata(y), rankdata(s), np.column_stack([rankdata(c) for c in z])
    Z = np.column_stack([np.ones(len(y)), rz])
    res = lambda v: v - Z @ np.linalg.lstsq(Z, v, rcond=None)[0]  # noqa: E731
    return float(np.corrcoef(res(ry), res(rs))[0, 1])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="mps")
    ap.add_argument("--perms", type=int, default=500)
    args = ap.parse_args()
    out: dict = {"stage": "exploratory_root_cause", "layers": LAYERS, "pca_dim": PCA_DIM}

    sents = pd.read_parquet(INPUTS / "sentences.parquet")
    feats = pd.read_parquet(INPUTS / "mvp_features.parquet")[
        ["sentence_id", "prefix_token_count"]]
    roll = pd.read_parquet(DERIVED / "rollouts_v2.parquet",
                           columns=["trajectory_id", "fork_idx", "sample_idx", "cat",
                                    "correct_v2"])
    fs, k = fork_stats(roll)
    sents = sents.merge(feats, on="sentence_id").sort_values(
        ["trajectory_id", "sentence_idx"]).reset_index(drop=True)
    nxt = {(t, i) for t, i in zip(sents.trajectory_id, sents.sentence_idx)}
    sents = sents[[(t, i + 1) in nxt and (t, i) in fs and (t, i + 1) in fs
                   for t, i in zip(sents.trajectory_id, sents.sentence_idx)]].reset_index(drop=True)
    rows = []
    for t, i in zip(sents.trajectory_id, sents.sentence_idx):
        rec = targets_for(fs, k, t, i, "all")
        rec.update({f"A_{n}": v for n, v in targets_for(fs, k, t, i, "A").items()})
        rec.update({f"B_{n}": v for n, v in targets_for(fs, k, t, i, "B").items()})
        rows.append(rec)
    frame = pd.concat([sents.reset_index(drop=True), pd.DataFrame(rows)], axis=1)
    rng = np.random.default_rng(R.seed_from("20260914:arv2:s14:perm"))
    frame["p_tv"] = [perm_p(fs, k, t, i, args.perms, rng)
                     for t, i in zip(frame.trajectory_id, frame.sentence_idx)]
    groups, traj = frame.problem_id.to_numpy(), frame.trajectory_id.to_numpy()
    out["rows"], out["problems"], out["trajectories"] = len(frame), int(len(set(groups))), int(
        len(set(traj)))
    p = np.sort(frame.p_tv.to_numpy())
    passed = np.nonzero(p <= 0.10 * (np.arange(len(p)) + 1) / len(p))[0]
    out["real_effects_mvp_bh10"] = int(passed.max() + 1) if len(passed) else 0

    # A. ceilings
    ceil = {}
    for n in TARGETS:
        a, b = frame[f"A_{n}"].to_numpy(), frame[f"B_{n}"].to_numpy()
        r = spearmanr(a, b)[0]
        rw = within(a, b, traj)
        sb = lambda x: 2 * x / (1 + x) if x > 0 else x  # noqa: E731
        ceil[n] = {"split_half_spearman": float(r), "rel_n100": float(sb(r)),
                   "ceiling_corr": float(np.sqrt(max(sb(r), 0))),
                   "split_half_within": rw, "within_rel_n100": float(sb(rw)),
                   "within_ceiling_corr": float(np.sqrt(max(sb(rw), 0)))}
    out["ceilings"] = ceil
    print("ceilings", json.dumps({n: round(v["ceiling_corr"], 2) for n, v in ceil.items()}),
          flush=True)

    # B. representations
    Y = frame[TARGETS].to_numpy(float)
    acts = load_acts(frame)
    # post state for row i = the stored state before sentence i+1 (i.e. after S_i).
    full = pd.read_parquet(INPUTS / "mvp_features.parquet").sort_values(
        ["trajectory_id", "sentence_idx"]).reset_index(drop=True)
    post_frame = full.set_index(["trajectory_id", "sentence_idx"]).loc[
        list(zip(frame.trajectory_id, frame.sentence_idx + 1))].reset_index()
    post_acts = load_acts(post_frame.assign(sentence_id=post_frame.sentence_id))
    pos = np.column_stack([frame.sentence_idx, np.log1p(frame.sentence_idx),
                           np.log1p(frame.prefix_token_count)]).astype(float)
    txt = text_embeddings(frame.text.tolist(), args.device)
    preds, table = {}, []

    def record(rep, layer, oof):
        preds[(rep, layer)] = oof
        for j, n in enumerate(TARGETS):
            y, s = Y[:, j], oof[:, j]
            c = ceil[n]["ceiling_corr"]
            sp = float(spearmanr(y, s)[0])
            table.append({"rep": rep, "layer": layer, "target": n, "spearman": sp,
                          "within": within(y, s, traj),
                          "frac_of_ceiling": sp / c if c > 0 else None})

    record("position", None, lopo_predict([pos], Y, groups, pca=False))
    record("text_S_i", None, lopo_predict([txt], Y, groups, pca=True))
    record("text+position", None, lopo_predict([txt, pos], Y, groups, pca=True))
    for L in LAYERS:
        pre, post = acts[L], post_acts[L]
        record("pre", L, lopo_predict([pre], Y, groups, pca=True))
        record("post", L, lopo_predict([post], Y, groups, pca=True))
        record("delta", L, lopo_predict([post - pre], Y, groups, pca=True))
        record("pre+post", L, lopo_predict([pre, post], Y, groups, pca=True))
        record("pre+position", L, lopo_predict([pre, pos], Y, groups, pca=True))
        print("layer", L, "done", flush=True)
    tab = pd.DataFrame(table)
    out["predictability"] = tab.to_dict("records")

    # best layer per (rep, target), selected on the overall Spearman (descriptive only)
    best = (tab.sort_values("spearman", ascending=False).groupby(["rep", "target"]).head(1)
            .sort_values(["target", "spearman"], ascending=[True, False]))
    out["best_layer_by_rep_target"] = best.to_dict("records")

    # C. what the pre-emission fork_tv prediction carries (layer chosen by best pre fork_tv)
    Lb = int(best[(best.rep == "pre") & (best.target == "fork_tv")].layer.iloc[0])
    s_ftv = preds[("pre", Lb)][:, TARGETS.index("fork_tv")]
    y = frame.fork_tv.to_numpy()
    out["pre_fork_tv_decomposition"] = {
        "layer": Lb,
        "corr_pred_with": {n: float(spearmanr(s_ftv, frame[n])[0])
                           for n in ["fork_tv", "floor", "entropy", "acc_level", "tv_debiased",
                                     "abs_dacc"]},
        "partial_given_floor": partial_spearman(y, s_ftv, [frame.floor.to_numpy()]),
        "partial_given_floor_entropy_position": partial_spearman(
            y, s_ftv, [frame.floor.to_numpy(), frame.entropy.to_numpy(), pos[:, 0]]),
        "floor_alone_vs_fork_tv": float(spearmanr(frame.floor, y)[0]),
        "floor_alone_within": within(y, frame.floor.to_numpy(), traj),
    }

    # D. value-probe route: p̂ at each boundary, then differences
    vp = {}
    for L in LAYERS:
        ph_pre = preds[("pre", L)][:, TARGETS.index("acc_level")]
        # p̂ at the post boundary: the probe fit on pre states (p_i), applied to h_{i+1}.
        oof_post = np.full(len(frame), np.nan)
        for g in np.unique(groups):
            tr, te = groups != g, groups == g
            sc = StandardScaler().fit(acts[L][tr])
            pc = PCA(PCA_DIM, svd_solver="randomized", random_state=0).fit(sc.transform(acts[L][tr]))
            m = RidgeCV(alphas=ALPHAS).fit(pc.transform(sc.transform(acts[L][tr])),
                                           frame.acc_level.to_numpy()[tr])
            oof_post[te] = m.predict(pc.transform(sc.transform(post_acts[L][te])))
        d_hat = oof_post - ph_pre
        vp[L] = {"p_hat_spearman": float(spearmanr(ph_pre, frame.acc_level)[0]),
                 "p_hat_within": within(frame.acc_level.to_numpy(), ph_pre, traj),
                 "dphat_vs_signed_dacc": float(spearmanr(d_hat, frame.signed_dacc)[0]),
                 "dphat_vs_signed_dacc_within": within(frame.signed_dacc.to_numpy(), d_hat, traj),
                 "abs_dphat_vs_abs_dacc": float(spearmanr(np.abs(d_hat), frame.abs_dacc)[0]),
                 "abs_dphat_vs_tv_debiased": float(spearmanr(np.abs(d_hat), frame.tv_debiased)[0]),
                 "abs_dphat_vs_fork_tv": float(spearmanr(np.abs(d_hat), frame.fork_tv)[0])}
        preds[("valueprobe_dphat", L)] = d_hat
    out["value_probe"] = vp

    # E. clean-label check on the Phase 2 sentences
    eff = pd.read_csv(RESULTS / "s13_sentence_effects.csv")
    key = frame[["trajectory_id", "sentence_idx"]].reset_index().merge(
        eff, on=["trajectory_id", "sentence_idx"])
    ii = key["index"].to_numpy()
    real = (key.real_tv_bh10 | key.real_dacc_bh10).to_numpy()
    clean_signed = (key.acc_after - key.acc_before).to_numpy()
    clean = {"rows": len(key), "real_effects": int(real.sum()), "by_rep": []}
    cands = [("position", None, "fork_tv"), ("text_S_i", None, "fork_tv"),
             ("text+position", None, "fork_tv")]
    for L in LAYERS:
        cands += [("pre", L, "fork_tv"), ("post", L, "fork_tv"), ("delta", L, "fork_tv"),
                  ("pre+post", L, "fork_tv"), ("pre", L, "tv_debiased"),
                  ("pre+post", L, "tv_debiased"), ("delta", L, "abs_dacc")]
    for rep, L, tgt in cands:
        s = preds[(rep, L)][ii, TARGETS.index(tgt)]
        clean["by_rep"].append({
            "rep": rep, "layer": L, "trained_on": tgt,
            "spearman_vs_pooled_fork_tv": float(spearmanr(s, key.pooled_fork_tv)[0]),
            "within_vs_pooled_fork_tv": within(key.pooled_fork_tv.to_numpy(), s,
                                               key.trajectory_id.to_numpy()),
            "ap_real_effects": float(average_precision_score(real, s))})
    for L in LAYERS:
        d = preds[("valueprobe_dphat", L)][ii]
        clean["by_rep"].append({
            "rep": "valueprobe_|dphat|", "layer": L, "trained_on": "acc_level",
            "spearman_vs_pooled_fork_tv": float(spearmanr(np.abs(d), key.pooled_fork_tv)[0]),
            "within_vs_pooled_fork_tv": within(key.pooled_fork_tv.to_numpy(), np.abs(d),
                                               key.trajectory_id.to_numpy()),
            "ap_real_effects": float(average_precision_score(real, np.abs(d))),
            "signed_dphat_vs_clean_signed_dacc": float(spearmanr(d, clean_signed)[0])})
    clean["ap_prevalence"] = float(real.mean())
    out["phase2_clean_labels"] = clean

    pd.DataFrame({f"{r}_{L}": v[:, TARGETS.index("fork_tv")] if v.ndim == 2 else v
                  for (r, L), v in preds.items()}).assign(
        sentence_id=frame.sentence_id.to_numpy()).to_parquet(RESULTS / "s14_oof_predictions.parquet")
    write_json(RESULTS / "s14_representation_rootcause.json", out)
    print("done", flush=True)


if __name__ == "__main__":
    main()
