"""EXPLORATORY: is the failure to decode the outcome from the stored state a limitation of
linear PCA-128 probes, or of the representation itself? And does the state encode WHICH
answer the trace is heading for better than whether that answer is correct?

Builds on s14 (docs/representation_rootcause.md, root cause 5). Same data, same protocol:
leave-one-problem-out over the 20 problems; every scaler / PCA / model is fit on the
training problems only.

Probe capacity (task 1)
- Layers 24, 32, 40. One StandardScaler + PCA-512 per (fold, layer, variant), fit on all
  stored boundary states of the training problems; pre = h_i and post = h_{i+1} are both
  projected into that basis, and PCA-32 / PCA-128 are its leading components.
- Variants: ``raw`` (as s14) and ``demean`` (each trajectory's mean scaled activation is
  subtracted before PCA, and each trajectory's mean target is subtracted from the training
  targets, so the model can only learn within-trace variation; problem identity removed).
  Test-trace demeaning of features uses that trace's own activations (no labels).
- Models: RidgeCV on PCA-32/128/512; optional RidgeCV on all 5,120 scaled dims (pre only;
  off by default: ~75 s per fit); HistGradientBoosting (150 iters, lr 0.1, early stopping)
  on PCA-32 and PCA-128 (pre and pair only, for runtime); MLP (256) alpha=1 and MLP (512, 256)
  alpha=10 on whitened PCA-128, both with early stopping.
- Feature sets: pre, post and pair = [h_i, h_{i+1}, h_{i+1} − h_i] (each block PCA-k).

Answer identity (task 2)
- own answer = normalize_v2(trajectories.normalized_final_answer), i.e. the trace's own
  eventual final answer; it matches the rollout ``cat`` vocabulary for all 40 traces.
- p_own = share of the 100 resamples equal to the own answer; p_modal = modal share;
  dp_own / dp_modal = change from fork i to i+1.

Ceilings: split-half (even vs odd sample_idx), Spearman–Brown to n = 100, sqrt.
Uncertainty: 1,000-replicate problem-cluster bootstrap (shared draws, so paired
differences are valid) for every layer-32 row and for the pre-specified comparisons.

Output: results/s16_probe_capacity.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from safetensors import safe_open
from scipy.stats import rankdata, spearmanr
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import RidgeCV
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from arv2 import reliability as R  # noqa: E402
from arv2.answers import normalize_v2  # noqa: E402
from arv2.paths import DERIVED, INPUTS, RESULTS, sha256_file, write_json  # noqa: E402

LAYERS = [24, 32, 40]
HEADLINE_LAYER = 32
PCA_MAX = 512
ALPHAS = np.logspace(-1, 7, 17)
TARGETS = ["acc_level", "entropy", "tv_debiased", "signed_dacc",
           "p_own", "p_modal", "dp_own", "dp_modal"]
BOOT = 1000


# ---------------------------------------------------------------- targets (s14 + identity)
def fork_stats(roll: pd.DataFrame, own: dict) -> tuple[dict, int]:
    codes = {c: i for i, c in enumerate(sorted(roll.cat.unique()))}
    roll = roll.assign(code=roll.cat.map(codes))
    out = {}
    for key, g in roll.groupby(["trajectory_id", "fork_idx"]):
        g = g.sort_values("sample_idx")
        c, y = g.code.to_numpy(), g.correct_v2.to_numpy(float)
        o = (g.cat.to_numpy() == own[key[0]]).astype(float)
        h = g.sample_idx.to_numpy() % 2 == 0
        out[key] = {"all": (c, y, o), "A": (c[h], y[h], o[h]), "B": (c[~h], y[~h], o[~h])}
    return out, len(codes)


def entropy(c, k):
    p = np.bincount(c, minlength=k) / len(c)
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


def modal_share(c, k):
    return float(np.bincount(c, minlength=k).max() / len(c))


def tv(a, b, k):
    return 0.5 * float(np.abs(np.bincount(a, minlength=k) / len(a)
                              - np.bincount(b, minlength=k) / len(b)).sum())


def floor(a, b, k):
    p = np.bincount(np.concatenate([a, b]), minlength=k) / (len(a) + len(b))
    return 0.5 * float((np.sqrt(p * (1 - p) * (1 / len(a) + 1 / len(b)))
                        * np.sqrt(2 / np.pi)).sum())


def targets_for(fs, k, tid, i, part):
    (ca, ya, oa), (cb, yb, ob) = fs[(tid, i)][part], fs[(tid, i + 1)][part]
    t, f = tv(ca, cb, k), floor(ca, cb, k)
    return {"acc_level": ya.mean(), "entropy": entropy(ca, k), "tv_debiased": t - f,
            "signed_dacc": yb.mean() - ya.mean(), "p_own": oa.mean(),
            "p_modal": modal_share(ca, k), "dp_own": ob.mean() - oa.mean(),
            "dp_modal": modal_share(cb, k) - modal_share(ca, k)}


# ---------------------------------------------------------------- data
def build_frame() -> tuple[pd.DataFrame, dict]:
    traj = pd.read_parquet(INPUTS / "trajectories.parquet")
    own = {t: normalize_v2(a) for t, a in zip(traj.trajectory_id, traj.normalized_final_answer)}
    sents = pd.read_parquet(INPUTS / "sentences.parquet")
    feats = pd.read_parquet(INPUTS / "mvp_features.parquet")[["sentence_id", "prefix_token_count"]]
    roll = pd.read_parquet(DERIVED / "rollouts_v2.parquet",
                           columns=["trajectory_id", "fork_idx", "sample_idx", "cat",
                                    "correct_v2"])
    vocab = roll.groupby("trajectory_id").cat.agg(set)
    own_info = {t: {"own_answer": a, "in_rollout_vocab": bool(a in vocab[t]),
                    "is_correct": bool(c)}
                for t, a, c in zip(traj.trajectory_id, own.values(), traj.is_correct)}
    fs, k = fork_stats(roll, own)
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
    frame = pd.concat([sents, pd.DataFrame(rows)], axis=1)
    frame["trace_correct"] = frame.trajectory_id.map(lambda t: own_info[t]["is_correct"])
    return frame, own_info


def load_states(feats: pd.DataFrame, cache: Path | None) -> np.ndarray:
    """(n_sentences, n_layers, 5120) float32 for every stored boundary, in ``feats`` order."""
    if cache is not None and cache.exists():
        return np.load(cache)
    manifest = json.loads((INPUTS / "MANIFEST.json").read_text())
    root = Path(manifest["referenced_in_place"]["activation_root"])
    S = np.full((len(feats), len(LAYERS), 5120), np.nan, np.float32)
    for rel, digest in manifest["referenced_in_place"]["shards"].items():
        if sha256_file(root / rel) != digest:
            raise SystemExit(f"activation shard changed: {rel}")
        sub = feats[feats.shard_path == rel]
        with safe_open(root / rel, "pt") as f:
            sl = f.get_slice("resid")
            arr = np.stack([sl[:, L:L + 1].float().numpy()[:, 0] for L in LAYERS], 1)
        S[sub.index.to_numpy()] = arr[sub.row_idx.to_numpy()]
    if np.isnan(S).any():
        raise SystemExit("missing activations")
    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.save(cache, S)
    return S


def group_mean(X: np.ndarray, g: np.ndarray) -> np.ndarray:
    out = np.empty_like(X)
    for v in np.unique(g):
        m = g == v
        out[m] = X[m].mean(0)
    return out


# ---------------------------------------------------------------- metrics
def per_traj_spearman(y, s, traj, keys):
    out = np.full(len(keys), np.nan)
    for j, t in enumerate(keys):
        m = traj == t
        out[j] = R.spearman(y[m], s[m])
    return out


def fast_spearman(a, b):
    ra, rb = rankdata(a), rankdata(b)
    return float(np.corrcoef(ra, rb)[0, 1])


# ---------------------------------------------------------------- E. controls
def partial_within(y, s, z, traj, keys):
    """Per-trajectory partial Spearman of y and s given the columns of z (ranks)."""
    out = np.full(len(keys), np.nan)
    for j, t in enumerate(keys):
        m = traj == t
        ry, rs = rankdata(y[m]), rankdata(s[m])
        Z = np.column_stack([np.ones(m.sum())] + [rankdata(c[m]) for c in z])
        res = lambda v: v - Z @ np.linalg.lstsq(Z, v, rcond=None)[0]  # noqa: E731
        a, b = res(ry), res(rs)
        if a.std() > 0 and b.std() > 0:
            out[j] = float(np.corrcoef(a, b)[0, 1])
    return out


def section_e(frame, Y, states, ip, iq, prob_all, traj_all, tkeys, tkey_correct, tkey_prob,
              ceil, layers) -> dict:
    """Controls for the identity result (fast: ridge only, raw variant).
    1. position-only baseline (s14 features) for every target;
    2. ridge PCA-128 on pre, and pre + position, per layer;
    3. within-trace partial Spearman of the decoded p_own with p_own given position, given
       position + entropy, and given position + p_modal (does it carry more than
       commitment-as-uncertainty and time?);
    4. how the acc_level probe relates to the p_own probe (sign-cancellation check);
    5. trace-correctness decoding: can the state tell the correct trace from the incorrect
       one of the same problem? (row-level AUC and pairwise per-problem accuracy).
    """
    from sklearn.linear_model import LogisticRegressionCV
    from sklearn.metrics import roc_auc_score

    groups, traj = frame.problem_id.to_numpy(), frame.trajectory_id.to_numpy()
    posf = np.column_stack([frame.sentence_idx, np.log1p(frame.sentence_idx),
                            np.log1p(frame.prefix_token_count)]).astype(float)
    rng = np.random.default_rng(R.seed_from("20260914:arv2:s16:boot_e"))
    uprob = np.unique(groups)
    ptraj = {g: np.flatnonzero(tkey_prob == g) for g in uprob}
    boot_traj = [np.concatenate([ptraj[g] for g in rng.choice(uprob, len(uprob))])
                 for _ in range(BOOT)]

    def ci_w(v):
        lo, hi, _ = R.interval(np.array([np.nanmean(v[bt]) for bt in boot_traj]))
        return [lo, hi]

    def lopo(blocks_fn):
        oof = np.full(Y.shape, np.nan)
        for g in uprob:
            tr, te = groups != g, groups == g
            Ftr, Fte = blocks_fn(g, tr, te)
            ys = StandardScaler().fit(Y[tr])
            m = RidgeCV(alphas=ALPHAS, alpha_per_target=True).fit(Ftr, ys.transform(Y[tr]))
            oof[te] = ys.inverse_transform(m.predict(Fte))
        return oof

    def summarise(P):
        res = {}
        for j, n in enumerate(TARGETS):
            pt = per_traj_spearman(Y[:, j], P[:, j], traj, tkeys)
            res[n] = {"overall": R.spearman(Y[:, j], P[:, j]), "within": float(np.nanmean(pt)),
                      "within_ci": ci_w(pt),
                      "within_correct_traces": float(np.nanmean(pt[tkey_correct])),
                      "within_incorrect_traces": float(np.nanmean(pt[~tkey_correct])),
                      "within_ceiling": ceil[n]["within_ceiling_corr"]}
        return res

    out: dict = {}
    pos_sc = lambda tr, te: (StandardScaler().fit(posf[tr]).transform(posf[tr]),  # noqa: E731
                             StandardScaler().fit(posf[tr]).transform(posf[te]))
    P_pos = lopo(lambda g, tr, te: pos_sc(tr, te))
    out["position_only"] = summarise(P_pos)

    out["by_layer"] = {}
    acts_pca = {}
    for L in layers:
        X = states[:, LAYERS.index(L)]
        T_by_fold = {}
        for g in uprob:
            tr_s = prob_all != g
            sc = StandardScaler().fit(X[tr_s])
            Z = sc.transform(X)
            T_by_fold[g] = PCA(128, svd_solver="randomized", random_state=0).fit(
                Z[tr_s]).transform(Z)
        acts_pca[L] = T_by_fold
        P_pre = lopo(lambda g, tr, te: (T_by_fold[g][ip][tr], T_by_fold[g][ip][te]))

        def pre_pos(g, tr, te):
            a, b = pos_sc(tr, te)
            return (np.hstack([T_by_fold[g][ip][tr], a]), np.hstack([T_by_fold[g][ip][te], b]))
        P_pp = lopo(pre_pos)
        jo, ja, je = TARGETS.index("p_own"), TARGETS.index("acc_level"), TARGETS.index("entropy")
        idx, ent, pmod = frame.sentence_idx.to_numpy(float), frame.entropy.to_numpy(), \
            frame.p_modal.to_numpy()
        po = frame.p_own.to_numpy()
        part = {}
        for name, z in (("given_position", [idx]), ("given_position_entropy", [idx, ent]),
                        ("given_position_p_modal", [idx, pmod]),
                        ("given_position_entropy_p_modal", [idx, ent, pmod])):
            v = partial_within(po, P_pre[:, jo], z, traj, tkeys)
            vpos = partial_within(po, P_pos[:, jo], z[1:], traj, tkeys) if len(z) > 1 else None
            part[name] = {"activation_probe": float(np.nanmean(v)), "ci": ci_w(v),
                          "correct_traces": float(np.nanmean(v[tkey_correct])),
                          "incorrect_traces": float(np.nanmean(v[~tkey_correct]))}
            if vpos is not None:
                part[name]["position_probe_same_controls_minus_position"] = float(np.nanmean(vpos))
        # true-target partial: how much of p_own's within-trace variation is NOT explained by
        # position + entropy + p_modal (descriptive context for the partials above)
        acc_vs_own = per_traj_spearman(P_pre[:, ja], P_pre[:, jo], traj, tkeys)
        # oracle sign: use the trace's correctness bit to turn decoded p_own into accuracy
        oracle = np.where(frame.trace_correct.to_numpy(), P_pre[:, jo], -P_pre[:, jo])
        orc = per_traj_spearman(frame.acc_level.to_numpy(), oracle, traj, tkeys)
        out["by_layer"][str(L)] = {
            "pre_ridge_pca128": summarise(P_pre),
            "pre_plus_position_ridge_pca128": summarise(P_pp),
            "p_own_partial_within": part,
            "acc_probe_vs_p_own_probe_within": {
                "all": float(np.nanmean(acc_vs_own)),
                "correct_traces": float(np.nanmean(acc_vs_own[tkey_correct])),
                "incorrect_traces": float(np.nanmean(acc_vs_own[~tkey_correct]))},
            "oracle_sign_acc_from_p_own_probe_within": {
                "note": "uses the trace's correctness label; NOT a valid probe, shows that "
                        "accuracy is recoverable from decoded p_own only given that bit",
                "within": float(np.nanmean(orc)), "ci": ci_w(orc)},
        }
        print(f"controls layer {L}", flush=True)

    # 5. trace correctness from the pre state (row-level, LOPO, logistic on PCA-128)
    tc = frame.trace_correct.to_numpy().astype(int)
    corr = {}
    for L in layers:
        prob = np.full(len(frame), np.nan)
        for g in uprob:
            tr, te = groups != g, groups == g
            F = acts_pca[L][g][ip]
            sc = StandardScaler().fit(F[tr])
            m = LogisticRegressionCV(Cs=np.logspace(-4, 1, 6), cv=5, max_iter=2000,
                                     scoring="roc_auc").fit(sc.transform(F[tr]), tc[tr])
            prob[te] = m.predict_proba(sc.transform(F[te]))[:, 1]
        tm = pd.DataFrame({"t": traj, "g": groups, "c": tc, "p": prob,
                           "late": frame.sentence_idx.to_numpy()
                           >= frame.groupby("trajectory_id").sentence_idx.transform("median")})
        pair = tm.groupby(["g", "c"]).p.mean().unstack()
        pair_late = tm[tm.late].groupby(["g", "c"]).p.mean().unstack()
        within_prob_auc = [roc_auc_score(d.c, d.p) for _, d in tm.groupby("g")]
        corr[str(L)] = {"row_auc_overall": float(roc_auc_score(tc, prob)),
                        "row_auc_within_problem_mean": float(np.mean(within_prob_auc)),
                        "pairwise_correct_trace_scored_higher": float((pair[1] > pair[0]).mean()),
                        "pairwise_late_half": float((pair_late[1] > pair_late[0]).mean()),
                        "n_problems": int(len(pair))}
    out["trace_correctness_decoding"] = corr
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", type=Path, default=None,
                    help="optional .npy cache of the stored states (outside the repo)")
    ap.add_argument("--full-ridge", action="store_true",
                    help="also fit RidgeCV on all 5,120 scaled dims (pre only)")
    ap.add_argument("--layers", type=int, nargs="*", default=LAYERS)
    ap.add_argument("--extra-only", action="store_true",
                    help="recompute only section E (controls) and merge it into the existing "
                         "results JSON; sections A-D are left as written")
    args = ap.parse_args()
    t0 = time.time()
    out: dict = {"stage": "exploratory", "preregistered": False,
                 "analysis": "probe_capacity", "layers": args.layers,
                 "headline_layer": HEADLINE_LAYER, "targets": TARGETS, "bootstrap_reps": BOOT}

    frame, own_info = build_frame()
    out["own_answer_source"] = ("normalize_v2(inputs/trajectories.parquet:normalized_final_answer)"
                                " = the trace's own eventual final answer")
    out["own_answers"] = own_info
    groups, traj = frame.problem_id.to_numpy(), frame.trajectory_id.to_numpy()
    tkeys = np.unique(traj)
    tkey_correct = np.array([own_info[t]["is_correct"] for t in tkeys])
    tkey_prob = np.array([int(t.split(":")[0]) for t in tkeys])
    out["rows"], out["problems"], out["trajectories"] = len(frame), int(len(set(groups))), len(tkeys)
    Y = frame[TARGETS].to_numpy(float)

    # descriptive: how different are p_own and acc_level?
    out["target_relations"] = {
        "spearman_p_own_acc_all": R.spearman(frame.p_own, frame.acc_level),
        "spearman_p_own_acc_incorrect_traces": R.spearman(
            frame.p_own[~frame.trace_correct], frame.acc_level[~frame.trace_correct]),
        "max_abs_diff_p_own_acc_correct_traces": float(
            (frame.p_own - frame.acc_level)[frame.trace_correct].abs().max()),
        "spearman_p_modal_entropy": R.spearman(frame.p_modal, frame.entropy),
        "spearman_dp_own_signed_dacc": R.spearman(frame.dp_own, frame.signed_dacc),
    }

    # ---------------- A. ceilings (overall, within, within by trace kind)
    sb = lambda x: 2 * x / (1 + x) if x > 0 else x  # noqa: E731
    root = lambda x: float(np.sqrt(max(x, 0)))  # noqa: E731
    ceil = {}
    for n in TARGETS:
        a, b = frame[f"A_{n}"].to_numpy(), frame[f"B_{n}"].to_numpy()
        r = R.spearman(a, b)
        pt = per_traj_spearman(a, b, traj, tkeys)
        rw, rwc, rwi = np.nanmean(pt), np.nanmean(pt[tkey_correct]), np.nanmean(pt[~tkey_correct])
        ad, bd = a - group_mean(a[:, None], traj)[:, 0], b - group_mean(b[:, None], traj)[:, 0]
        rd = R.spearman(ad, bd)
        ceil[n] = {"split_half_spearman": r, "ceiling_corr": root(sb(r)),
                   "split_half_within": float(rw), "within_ceiling_corr": root(sb(rw)),
                   "within_ceiling_correct_traces": root(sb(rwc)),
                   "within_ceiling_incorrect_traces": root(sb(rwi)),
                   "pooled_demeaned_ceiling_corr": root(sb(rd))}
    out["ceilings"] = ceil
    print("ceilings", {n: (round(v["ceiling_corr"], 2), round(v["within_ceiling_corr"], 2))
                       for n, v in ceil.items()}, flush=True)

    # ---------------- B. representations
    feats = pd.read_parquet(INPUTS / "mvp_features.parquet").reset_index(drop=True)
    states = load_states(feats, args.cache)
    lidx = {L: LAYERS.index(L) for L in args.layers}
    pos = {(t, i): j for j, (t, i) in enumerate(zip(feats.trajectory_id, feats.sentence_idx))}
    ip = np.array([pos[(t, i)] for t, i in zip(frame.trajectory_id, frame.sentence_idx)])
    iq = np.array([pos[(t, i + 1)] for t, i in zip(frame.trajectory_id, frame.sentence_idx)])
    prob_all, traj_all = feats.problem_id.to_numpy(), feats.trajectory_id.to_numpy()
    print(f"states loaded {states.shape} {time.time() - t0:.0f}s", flush=True)
    ctx = dict(frame=frame, Y=Y, states=states, ip=ip, iq=iq, prob_all=prob_all,
               traj_all=traj_all, tkeys=tkeys, tkey_correct=tkey_correct,
               tkey_prob=tkey_prob, ceil=ceil, layers=args.layers)
    if args.extra_only:
        prev = json.loads((RESULTS / "s16_probe_capacity.json").read_text())
        prev["controls"] = section_e(**ctx)
        prev["preregistered"], prev["stage"] = False, "exploratory"
        write_json(RESULTS / "s16_probe_capacity.json", prev)
        print("extra done", round(time.time() - t0), flush=True)
        return

    Ytraj_mean = group_mean(Y, traj)
    preds: dict[tuple, np.ndarray] = {}

    def put(key, te, val):
        if key not in preds:
            preds[key] = np.full(Y.shape, np.nan)
        preds[key][te] = val

    hgb_kw = dict(max_iter=150, learning_rate=0.1, max_leaf_nodes=15, min_samples_leaf=40,
                  l2_regularization=1.0, early_stopping=True, validation_fraction=0.1,
                  n_iter_no_change=20, random_state=0)
    mlp_cfg = {"mlp256_a1": dict(hidden_layer_sizes=(256,), alpha=1.0),
               "mlp512x256_a10": dict(hidden_layer_sizes=(512, 256), alpha=10.0)}
    for L in args.layers:
        X = states[:, lidx[L]]
        for g in np.unique(groups):
            tr_s = prob_all != g
            tr, te = groups != g, groups == g
            sc = StandardScaler().fit(X[tr_s])
            Z = sc.transform(X).astype(np.float32)
            for variant in ("raw", "demean"):
                Zv = Z if variant == "raw" else Z - group_mean(Z, traj_all)
                p = PCA(PCA_MAX, svd_solver="randomized", random_state=0).fit(Zv[tr_s])
                T = p.transform(Zv)
                if variant == "raw":
                    Yt = Y[tr]
                else:
                    Yt = Y[tr] - Ytraj_mean[tr]
                ys = StandardScaler().fit(Yt)
                Ytr = ys.transform(Yt)
                inv = ys.inverse_transform

                def block(k, fs):
                    a, b = T[ip][:, :k], T[iq][:, :k]
                    return {"pre": a, "post": b, "pair": np.hstack([a, b, b - a])}[fs]

                # ridge on PCA-k
                for k in (32, 128, 512):
                    for fs in ("pre", "post", "pair"):
                        if fs == "pair" and k == 512:
                            continue
                        F = block(k, fs)
                        m = RidgeCV(alphas=ALPHAS, alpha_per_target=True).fit(F[tr], Ytr)
                        put((variant, L, f"ridge_pca{k}", fs), te, inv(m.predict(F[te])))
                if args.full_ridge:
                    F = Zv[ip]
                    m = RidgeCV(alphas=ALPHAS, alpha_per_target=True).fit(F[tr], Ytr)
                    put((variant, L, "ridge_full5120", "pre"), te, inv(m.predict(F[te])))
                # gradient boosting
                for k, sets in ((32, ("pre", "pair")), (128, ("pre", "pair"))):
                    for fs in sets:
                        F = block(k, fs)
                        col = np.column_stack([
                            HistGradientBoostingRegressor(**hgb_kw).fit(F[tr], Ytr[:, j])
                            .predict(F[te]) for j in range(len(TARGETS))])
                        put((variant, L, f"hgb_pca{k}", fs), te, inv(col))
                # MLPs on whitened PCA-128
                for fs in ("pre", "post", "pair"):
                    F = block(128, fs)
                    fsc = StandardScaler().fit(F[tr])
                    Ftr, Fte = fsc.transform(F[tr]), fsc.transform(F[te])
                    for name, cfg in mlp_cfg.items():
                        m = MLPRegressor(**cfg, early_stopping=True, validation_fraction=0.1,
                                         n_iter_no_change=10, max_iter=300, batch_size=256,
                                         learning_rate_init=1e-3, random_state=0).fit(Ftr, Ytr)
                        put((variant, L, name, fs), te, inv(m.predict(Fte)))
            print(f"layer {L} fold {g} {time.time() - t0:.0f}s", flush=True)

    # ---------------- C. evaluation + bootstrap
    rng = np.random.default_rng(R.seed_from("20260914:arv2:s16:boot"))
    uprob = np.unique(groups)
    picks = [rng.choice(uprob, size=len(uprob), replace=True) for _ in range(BOOT)]
    prow = {g: np.flatnonzero(groups == g) for g in uprob}
    ptraj = {g: np.flatnonzero(tkey_prob == g) for g in uprob}
    boot_rows = [np.concatenate([prow[g] for g in pk]) for pk in picks]
    boot_traj = [np.concatenate([ptraj[g] for g in pk]) for pk in picks]

    def boot_within(v):  # v: per-trajectory values
        return np.array([np.nanmean(v[bt]) for bt in boot_traj])

    def ci(d):
        lo, hi, _ = R.interval(np.asarray(d))
        return [lo, hi]

    table, within_store, overall_boot = [], {}, {}
    for key, P in preds.items():
        variant, L, model, fs = key
        for j, n in enumerate(TARGETS):
            y, s = Y[:, j], P[:, j]
            pt = per_traj_spearman(y, s, traj, tkeys)
            within_store[key + (n,)] = pt
            yd, sd = y - Ytraj_mean[:, j], s - group_mean(s[:, None], traj)[:, 0]
            rec = {"variant": variant, "layer": L, "model": model, "features": fs,
                   "target": n,
                   "overall": R.spearman(y, s) if variant == "raw" else None,
                   "pooled_demeaned": R.spearman(yd, sd),
                   "within": float(np.nanmean(pt)),
                   "within_correct_traces": float(np.nanmean(pt[tkey_correct])),
                   "within_incorrect_traces": float(np.nanmean(pt[~tkey_correct])),
                   "ceiling": ceil[n]["ceiling_corr"],
                   "within_ceiling": ceil[n]["within_ceiling_corr"]}
            c, cw = ceil[n]["ceiling_corr"], ceil[n]["within_ceiling_corr"]
            rec["overall_frac_ceiling"] = rec["overall"] / c if rec["overall"] is not None and c > 0 else None
            rec["within_frac_ceiling"] = rec["within"] / cw if cw > 0 else None
            if L == HEADLINE_LAYER:
                rec["within_ci"] = ci(boot_within(pt))
                if variant == "raw":
                    ob = np.array([fast_spearman(y[br], s[br]) for br in boot_rows])
                    overall_boot[key + (n,)] = ob
                    rec["overall_ci"] = ci(ob)
            table.append(rec)
        print(f"evaluated {key} {time.time() - t0:.0f}s", flush=True)
    tab = pd.DataFrame(table)
    out["predictability"] = tab.to_dict("records")

    # ---------------- D. pre-specified paired comparisons (layer 32, bootstrap diffs)
    comps = []
    base = ("raw", HEADLINE_LAYER, "ridge_pca128", "pre")
    for key in preds:
        if key[1] != HEADLINE_LAYER or key == base:
            continue
        for n in TARGETS:
            d = within_store[key + (n,)] - within_store[base + (n,)]
            rec = {"comparison": "within: model minus s14 baseline (raw ridge_pca128 pre)",
                   "variant": key[0], "model": key[2], "features": key[3], "target": n,
                   "diff": float(np.nanmean(d)), "ci": ci(boot_within(d))}
            if key[0] == "raw":
                db = overall_boot[key + (n,)] - overall_boot[base + (n,)]
                rec["overall_diff"] = float(R.spearman(Y[:, TARGETS.index(n)],
                                                       preds[key][:, TARGETS.index(n)])
                                            - R.spearman(Y[:, TARGETS.index(n)],
                                                         preds[base][:, TARGETS.index(n)]))
                rec["overall_diff_ci"] = ci(db)
            comps.append(rec)
    out["vs_baseline_layer32"] = comps

    ident = []
    for key in preds:
        for a, b in (("p_own", "acc_level"), ("dp_own", "signed_dacc")):
            for subset, mask in (("all", np.ones(len(tkeys), bool)),
                                 ("incorrect_traces", ~tkey_correct)):
                d = np.where(mask, within_store[key + (a,)] - within_store[key + (b,)], np.nan)
                rec = {"variant": key[0], "layer": key[1], "model": key[2], "features": key[3],
                       "contrast": f"{a} - {b}", "subset": subset,
                       "within_diff": float(np.nanmean(d)),
                       "within_diff_ci": ci(boot_within(d))}
                # fraction-of-ceiling contrast (within)
                ca = ceil[a]["within_ceiling_corr"] if subset == "all" else ceil[a]["within_ceiling_incorrect_traces"]
                cb = ceil[b]["within_ceiling_corr"] if subset == "all" else ceil[b]["within_ceiling_incorrect_traces"]
                va = np.nanmean(np.where(mask, within_store[key + (a,)], np.nan))
                vb = np.nanmean(np.where(mask, within_store[key + (b,)], np.nan))
                rec.update({"a_within": float(va), "b_within": float(vb),
                            "a_frac_ceiling": float(va / ca) if ca > 0 else None,
                            "b_frac_ceiling": float(vb / cb) if cb > 0 else None})
                if key[0] == "raw" and subset == "all" and key[1] == HEADLINE_LAYER:
                    db = overall_boot[key + (a,)] - overall_boot[key + (b,)]
                    rec["overall_diff"] = float(
                        R.spearman(Y[:, TARGETS.index(a)], preds[key][:, TARGETS.index(a)])
                        - R.spearman(Y[:, TARGETS.index(b)], preds[key][:, TARGETS.index(b)]))
                    rec["overall_diff_ci"] = ci(db)
                ident.append(rec)
    out["identity_vs_correctness"] = ident

    # selection-inflated maxima (descriptive only; flagged as such)
    mx = {}
    for n in TARGETS:
        sub = tab[tab.target == n]
        r = sub.loc[sub.within.idxmax()]
        mx[n] = {"max_within_over_all_rows": float(r.within), "at": [r.variant, int(r.layer),
                                                                     r.model, r.features],
                 "n_rows_searched": int(len(sub)),
                 "max_overall_raw": float(sub[sub.variant == "raw"].overall.max())}
    out["selection_inflated_maxima"] = mx
    out["controls"] = section_e(**ctx)
    out["runtime_s"] = time.time() - t0
    write_json(RESULTS / "s16_probe_capacity.json", out)
    print("done", round(time.time() - t0), flush=True)


if __name__ == "__main__":
    main()
