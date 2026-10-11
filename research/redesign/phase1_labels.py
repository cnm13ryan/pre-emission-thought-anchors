"""Phase 1 (redesign) items 1, 2, 3 and 5 from reacquired rollouts. Exploratory only.

Rules are fixed in ``phase1_analysis_plan.md`` (written before the data were read). No
probe output is read. Writes ``candidate_labels.parquet`` under the redesign data root and
small CSV/JSON tables under ``research/redesign/tables``.

Usage: python research/redesign/phase1_labels.py
"""

from __future__ import annotations

import glob
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path.home() / "preanchor-root"
DATA = ROOT / "data/processed/redesign_phase1"
REPO = Path(__file__).resolve().parents[2]
TABLES = REPO / "research/redesign/tables"
NA = "<NO_ANSWER>"
ALPHAS = [1e-9, 0.5, 1.0]
FROZEN_THRESHOLD = 1.3549810893213545
SIM_THRESHOLD = 0.8
CAP, MARGIN = 16384, 64
SEED_ROOT = 20260914
SPLITS = {"train": [2238, 1591, 4682, 6596, 3448, 2189, 3550, 4605, 2137, 4019, 3360, 3935],
          "validation": [330, 4164, 2050, 6481], "test": [6998, 2236, 2870, 3916]}
SPLIT_OF = {p: s for s, ps in SPLITS.items() for p in ps}


def q90(values) -> float:
    """Frozen interpolation of ``labels/threshold.py`` (numpy linear)."""
    return float(np.quantile(np.asarray(values, dtype=float), 0.90))


def na_class(r) -> str | None:
    if r.category != NA:
        return None
    if r.has_error or r.rollout_chars == 0:
        return "sample_error"
    if r.boxed_count > 0:
        return "parse_failure"
    if r.rollout_tokens is not None and r.rollout_tokens >= CAP - MARGIN:
        return "length_truncated"
    return "genuine_non_answer"


def kl_parts(without: list[str], with_: list[str], alpha: float) -> dict:
    p_c, q_c = Counter(without), Counter(with_)
    cats = sorted(set(p_c) | set(q_c) | {NA})
    k, n_p, n_q = len(cats), len(without), len(with_)
    contrib = {}
    for a in cats:
        p = (p_c[a] + alpha) / (n_p + alpha * k)
        q = (q_c[a] + alpha) / (n_q + alpha * k)
        contrib[a] = p * math.log(p / q)
    total = sum(contrib.values())
    absent = [a for a in cats if q_c[a] == 0 and p_c[a] > 0]
    return {"I": max(total, 0.0), "contrib": contrib, "absent": absent}


def tv(a: list[str], b: list[str]) -> float:
    ca, cb = Counter(a), Counter(b)
    return 0.5 * sum(abs(ca[x] / len(a) - cb[x] / len(b)) for x in set(ca) | set(cb))


def half_mask(trajectory: str, fork: int, n: int) -> np.ndarray:
    seed = int(hashlib.sha256(f"{SEED_ROOT}:split_half:{trajectory}:{fork}".encode())
               .hexdigest()[:8], 16)
    perm = np.random.default_rng(seed).permutation(n)
    mask = np.zeros(n, dtype=bool)
    mask[perm[: n // 2]] = True
    return mask


def cohen_kappa(x: np.ndarray, y: np.ndarray) -> float:
    x, y = x.astype(bool), y.astype(bool)
    po = float(np.mean(x == y))
    pe = float(x.mean() * y.mean() + (1 - x.mean()) * (1 - y.mean()))
    return (po - pe) / (1 - pe) if pe < 1 else float("nan")


def load_rollouts() -> pd.DataFrame:
    files = sorted(glob.glob(str(DATA / "rollouts/*/fork_*.parquet")))
    cols = ["trajectory_id", "fork_idx", "sample_idx", "has_error", "category", "is_correct",
            "boxed_count", "rollout_tokens", "rollout_chars", "has_think_close"]
    frame = pd.concat([pd.read_parquet(f, columns=cols) for f in files], ignore_index=True)
    sim = pd.read_parquet(DATA / "similarity.parquet")
    frame = frame.merge(sim, on=["trajectory_id", "fork_idx", "sample_idx"], how="left",
                        validate="one_to_one")
    frame["rollout_tokens"] = frame.rollout_tokens.astype("float")
    frame["na_class"] = [na_class(r) for r in frame.itertuples()]
    return frame


def main() -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    roll = load_rollouts()
    labels = pd.read_parquet(ROOT / "data/processed/stage4/labels.parquet")
    labels["split"] = labels.problem_id.map(SPLIT_OF)
    forks: dict[tuple[str, int], dict] = {}
    for (tid, fork), g in roll.groupby(["trajectory_id", "fork_idx"], sort=False):
        g = g.sort_values("sample_idx")
        forks[(tid, int(fork))] = {
            "cat": g.category.tolist(), "sim": g.similarity.to_numpy(),
            "na": g.na_class.tolist(), "correct": g.is_correct.to_numpy(),
            "half": half_mask(tid, int(fork), len(g))}

    # ---------------- item 2: <NO_ANSWER> classes per fork --------------------------
    na_rows = roll[roll.category == NA]
    by_fork = (roll.assign(na_class=roll.na_class.fillna("answered"))
               .pivot_table(index=["trajectory_id", "fork_idx"], columns="na_class",
                            values="sample_idx", aggfunc="count", fill_value=0)
               .reset_index())
    for c in ["answered", "sample_error", "parse_failure", "length_truncated",
              "genuine_non_answer"]:
        if c not in by_fork:
            by_fork[c] = 0
    by_fork.to_csv(TABLES / "no_answer_by_fork.csv", index=False)
    token_hist = np.histogram(na_rows.rollout_tokens.dropna(),
                              bins=[0, 2000, 4000, 8000, 12000, 16000, CAP - MARGIN, CAP + 1,
                                    10 ** 9])

    # ---------------- items 1 and 3: per-sentence recomputation ---------------------
    out = []
    for r in labels.itertuples():
        F, G = forks.get((r.trajectory_id, r.sentence_idx)), \
            forks.get((r.trajectory_id, r.sentence_idx + 1))
        rec = {"sentence_id": r.sentence_id, "status": "OK"}
        if F is None or G is None:
            rec["status"] = "MISSING_FORK"
            out.append(rec)
            continue
        if np.isnan(F["sim"]).any():
            rec["status"] = "NON_FINITE"
            out.append(rec)
            continue
        keep = F["sim"] < SIM_THRESHOLD
        without = [c for c, k in zip(F["cat"], keep, strict=True) if k]
        rec["fork_tv"] = tv(F["cat"], G["cat"])
        rec["dissimilar_count_recomputed"] = int(keep.sum())
        rec["na_frac_fork_i"] = float(np.mean([c == NA for c in F["cat"]]))
        rec["na_frac_with"] = float(np.mean([c == NA for c in G["cat"]]))
        na_without = Counter(n for n, k in zip(F["na"], keep, strict=True) if k and n)
        for c in ["sample_error", "parse_failure", "length_truncated", "genuine_non_answer"]:
            rec[f"without_{c}"] = na_without.get(c, 0)
        hA, hB = F["half"], G["half"]
        for tag, fm, gm in (("A", hA, hB), ("B", ~hA, ~hB)):
            w_h = [c for c, k, m in zip(F["cat"], keep, fm, strict=True) if k and m]
            g_h = [c for c, m in zip(G["cat"], gm, strict=True) if m]
            f_h = [c for c, m in zip(F["cat"], fm, strict=True) if m]
            rec[f"half{tag}_tv"] = tv(f_h, g_h) if f_h and g_h else None
            for alpha in (1e-9, 0.5):
                rec[f"half{tag}_kl_{alpha:g}"] = kl_parts(w_h, g_h, alpha)["I"] \
                    if w_h and g_h else None
        if not without:
            rec["status"] = "NO_SUPPORT"
            out.append(rec)
            continue
        rec["na_frac_without"] = float(np.mean([c == NA for c in without]))
        for alpha in ALPHAS:
            parts = kl_parts(without, G["cat"], alpha)
            tag = f"{alpha:g}"
            I = parts["I"]
            absent_sum = sum(parts["contrib"][a] for a in parts["absent"])
            absent_na = sum(parts["contrib"][a] for a in parts["absent"] if a == NA)
            rec[f"I_{tag}"] = I
            rec[f"absent_sum_{tag}"] = absent_sum
            rec[f"absent_na_sum_{tag}"] = absent_na
            rec[f"absent_share_{tag}"] = absent_sum / I if I > 0 else None
            rec[f"I_minus_absent_{tag}"] = I - absent_sum
            rec[f"n_absent_{tag}"] = len(parts["absent"])
        out.append(rec)
    res = labels.merge(pd.DataFrame(out), on="sentence_id", how="left", validate="one_to_one")

    # Reproduction gate (R1)
    elig = res[res.eligible]
    ok = elig[elig.status == "OK"]
    rho = float(spearmanr(ok.importance_canonical, ok["I_1e-09"]).statistic)
    flags_equal = float(np.mean((ok.importance_canonical >= FROZEN_THRESHOLD)
                                == (ok["I_1e-09"] >= FROZEN_THRESHOLD)))
    gate = {"eligible": int(len(elig)), "recomputed_ok": int(len(ok)),
            "status_counts_eligible": elig.status.value_counts().to_dict(),
            "spearman_vs_stored": rho, "anchor_flag_agreement": flags_equal,
            "dissimilar_count_exact": float(np.mean(ok.dissimilar_count
                                                    == ok.dissimilar_count_recomputed)),
            "max_abs_I_diff": float((ok.importance_canonical - ok["I_1e-09"]).abs().max()),
            "alpha_0_5_max_abs_diff": float((ok.importance_canonical_alpha_0_5
                                             - ok["I_0.5"]).abs().max()),
            "alpha_1_max_abs_diff": float((ok.importance_canonical_alpha_1_0
                                           - ok["I_1"]).abs().max())}
    gate["passed"] = bool(rho >= 0.999 and flags_equal >= 0.995)

    # Item 1: anchors driven by absent answers
    train_ok = ok[ok.split == "train"]
    thresholds = {"1e-09": FROZEN_THRESHOLD,
                  "0.5": q90(train_ok["I_0.5"]), "1": q90(train_ok["I_1"])}
    item1 = []
    for tag, thr in thresholds.items():
        for scope, frame in (("all_eligible", ok), ("train", train_ok)):
            anchors = frame[frame[f"I_{tag}"] >= thr]
            non = frame[frame[f"I_{tag}"] < thr]
            driven = anchors[f"absent_share_{tag}"] >= 0.5
            driven_na = anchors[f"absent_na_sum_{tag}"] >= 0.5 * anchors[f"I_{tag}"]
            dependent = anchors[f"I_minus_absent_{tag}"] < thr
            item1.append({"alpha": tag, "threshold": thr, "scope": scope,
                          "rows": len(frame), "anchors": len(anchors),
                          "driven_by_absent": int(driven.sum()),
                          "driven_share": float(driven.mean()) if len(anchors) else None,
                          "driven_by_absent_no_answer": int(driven_na.sum()),
                          "driven_by_absent_real_answer": int((driven & ~driven_na).sum()),
                          "dependent_on_absent": int(dependent.sum()),
                          "dependent_share": float(dependent.mean()) if len(anchors) else None,
                          "nonanchor_driven_share": float(
                              (non[f"absent_share_{tag}"].fillna(0) >= 0.5).mean())})
        # the frozen α=1e-9 anchor set examined under this α's decomposition
        frozen = ok[ok.importance_canonical >= FROZEN_THRESHOLD]
        item1.append({"alpha": tag, "threshold": thr, "scope": "frozen_1e-9_anchor_set",
                      "rows": len(ok), "anchors": len(frozen),
                      "driven_by_absent": int((frozen[f"absent_share_{tag}"] >= 0.5).sum()),
                      "driven_share": float((frozen[f"absent_share_{tag}"] >= 0.5).mean()),
                      "still_anchor_at_this_alpha": int((frozen[f"I_{tag}"] >= thr).sum())})
    pd.DataFrame(item1).to_csv(TABLES / "item1_absent_answer_drivers.csv", index=False)

    # Item 3: candidate labels on the frozen eligible set
    res["label_a_kl"] = res.importance_canonical.where(res.eligible)
    res["label_b_kl_kt"] = res["I_0.5"].where(res.eligible & (res.status == "OK"))
    res["label_c_fork_tv"] = res.fork_tv.where(res.eligible)
    res["label_d_log1p"] = np.log1p(res.label_a_kl)
    thr_c = {"label_a_kl": FROZEN_THRESHOLD,
             "label_b_kl_kt": q90(res.loc[res.eligible & (res.split == "train"),
                                          "label_b_kl_kt"].dropna()),
             "label_c_fork_tv": q90(res.loc[res.eligible & (res.split == "train"),
                                            "label_c_fork_tv"].dropna())}
    for name, thr in thr_c.items():
        res[f"{name}_anchor"] = (res[name] >= thr).where(res[name].notna())
    # (d) is continuous; it shares (a)'s ordering, so its binarisation is (a).
    item3 = []
    a_flag = res.label_a_kl_anchor == True  # noqa: E712
    for name in ["label_a_kl", "label_b_kl_kt", "label_c_fork_tv", "label_d_log1p"]:
        e = res[res[name].notna()]
        row = {"label": name, "rows": len(e), "missing": int(res.eligible.sum() - len(e)),
               "spearman_vs_a": float(spearmanr(e[name], e.label_a_kl).statistic),
               "spearman_vs_na_frac_without": float(spearmanr(
                   e[name], e.na_frac_without, nan_policy="omit").statistic),
               "median": float(e[name].median()), "p90_all": float(e[name].quantile(.9))}
        flag = f"{name}_anchor"
        if flag in res:
            f = res[flag] == True  # noqa: E712
            row["threshold_train_q90"] = thr_c[name]
            for s in SPLITS:
                sub = res[(res.split == s) & res[name].notna()]
                row[f"prevalence_{s}"] = float((sub[flag] == True).mean())  # noqa: E712
            row["jaccard_vs_a"] = float((f & a_flag).sum() / (f | a_flag).sum())
            anchors = res[f]
            row["anchors_with_any_no_answer_without"] = float(
                (anchors.na_frac_without.fillna(0) > 0).mean())
        item3.append(row)
    pd.DataFrame(item3).to_csv(TABLES / "item3_candidate_labels.csv", index=False)

    # Supplementary split-half reliability (existing samples; N≈50 per half)
    rel = []
    for name, ca, cb in (("kl_1e-9", "halfA_kl_1e-09", "halfB_kl_1e-09"),
                         ("kl_kt", "halfA_kl_0.5", "halfB_kl_0.5"),
                         ("fork_tv", "halfA_tv", "halfB_tv")):
        e = res[res.eligible & res[ca].notna() & res[cb].notna()]
        tr = e[e.split == "train"]
        fa, fb = e[ca] >= q90(tr[ca]), e[cb] >= q90(tr[cb])
        rel.append({"label": name, "rows": len(e),
                    "dropped_no_support": int(res.eligible.sum() - len(e)),
                    "spearman_halves": float(spearmanr(e[ca], e[cb]).statistic),
                    "spearman_halves_log1p": float(spearmanr(np.log1p(e[ca]),
                                                             np.log1p(e[cb])).statistic),
                    "pearson_log1p": float(np.corrcoef(np.log1p(e[ca]), np.log1p(e[cb]))[0, 1]),
                    "kappa_q90_flags": cohen_kappa(fa.to_numpy(), fb.to_numpy())})
    pd.DataFrame(rel).to_csv(TABLES / "supp_split_half_reliability.csv", index=False)

    # ---------------- item 5: inclusion rules available before S_i ------------------
    rules = []
    per_traj = []
    for tid, g in res.groupby("trajectory_id"):
        g = g.sort_values("sentence_idx").copy()
        acc = [0.5 if (a is None or (isinstance(a, float) and math.isnan(a))) else a
               for a in g.fork_accuracy]
        extreme = [(0 <= a <= .01) or (.99 <= a <= 1) for a in acc]
        low = [0 <= a <= .01 for a in acc]
        high = [.99 <= a <= 1 for a in acc]
        n = len(acc)
        # pre-S_i convergence: forks i-4..i all low or all high (fork i precedes S_i)
        conv_pre = [i >= 4 and (all(low[i - 4:i + 1]) or all(high[i - 4:i + 1]))
                    for i in range(n)]
        first = next((i for i, c in enumerate(conv_pre) if c), None)
        g["rule_convergence_pre"] = [first is not None and i >= first for i in range(n)]
        boxed = ["\\boxed{" in t for t in g.text]
        seen = np.cumsum(boxed) - np.array(boxed, dtype=int)  # boxed strictly before i
        g["rule_final_answer_pre"] = seen > 0
        g["rule_fork_i_missing"] = [(tid, i) not in forks for i in g.sentence_idx]
        g["rule_fork_next_missing"] = [(tid, i + 1) not in forks for i in g.sentence_idx]
        g["rule_alignment"] = ~g.alignment_ok
        g["rule_first_token"] = (g.sentence_idx == 0) & (g.char_start == 0)
        g["rule_final_answer_orig"] = g.final_answer_idx.notna() & \
            (g.sentence_idx >= g.final_answer_idx)
        g["rule_convergence_orig"] = g.convergence_idx.notna() & \
            (g.sentence_idx >= g.convergence_idx)
        g["rule_no_dissimilar"] = g.status == "NO_SUPPORT"
        per_traj.append(g)
        del extreme
    rr = pd.concat(per_traj)
    order_orig = ["rule_alignment", "rule_first_token", "rule_final_answer_orig",
                  "rule_fork_next_missing", "rule_convergence_orig", "rule_no_dissimilar"]
    order_pre = ["rule_alignment", "rule_first_token", "rule_final_answer_pre",
                 "rule_fork_i_missing", "rule_convergence_pre"]
    for order, tag in ((order_orig, "original"), (order_pre, "pre_emission")):
        remaining = np.ones(len(rr), dtype=bool)
        for rule in order:
            hit = rr[rule].to_numpy(dtype=bool)
            rules.append({"set": tag, "rule": rule, "marginal_removed": int(hit.sum()),
                          "sequential_removed": int((hit & remaining).sum())})
            remaining &= ~hit
        rules.append({"set": tag, "rule": "REMAINING", "marginal_removed": None,
                      "sequential_removed": int(remaining.sum())})
        rr[f"included_{tag}"] = remaining
    rules_df = pd.DataFrame(rules)
    rules_df.to_csv(TABLES / "item5_inclusion_rules.csv", index=False)
    pre = rr[rr.included_pre_emission]
    label_status_pre = {
        "included_pre_emission": int(len(pre)),
        "label_missing_fork_next": int(pre.rule_fork_next_missing.sum()),
        "label_no_dissimilar_support": int(pre.rule_no_dissimilar.sum()),
        "kl_label_defined": int((pre.status == "OK").sum()),
        "fork_tv_defined": int(pre.fork_tv.notna().sum()),
        "also_in_frozen_eligible": int(pre.eligible.sum()),
        "frozen_eligible_not_in_pre": int((rr.eligible & ~rr.included_pre_emission).sum()),
        "new_rows_vs_frozen": int((rr.included_pre_emission & ~rr.eligible).sum()),
        "new_rows_kl_anchor_1e-9": int(((rr.included_pre_emission & ~rr.eligible)
                                         & (rr["I_1e-09"] >= FROZEN_THRESHOLD)).sum())}

    # ---------------- persist ---------------------------------------------------------
    keep_cols = [c for c in rr.columns if c not in ("text",)]
    rr[keep_cols].to_parquet(DATA / "candidate_labels.parquet", index=False)
    summary = {"stage": "exploratory", "preregistered": False,
               "scope": "20 MVP problems, all previously seen",
               "rollouts": int(len(roll)), "forks": len(forks),
               "similarity_nan": int(roll.similarity.isna().sum()),
               "reproduction_gate": gate, "alpha_thresholds_train_q90": thresholds,
               "candidate_thresholds_train_q90": thr_c,
               "no_answer_totals": na_rows.na_class.value_counts().to_dict(),
               "no_answer_total": int(len(na_rows)),
               "no_answer_token_histogram": {"bins": token_hist[1].tolist(),
                                             "counts": token_hist[0].tolist()},
               "no_answer_tokens_missing": int(na_rows.rollout_tokens.isna().sum()),
               "no_answer_has_think_close": na_rows.groupby("na_class")
               .has_think_close.mean().to_dict(),
               "forks_with_any_no_answer": int((by_fork[["sample_error", "parse_failure",
                                                         "length_truncated",
                                                         "genuine_non_answer"]].sum(1) > 0)
                                               .sum()),
               "inclusion_label_status_pre": label_status_pre}
    (TABLES / "phase1_labels_summary.json").write_text(json.dumps(summary, indent=1,
                                                                  default=float))
    print(json.dumps(summary, indent=1, default=float))


if __name__ == "__main__":
    main()
