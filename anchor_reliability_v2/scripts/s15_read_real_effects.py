"""EXPLORATORY qualitative reading of the 7 sentences with a real effect (s13 BH 10%).

For each flagged sentence i: context sentences i-3..i+3, prompt tails at forks i and i+1
with boundary checks, pooled answer distributions (replicates 0, 1, 2), per-replicate
accuracy, and sample replacement sentences with their answers at fork i.

Prints a plain-text dump to stdout (used to write docs/real_effects_reading.md).
CPU only. Usage:
    python s15_read_real_effects.py --index <phase2 completions_index.parquet> [--samples 8]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from arv2 import answers as A  # noqa: E402
from arv2.paths import DERIVED, INPUTS, RESULTS  # noqa: E402

PHASE2_INPUTS = Path("~/preanchor-root/data/processed/redesign_phase2/inputs.parquet").expanduser()
EXTRA = {"2238:incorrect": [27, 28]}   # forks around the spike to tabulate


def short(s: str, n: int = 260) -> str:
    s = " ".join(str(s).split())
    return s if len(s) <= n else s[:n] + " ..."


def load(index: Path) -> pd.DataFrame:
    trajs = pd.read_parquet(INPUTS / "trajectories.parquet")
    gt = {t: A.normalize_v2(g) for t, g in zip(trajs.trajectory_id, trajs.ground_truth_answer)}
    idx = pd.read_parquet(index)
    idx = idx[idx.status == "OK"].copy()
    parsed = [A.parse_v2(t, finish_reason=f, token_count=c) for t, f, c in
              zip(idx.raw_answer, idx.finish_reason, idx.generated_token_count)]
    idx["cat"] = [A.category_v2(p) for p in parsed]
    idx["correct"] = [p.answer is not None and p.answer == gt[t]
                      for p, t in zip(parsed, idx.trajectory_id)]
    rel = pd.read_parquet(DERIVED / "rollouts_v2.parquet",
                          columns=["trajectory_id", "fork_idx", "sample_idx", "cat", "correct_v2"])
    keys = set(zip(idx.trajectory_id, idx.fork_idx))
    rel = rel[[k in keys for k in zip(rel.trajectory_id, rel.fork_idx)]]
    rel = rel.rename(columns={"correct_v2": "correct"}).assign(replicate=0)
    cols = ["replicate", "trajectory_id", "fork_idx", "sample_idx", "cat", "correct"]
    new = idx[cols + ["raw_answer", "first_replacement_sentence", "finish_reason",
                      "generated_token_count"]]
    return pd.concat([new, rel[cols]], ignore_index=True), gt


def boundary_report(prompt: str, target: str, nxt_prompt: str | None) -> list[str]:
    out = [f"  prompt ends with repr: {prompt[-60:]!r}"]
    out.append(f"  last char: {prompt[-1:]!r}; target starts: {target[:40]!r}")
    if nxt_prompt is not None:
        ok = nxt_prompt.startswith(prompt)
        added = nxt_prompt[len(prompt):] if ok else None
        out.append(f"  fork i+1 prompt extends fork i prompt: {ok}")
        if ok:
            out.append(f"  text added between forks: {added!r}")
            out.append(f"  added == target_text: {added == target}; "
                       f"added.strip() == target.strip(): {added.strip() == target.strip()}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", type=Path, required=True)
    ap.add_argument("--samples", type=int, default=8)
    args = ap.parse_args()
    pd.set_option("display.width", 200)
    eff = pd.read_csv(RESULTS / "s13_sentence_effects.csv")
    real = eff[eff.real_tv_bh10 | eff.real_dacc_bh10]
    sents = pd.read_parquet(INPUTS / "sentences.parquet")
    inp = pd.read_parquet(PHASE2_INPUTS)
    s, gt = load(args.index)
    rng = np.random.default_rng(0)

    print("FLAGGED:\n", real.to_string(index=False), "\n")
    # Duplicate-sentence check over whole trajectories.
    for tid in real.trajectory_id.unique():
        st = sents[sents.trajectory_id == tid]
        dup = st[st.text.str.strip().duplicated(keep=False)]
        print(f"{tid}: gt={gt[tid]!r}; duplicated sentence texts: "
              f"{dup.sentence_idx.tolist()}")
    print()

    for _, r in real.iterrows():
        tid, i = r.trajectory_id, int(r.sentence_idx)
        st = sents[sents.trajectory_id == tid].set_index("sentence_idx")
        print("=" * 100)
        print(f"{tid}  sentence {i}  acc {r.acc_before:.3f} -> {r.acc_after:.3f}  "
              f"tv={r.pooled_fork_tv:.3f}  gt={gt[tid]!r}")
        for j in range(i - 3, i + 4):
            if j in st.index:
                mark = ">>" if j == i else "  "
                print(f"{mark} S{j}: {short(st.loc[j, 'text'], 400)}")
        fi = inp[(inp.trajectory_id == tid) & (inp.fork_idx == i)]
        fj = inp[(inp.trajectory_id == tid) & (inp.fork_idx == i + 1)]
        if len(fi):
            p, tgt = fi.prompt.iloc[0], fi.target_text.iloc[0]
            print(f"\n  target_text(fork {i}): {tgt!r}")
            print(f"  sentences.parquet S{i} == target_text: {st.loc[i, 'text'] == tgt}")
            print(f"  --- fork {i} prompt tail ---\n{p[-400:]!r}")
            if len(fj):
                print(f"  --- fork {i+1} prompt tail ---\n{fj.prompt.iloc[0][-400:]!r}")
            print("\n".join(boundary_report(p, tgt, fj.prompt.iloc[0] if len(fj) else None)))
        forks = [i - 1, i, i + 1, i + 2] + EXTRA.get(tid, [])
        print("\n  accuracy per replicate (n):")
        for f in sorted(set(forks)):
            g = s[(s.trajectory_id == tid) & (s.fork_idx == f)]
            if not len(g):
                continue
            per = g.groupby("replicate").correct.agg(["mean", "size"])
            cells = "  ".join(f"r{k}={v['mean']:.3f}({int(v['size'])})" for k, v in per.iterrows())
            print(f"    fork {f}: pooled={g.correct.mean():.3f} ({len(g)})  {cells}")
        for f in (i, i + 1):
            g = s[(s.trajectory_id == tid) & (s.fork_idx == f)]
            vc = g.cat.value_counts(normalize=True).head(6)
            print(f"  fork {f} top answers: " + ", ".join(f"{k!r}:{v:.2f}" for k, v in vc.items()))
        # Replacement sentences at fork i (new run only).
        g = s[(s.trajectory_id == tid) & (s.fork_idx == i) & (s.replicate > 0)]
        same = (g.first_replacement_sentence.fillna("").str.strip()
                == str(st.loc[i, "text"]).strip()).mean()
        print(f"\n  fork {i}: share of replacement sentences identical to S{i}: {same:.2f}")
        for corr in (True, False):
            sub = g[g.correct == corr]
            if not len(sub):
                continue
            pick = sub.iloc[rng.choice(len(sub), min(args.samples // 2, len(sub)), replace=False)]
            print(f"  -- samples correct={corr} (of {len(sub)}) --")
            for _, x in pick.iterrows():
                print(f"    [{x['cat']!r}] first: {short(x.first_replacement_sentence, 220)}")
                print(f"        cont: {short(x.raw_answer, 420)}")
        print()
    extras(s, eff)



# ----------------------------------------------------------------------------- extras
KEYWORDS = {
    "2238:incorrect": {"regardless": r"regardless", "sum_of_coeffs": r"sum of (all )?(the )?coeff",
                       "constant/c^0": r"constant term|c\^0|c\^\{0\}|exponents? .*cancel"},
    "3935:correct": {"decimal_2.74x": r"2\.74", "exact_fraction": r"exact|fraction|2001/729|2001 / 729"},
    "330:correct": {"1_plus_3_times": r"1 plus 3|adding 1|plus 3 times|1 \+ 3"},
}
WRONG = {"3935:correct": ["\\frac53703125", "429.624"], "2238:incorrect": ["384"],
         "330:correct": ["36084", "29523", "265719"]}


def extras(s: pd.DataFrame, eff: pd.DataFrame) -> None:
    import re
    print("=" * 100, "\nEXTRAS")
    # Replicate offset: released (r0) vs new (r1, r2) accuracy, per trajectory, all forks.
    acc = s.groupby(["trajectory_id", "replicate"]).correct.mean().unstack()
    print("mean accuracy by replicate over all forks:\n", acc.round(3).to_string())
    for tid, kw in KEYWORDS.items():
        print(f"\n{tid}: first-replacement keyword shares and conditional accuracy (new run)")
        for f in sorted(s[s.trajectory_id == tid].fork_idx.unique()):
            g = s[(s.trajectory_id == tid) & (s.fork_idx == f) & (s.replicate > 0)]
            fr = g.first_replacement_sentence.fillna("")
            cells = []
            for name, pat in kw.items():
                m = fr.str.contains(pat, regex=True)
                a = g.correct[m].mean() if m.any() else float("nan")
                cells.append(f"{name}={m.mean():.2f}(acc {a:.2f})")
            pooled = s[(s.trajectory_id == tid) & (s.fork_idx == f)]
            w = pooled.cat.isin(WRONG[tid]).mean()
            print(f"  fork {f:2d}: acc={pooled.correct.mean():.3f} wrong{WRONG[tid]}={w:.2f}  "
                  + "  ".join(cells))
    # Outcome uncertainty of flagged vs other sentences.
    e = eff.copy()
    e["var_before"] = e.acc_before * (1 - e.acc_before)
    e["flag"] = e.real_tv_bh10 | e.real_dacc_bh10
    e["mid"] = e.acc_before.between(0.2, 0.8)
    print("\nacc_before p(1-p): flagged vs rest\n",
          e.groupby("flag").var_before.describe().round(3).to_string())
    print("share of sentences with 0.2<=acc_before<=0.8: flagged",
          round(e[e.flag].mid.mean(), 2), "rest", round(e[~e.flag].mid.mean(), 2))
    print("flagged per trajectory / sentences per trajectory:\n",
          e.groupby("trajectory_id").agg(n=("flag", "size"), flagged=("flag", "sum"),
                                         mean_acc=("acc_before", "mean")).round(3).to_string())


if __name__ == "__main__":
    main()
