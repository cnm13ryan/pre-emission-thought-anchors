"""Phase 2 fork selection by a rule fixed before any Phase 2 generation. Exploratory.

Rule:
1. Candidate problems are the 16 non-test MVP problems; the 4 test problems are untouched.
2. Draw 10 problems without replacement, and one trace kind per problem, from
   ``numpy.random.default_rng(sha256("20260914:phase2_selection")[:8])``.
3. In that trajectory, the window is 20 contiguous forks [s, s+19]. Every fork must exist,
   and every sentence s..s+18 must precede the frozen K=5 convergence index and the
   final-answer sentence. s is drawn uniformly from the admissible starts with the same
   generator. A trajectory with no admissible start is replaced by the next draw, and the
   replacement is recorded.
4. Each window yields 19 sentence labels (forks i and i+1 both rerun) and 20 fork-level
   answer distributions.

Also projects generated tokens from the released rollouts of the selected forks
(chars → tokens, calibrated on rollouts with a measured token count).

Usage: python research/redesign/phase2_select.py
"""

from __future__ import annotations

import glob
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path.home() / "preanchor-root"
DATA = ROOT / "data/processed/redesign_phase1"
OUT = Path(__file__).resolve().parent / "phase2_forks.json"
TEST = {6998, 2236, 2870, 3916}
N_PROBLEMS, WINDOW = 10, 20


def main() -> None:
    labels = pd.read_parquet(ROOT / "data/processed/stage4/labels.parquet")
    trajs = pd.read_parquet(ROOT / "data/processed/stage3/trajectories.parquet")
    rng = np.random.default_rng(
        int(hashlib.sha256(b"20260914:phase2_selection").hexdigest()[:8], 16))
    problems = sorted(set(labels.problem_id) - TEST)
    order = rng.permutation(problems).tolist()
    kinds = rng.choice(["correct", "incorrect"], size=len(order)).tolist()
    chosen, skipped = [], []
    for problem, kind in zip(order, kinds, strict=True):
        if len(chosen) == N_PROBLEMS:
            break
        g = labels[labels.trajectory_id == f"{problem}:{kind}"].sort_values("sentence_idx")
        n = len(g)
        conv = g.convergence_idx.iloc[0]
        final = g.final_answer_idx.iloc[0]
        limit = min(x for x in (n - 1, conv, final) if pd.notna(x))   # sentences < limit
        # sentences s..s+18 < limit and fork s+19 <= n-1 exists
        starts = [s for s in range(0, n) if s + WINDOW - 2 < limit and s + WINDOW - 1 <= n - 1]
        if not starts:
            skipped.append({"problem_id": problem, "trace_kind": kind,
                            "reason": "no admissible window"})
            continue
        s = int(rng.choice(starts))
        chosen.append({"problem_id": int(problem), "trace_kind": kind,
                       "trajectory_id": f"{problem}:{kind}", "start_fork": s,
                       "forks": list(range(s, s + WINDOW)),
                       "sentence_labels": list(range(s, s + WINDOW - 1)),
                       "admissible_starts": len(starts)})
    if len(chosen) < N_PROBLEMS:
        raise SystemExit("Fewer than 10 admissible trajectories")

    # token projection from released rollouts of the selected forks
    files = sorted(glob.glob(str(DATA / "rollouts/*/fork_*.parquet")))
    cols = ["trajectory_id", "fork_idx", "rollout_chars", "rollout_tokens"]
    roll = pd.concat([pd.read_parquet(f, columns=cols) for f in files], ignore_index=True)
    measured = roll[roll.rollout_tokens.notna() & (roll.rollout_tokens < 16000)
                    & (roll.rollout_chars > 0)]
    ratio = float(measured.rollout_tokens.sum() / measured.rollout_chars.sum())
    sel = pd.concat([roll[(roll.trajectory_id == c["trajectory_id"])
                          & roll.fork_idx.isin(c["forks"])] for c in chosen])
    est = np.minimum(sel.rollout_chars * ratio, 16384)
    prefix_tokens = {}
    for c in chosen:
        full = trajs.loc[trajs.trajectory_id == c["trajectory_id"], "full_cot"].item()
        last = labels[(labels.trajectory_id == c["trajectory_id"])
                      & (labels.sentence_idx == c["forks"][-1])].char_start.item()
        prefix_tokens[c["trajectory_id"]] = round(len(full[:last]) * ratio)
    per_traj = (est.groupby(sel.trajectory_id).mean()).round(0).to_dict()
    projection = {"chars_per_token_ratio_tokens_per_char": ratio,
                  "released_rollouts_in_selected_forks": int(len(sel)),
                  "mean_tokens_per_rollout_est": float(est.mean()),
                  "median_tokens_per_rollout_est": float(np.median(est)),
                  "share_at_cap_est": float((est >= 16384).mean()),
                  "mean_tokens_by_trajectory": per_traj,
                  "max_prefix_tokens_est_by_trajectory": prefix_tokens,
                  "rollouts_planned": N_PROBLEMS * WINDOW * 100 * 2,
                  "generated_tokens_projected": float(est.mean() * N_PROBLEMS * WINDOW * 200)}
    plan = {"schema_version": 1, "stage": "exploratory", "preregistered": False,
            "rule": __doc__.split("Rule:")[1].split("Also projects")[0].strip(),
            "selection_seed_material": "20260914:phase2_selection",
            "draw_order": [int(p) for p in order], "kinds_drawn": kinds,
            "skipped": skipped, "trajectories": chosen, "projection": projection}
    OUT.write_text(json.dumps(plan, indent=1))
    print(json.dumps({k: plan[k] for k in ("skipped", "projection")}, indent=1))
    for c in chosen:
        print(c["trajectory_id"], c["start_fork"], c["admissible_starts"])


if __name__ == "__main__":
    main()
