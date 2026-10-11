"""v3 follow-up T1 (item 5): released (r0) vs our regenerations (r1, r2). EXPLORATORY, POST HOC.

Step 1  fork-by-fork comparison on the 200 Phase 2 window forks (10 trajectories):
        length, truncation, answer distribution, accuracy; the 2 incorrect-trace
        trajectories against the 8 correct-trace ones; and a decomposition of the r0-vs-ours
        accuracy gap (exclusion of truncated / no-answer, length matching, answered-only).
Step 2  generation settings. The released prompts are read back from the released
        solutions.json files (``full_cot`` minus ``rollout``) for the 200 window forks, at the
        pinned dataset revision, hash-verified against the MVP manifest, and diffed against
        our stored Phase 2 prompts. The released code settings are recorded in summary.md.

Inputs are read only. Compact per-sample features of the released window files are cached
under DATA/derived/v3_followup/t1_released_vs_ours/ (no raw JSON is kept).

Usage: python analysis/v3_followup/t1_released_vs_ours/run.py [--reps 2000] [--perms 2000]
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

STUDY = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(STUDY / "src"))
from arv2 import reliability as R  # noqa: E402
from arv2.paths import DATA, INPUTS, write_json  # noqa: E402

OUT = Path(__file__).resolve().parent
TASK = "t1_released_vs_ours"
SHARED = DATA / "derived/v3_followup"
CACHE = SHARED / TASK
PHASE2_INPUTS = Path.home() / "preanchor-root/data/processed/redesign_phase2/inputs.parquet"
RUN_DIR = DATA / "generations/phase2_mirror/20261009T0717Z-phase2-23f5e3e6"
INDEX = RUN_DIR / "completions_index.parquet"
TOKENIZER = Path(os.environ.get("ARV2_TOKENIZER", Path.home() / "preanchor-root/cache/hf/hub/"
                 "models--deepseek-ai--DeepSeek-R1-Distill-Qwen-14B/snapshots/"
                 "1df8507178afcc1bef68cd8c393f61a886323761/tokenizer.json"))
HF_REVISION = "a3a2fe1eb6d52e6ef81860baff018a8df2eb8ae2"
URL = "https://huggingface.co/datasets/uzaymacar/math-rollouts/resolve/{rev}/{path}"

# Fixed before any result was seen.
INCORRECT = ("2238:incorrect", "4019:incorrect")
LENGTH_BINS_TRAJ = 10      # primary length matching: pooled within-trajectory deciles
LENGTH_BINS_FORK = 5       # secondary: pooled within-fork quintiles
PCTS = (5, 10, 25, 50, 75, 90, 95)


def seed(name: str) -> int:
    return R.seed_from(f"20260914:arv2:v3f:{TASK}:{name}")


def load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, STUDY / f"scripts/{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


s13 = load_script("s13_phase2_diagnostics")


# ----------------------------------------------------------------------------- data

def windows() -> dict[str, list[int]]:
    cfg = json.loads((STUDY / "configs/phase2_forks.json").read_text())
    return {t["trajectory_id"]: list(t["forks"]) for t in cfg["trajectories"]}


def lead_ws(text: str) -> str:
    m = re.match(r"\s*", text)
    s = m.group(0)
    return {"": "none", " ": "space", "\n": "nl", "\n\n": "nlnl"}.get(s, "other")


def trail_ws(text: str) -> str:
    m = re.search(r"\s*$", text)
    s = m.group(0)
    return {"": "none", " ": "space", "\n": "nl", "\n\n": "nlnl"}.get(s, "other")


def text_features(text: str) -> dict:
    return {"n_chars": len(text), "n_parabreak": text.count("\n\n"),
            "n_newline": text.count("\n"), "has_think_close": "</think>" in text,
            "lead_ws": lead_ws(text)}


def fetch_released_window() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per-sample features and per-fork prompts of the released window files."""
    feat_path, prompt_path = CACHE / "released_window_features.parquet", \
        CACHE / "released_window_prompts.parquet"
    if feat_path.exists() and prompt_path.exists():
        return pd.read_parquet(feat_path), pd.read_parquet(prompt_path)
    import requests
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(str(TOKENIZER))
    win = windows()
    manifest = json.loads((INPUTS / "raw_source_manifest.json").read_text())
    recs = []
    for r in manifest["files"]:
        tid = f"{r['problem_id']}:{r['trace_kind']}"
        fork = int(Path(r["path"]).parent.name.removeprefix("chunk_"))
        if tid in win and fork in win[tid]:
            recs.append((tid, fork, r))
    assert len(recs) == 200, len(recs)
    session = requests.Session()

    def one(item):
        tid, fork, rec = item
        for attempt in range(7):
            try:
                resp = session.get(URL.format(rev=HF_REVISION, path=rec["path"]), timeout=180)
                resp.raise_for_status()
                payload = resp.content
                if hashlib.sha256(payload).hexdigest() != rec["sha256"]:
                    raise ValueError("sha256 differs from manifest")
                break
            except Exception:
                if attempt == 6:
                    raise
                time.sleep(min(60, 5 * 2 ** attempt))
        samples = json.loads(payload)
        rows, prompts = [], set()
        for i, s in enumerate(samples):
            roll = s.get("rollout") if isinstance(s.get("rollout"), str) else ""
            full = s.get("full_cot") if isinstance(s.get("full_cot"), str) else ""
            ok = bool(roll) and full.endswith(roll)
            prompt = full[: len(full) - len(roll)] if ok else None
            if prompt is not None:
                prompts.add(prompt)
            rows.append({"trajectory_id": tid, "fork_idx": fork, "sample_idx": i,
                         "has_error": "error" in s, "fullcot_endswith_rollout": ok,
                         "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest()
                         if prompt is not None else None,
                         "prefix_without_chunk": s.get("prefix_without_chunk"),
                         "chunk_removed_rel": s.get("chunk_removed"),
                         **text_features(roll), "_roll": roll})
        return rows, prompts, len(samples)

    with ThreadPoolExecutor(6) as pool:
        results = list(pool.map(one, recs))
    rows, prow = [], []
    for (tid, fork, _), (rr, prompts, n) in zip(recs, results, strict=True):
        rows.extend(rr)
        for p in sorted(prompts):
            prow.append({"trajectory_id": tid, "fork_idx": fork, "n_samples": n,
                         "n_distinct_prompts": len(prompts), "prompt": p,
                         "prompt_sha256": hashlib.sha256(p.encode()).hexdigest()})
    feat = pd.DataFrame(rows)
    feat["tokens_retok"] = [len(e.ids) for e in tok.encode_batch(
        feat["_roll"].tolist(), add_special_tokens=False)]
    pref = feat["prefix_without_chunk"].fillna("")
    feat["prefix_sha256"] = [hashlib.sha256(x.encode()).hexdigest() for x in pref]
    feat = feat.drop(columns=["_roll", "prefix_without_chunk"])
    CACHE.mkdir(parents=True, exist_ok=True)
    feat.to_parquet(feat_path, index=False)
    pd.DataFrame(prow).to_parquet(prompt_path, index=False)
    return feat, pd.DataFrame(prow)


def our_text_features() -> pd.DataFrame:
    path = CACHE / "ours_window_features.parquet"
    if path.exists():
        return pd.read_parquet(path)
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(str(TOKENIZER))
    idx = pd.read_parquet(INDEX, columns=["replicate", "trajectory_id", "fork_idx",
                                          "sample_idx", "raw_answer", "prompt_token_count"])
    texts = idx["raw_answer"].fillna("").tolist()
    f = pd.DataFrame([text_features(t) for t in texts])
    f["tokens_retok"] = [len(e.ids) for e in tok.encode_batch(texts, add_special_tokens=False)]
    out = pd.concat([idx.drop(columns=["raw_answer"]).reset_index(drop=True), f], axis=1)
    CACHE.mkdir(parents=True, exist_ok=True)
    out.to_parquet(path, index=False)
    return out


def sample_table() -> pd.DataFrame:
    """One row per sample on the window forks: replicate 0 (released), 1, 2."""
    win = windows()
    keep = {(t, f) for t, fs in win.items() for f in fs}
    rel = pd.read_parquet(SHARED / "released_canon.parquet",
                          columns=["trajectory_id", "problem_id", "fork_idx", "sample_idx",
                                   "v2_status", "rollout_tokens", "cat_raw", "cat_canon",
                                   "correct_raw", "correct_canon", "chunk_resampled"])
    rel = rel[[(t, f) in keep for t, f in zip(rel.trajectory_id, rel.fork_idx)]].copy()
    rel["replicate"] = 0
    rel["truncated"] = rel.v2_status.eq("length_truncated")
    rel["tokens"] = rel.rollout_tokens
    feat, _ = fetch_released_window()
    rel = rel.merge(feat[["trajectory_id", "fork_idx", "sample_idx", "tokens_retok",
                          "n_chars", "n_parabreak", "has_think_close", "lead_ws"]],
                    on=["trajectory_id", "fork_idx", "sample_idx"], how="left",
                    validate="one_to_one")
    assert rel.tokens_retok.notna().all()
    ours = pd.read_parquet(SHARED / "phase2_canon.parquet")
    ours["problem_id"] = ours.trajectory_id.str.split(":").str[0].astype(int)
    ours["truncated"] = ours.finish_reason.eq("length")
    ours["tokens"] = ours.generated_token_count
    of = our_text_features()
    ours = ours.merge(of[["replicate", "trajectory_id", "fork_idx", "sample_idx", "tokens_retok",
                          "n_chars", "n_parabreak", "has_think_close", "lead_ws"]],
                      on=["replicate", "trajectory_id", "fork_idx", "sample_idx"],
                      how="left", validate="one_to_one")
    cols = ["replicate", "trajectory_id", "problem_id", "fork_idx", "sample_idx", "v2_status",
            "truncated", "tokens", "tokens_retok", "n_chars", "n_parabreak", "has_think_close",
            "lead_ws", "cat_raw", "cat_canon", "correct_raw", "correct_canon"]
    s = pd.concat([rel[cols], ours[cols]], ignore_index=True)
    s["no_answer"] = s.v2_status.eq("no_answer")
    s["answered"] = s.v2_status.eq("answered")
    s["valid"] = ~(s.truncated | s.no_answer)
    s["group"] = np.where(s.trajectory_id.isin(INCORRECT), "incorrect", "correct")
    assert s.groupby("replicate").size().to_dict() == {0: 20000, 1: 20000, 2: 20000}
    return s


# ----------------------------------------------------------------------------- step 1

def describe_lengths(x: np.ndarray) -> dict:
    x = np.asarray(x, float)
    return {"n": int(len(x)), "mean": float(x.mean()),
            **{f"p{p}": float(np.percentile(x, p)) for p in PCTS}}


def wmean(v, w):
    w = np.asarray(w, float)
    return float(np.sum(np.asarray(v, float) * w) / w.sum()) if w.sum() > 0 else float("nan")


def length_weights(d: pd.DataFrame, by: list[str], nbins: int) -> tuple[np.ndarray, float]:
    """Weights on our samples (r1, r2) so that, within each ``by`` stratum, their pooled
    length-bin mass matches r0's. Bins are pooled (r0 u r1 u r2) quantiles of tokens_retok.
    r0 rows get weight 1 unless their bin has no samples of ours (then 0). Returns the weights
    and the fraction of r0 rows dropped for lack of support."""
    w = np.zeros(len(d))
    dropped = 0
    for _, g in d.groupby(by, sort=False):
        t = g.tokens_retok.to_numpy(float)
        edges = np.unique(np.quantile(t, np.linspace(0, 1, nbins + 1)))
        b = np.clip(np.searchsorted(edges, t, side="right") - 1, 0, len(edges) - 2)
        is0 = g.replicate.to_numpy() == 0
        pos = d.index.get_indexer(g.index)
        n0 = np.bincount(b[is0], minlength=len(edges))
        n1 = np.bincount(b[~is0], minlength=len(edges))
        for k in range(len(edges)):
            m0, m1 = is0 & (b == k), (~is0) & (b == k)
            if n0[k] and n1[k]:
                w[pos[m0]] = 1.0
                w[pos[m1]] = n0[k] / n1[k]
            elif n0[k]:
                dropped += n0[k]
    return w, dropped / max(1, int((d.replicate == 0).sum()))


def traj_metrics(s: pd.DataFrame) -> pd.DataFrame:
    """Per trajectory: accuracies and gap variants. Gap = r0 - (r1 u r2); ref = r1 - r2."""
    s = s.copy()
    s = s.reset_index(drop=True)
    w_traj, drop_traj = length_weights(s, ["trajectory_id"], LENGTH_BINS_TRAJ)
    s["w_len_traj"] = w_traj
    w_fork, drop_fork = length_weights(s, ["trajectory_id", "fork_idx"], LENGTH_BINS_FORK)
    s["w_len_fork"] = w_fork
    sv = s[s.valid].reset_index(drop=True)
    wv, drop_v = length_weights(sv, ["trajectory_id"], LENGTH_BINS_TRAJ)
    sv["w_len_traj_valid"] = wv
    rows = []
    for tid, g in s.groupby("trajectory_id"):
        gv = sv[sv.trajectory_id == tid]
        r0, new = g[g.replicate == 0], g[g.replicate > 0]
        r1, r2 = g[g.replicate == 1], g[g.replicate == 2]
        row = {"trajectory_id": tid, "problem_id": int(g.problem_id.iloc[0]),
               "group": g.group.iloc[0]}
        for name, d in (("r0", r0), ("r1", r1), ("r2", r2), ("new", new)):
            row[f"acc_{name}"] = float(d.correct_canon.mean())
            row[f"acc_raw_{name}"] = float(d.correct_raw.mean())
            row[f"trunc_{name}"] = float(d.truncated.mean())
            row[f"noans_{name}"] = float(d.no_answer.mean())
            row[f"answered_{name}"] = float(d.answered.mean())
            row[f"acc_valid_{name}"] = float(d[d.valid].correct_canon.mean())
            row[f"acc_answered_{name}"] = float(d[d.answered].correct_canon.mean())
            row[f"tok_median_{name}"] = float(d.tokens_retok.median())
            row[f"tok_mean_{name}"] = float(d.tokens_retok.mean())
            row[f"parabreak_per_kchar_{name}"] = float(1000 * d.n_parabreak.sum() / d.n_chars.sum())
        for name in ("r1", "r2", "new"):
            d = g[g.replicate > 0] if name == "new" else g[g.replicate == int(name[1])]
            dv = gv[gv.replicate > 0] if name == "new" else gv[gv.replicate == int(name[1])]
            row[f"acc_lenmatch_traj_{name}"] = wmean(d.correct_canon, d.w_len_traj)
            row[f"acc_lenmatch_fork_{name}"] = wmean(d.correct_canon, d.w_len_fork)
            row[f"acc_valid_lenmatch_{name}"] = wmean(dv.correct_canon, dv.w_len_traj_valid)
        z0 = r0.w_len_traj > 0
        row["acc_r0_lensupport_traj"] = float(r0[z0].correct_canon.mean())
        z0f = r0.w_len_fork > 0
        row["acc_r0_lensupport_fork"] = float(r0[z0f].correct_canon.mean())
        r0v = gv[gv.replicate == 0]
        row["acc_r0_valid_lensupport"] = float(r0v[r0v.w_len_traj_valid > 0].correct_canon.mean())
        # gap variants
        row["gap_D0_all"] = row["acc_r0"] - row["acc_new"]
        row["gap_D0_all_raw"] = row["acc_raw_r0"] - row["acc_raw_new"]
        row["gap_D1_excl_trunc_noans"] = row["acc_valid_r0"] - row["acc_valid_new"]
        row["gap_D2_lenmatch_traj"] = row["acc_r0_lensupport_traj"] - row["acc_lenmatch_traj_new"]
        row["gap_D2b_lenmatch_fork"] = row["acc_r0_lensupport_fork"] - row["acc_lenmatch_fork_new"]
        row["gap_D12_excl_and_lenmatch"] = row["acc_r0_valid_lensupport"] - \
            row["acc_valid_lenmatch_new"]
        row["gap_D3_answered_only"] = row["acc_answered_r0"] - row["acc_answered_new"]
        row["ref_r1_minus_r2"] = row["acc_r1"] - row["acc_r2"]
        row["ref_r1_minus_r2_answered"] = row["acc_answered_r1"] - row["acc_answered_r2"]
        row["dtrunc_r0_minus_new"] = row["trunc_r0"] - row["trunc_new"]
        row["dnoans_r0_minus_new"] = row["noans_r0"] - row["noans_new"]
        row["dtok_median_r0_minus_new"] = row["tok_median_r0"] - row["tok_median_new"]
        rows.append(row)
    out = pd.DataFrame(rows)
    out.attrs["len_support_dropped"] = {"traj": drop_traj, "fork": drop_fork, "valid": drop_v}
    return out


def fork_tv(s: pd.DataFrame, col: str) -> pd.DataFrame:
    rows = []
    for (tid, fork), g in s.groupby(["trajectory_id", "fork_idx"]):
        codes, _ = pd.factorize(g[col])
        k = int(codes.max()) + 1
        rep = g.replicate.to_numpy()
        a0, a1, a2 = codes[rep == 0], codes[rep == 1], codes[rep == 2]
        an = codes[rep > 0]
        t01, t02, t12 = s13.tv(a0, a1, k), s13.tv(a0, a2, k), s13.tv(a1, a2, k)
        rows.append({"trajectory_id": tid, "problem_id": int(g.problem_id.iloc[0]),
                     "group": g.group.iloc[0], "fork_idx": fork,
                     "tv_r0_r1": t01, "tv_r0_r2": t02, "tv_r1_r2": t12,
                     "tv_r0_new": s13.tv(a0, an, k),
                     "floor_r0_new": s13.tv_null_analytic(a0, an, k),
                     "floor_r1_r2": s13.tv_null_analytic(a1, a2, k),
                     "excess": 0.5 * (t01 + t02) - t12,
                     "acc_r0": float(g.correct_canon[rep == 0].mean()),
                     "acc_new": float(g.correct_canon[rep > 0].mean()),
                     "acc_r1": float(g.correct_canon[rep == 1].mean()),
                     "acc_r2": float(g.correct_canon[rep == 2].mean())})
    return pd.DataFrame(rows)


def perm_test(s: pd.DataFrame, col: str, perms: int, name: str) -> pd.DataFrame:
    """Per trajectory: shuffle the replicate labels (sizes 100/100/100) within each fork.
    Statistics: accuracy gap r0 - new, and mean excess TV 0.5(TV01+TV02) - TV12."""
    rng = np.random.default_rng(seed(f"perm:{name}"))
    rows = []
    for tid, g in s.groupby("trajectory_id"):
        forks = []
        for _, h in g.groupby("fork_idx"):
            codes, _ = pd.factorize(h[col])
            forks.append((codes, h.correct_canon.to_numpy(float), h.replicate.to_numpy()))

        def stat(labels_list):
            gaps, exc = [], []
            for (codes, corr, _), rep in zip(forks, labels_list, strict=True):
                k = int(codes.max()) + 1
                a0, a1, a2 = codes[rep == 0], codes[rep == 1], codes[rep == 2]
                gaps.append(corr[rep == 0].mean() - corr[rep > 0].mean())
                exc.append(0.5 * (s13.tv(a0, a1, k) + s13.tv(a0, a2, k)) - s13.tv(a1, a2, k))
            return float(np.mean(gaps)), float(np.mean(exc))

        obs = stat([f[2] for f in forks])
        null = np.array([stat([rng.permutation(f[2]) for f in forks]) for _ in range(perms)])
        rows.append({"trajectory_id": tid, "gap_obs": obs[0], "excess_obs": obs[1],
                     "p_gap_two_sided": float((1 + np.sum(np.abs(null[:, 0]) >= abs(obs[0])
                                                          - 1e-12)) / (perms + 1)),
                     "p_excess_upper": float((1 + np.sum(null[:, 1] >= obs[1] - 1e-12))
                                             / (perms + 1)),
                     "null_excess_mean": float(null[:, 1].mean())})
    return pd.DataFrame(rows)


def boot_mean(df: pd.DataFrame, col: str, reps: int, name: str) -> dict:
    """Mean over trajectories (one per problem), cluster bootstrap over problems."""
    v = df[col].to_numpy(float)
    groups = df.problem_id.to_numpy()
    est = float(np.nanmean(v))
    if len(np.unique(groups)) < 2:
        return {"est": est, "lo": float("nan"), "hi": float("nan"), "n_problems": 1}
    draws = R.cluster_bootstrap(groups, lambda rows: float(np.nanmean(v[rows])),
                                reps=reps, seed=seed(f"boot:{name}:{col}"))
    lo, hi, nan = R.interval(draws)
    return {"est": est, "lo": lo, "hi": hi, "n_problems": int(len(np.unique(groups))),
            "nan_draws": nan}


def boot_diff(df: pd.DataFrame, col: str, reps: int, name: str) -> dict:
    """Mean(incorrect) - mean(correct), resampling problems within each group."""
    rng = np.random.default_rng(seed(f"bootdiff:{name}:{col}"))
    a = df[df.group == "incorrect"][col].to_numpy(float)
    b = df[df.group == "correct"][col].to_numpy(float)
    est = float(a.mean() - b.mean())
    draws = np.array([rng.choice(a, len(a)).mean() - rng.choice(b, len(b)).mean()
                      for _ in range(reps)])
    lo, hi, _ = R.interval(draws)
    return {"est": est, "lo": lo, "hi": hi}


# ----------------------------------------------------------------------------- step 2

def think_part(prompt: str) -> str:
    return prompt.split("<think>\n", 1)[1] if "<think>\n" in prompt else prompt


def norm_ws(x: str) -> str:
    return re.sub(r"\s+", " ", x).strip()


def prompt_diff() -> tuple[pd.DataFrame, dict]:
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(str(TOKENIZER))
    ours = pd.read_parquet(PHASE2_INPUTS)
    feat, rel = fetch_released_window()
    rows = []
    for _, o in ours.iterrows():
        rr = rel[(rel.trajectory_id == o.trajectory_id) & (rel.fork_idx == o.fork_idx)]
        assert len(rr) == 1, (o.trajectory_id, o.fork_idx, len(rr))
        rp, op = rr.prompt.iloc[0], o.prompt
        rt, ot = think_part(rp), think_part(op)
        rhead, ohead = rp[: len(rp) - len(rt)], op[: len(op) - len(ot)]
        ntok_r = len(tok.encode(rp, add_special_tokens=False).ids)
        ntok_o = len(tok.encode(op, add_special_tokens=False).ids)
        nr, no = norm_ws(rt), norm_ws(ot)
        first = next((i for i, (a, b) in enumerate(zip(nr, no)) if a != b), min(len(nr), len(no)))
        rows.append({
            "trajectory_id": o.trajectory_id, "fork_idx": int(o.fork_idx),
            "exact_equal": rp == op,
            "header_equal": rhead == ohead,
            "header_equal_ws_norm": norm_ws(rhead) == norm_ws(ohead),
            "think_equal_ws_norm": nr == no,
            "think_released_is_prefix_of_ours_ws_norm": no.startswith(nr),
            "chars_missing_in_released_ws_norm": len(no) - len(nr),
            "first_diff_context_ours": no[max(0, first - 60): first + 60] if nr != no else "",
            "first_diff_context_released": nr[max(0, first - 60): first + 60] if nr != no else "",
            "ours_trailing_ws": trail_ws(op), "released_trailing_ws": trail_ws(rp),
            "parabreaks_ours_think": ot.count("\n\n"), "parabreaks_released_think": rt.count("\n\n"),
            "newlines_ours_think": ot.count("\n"), "newlines_released_think": rt.count("\n"),
            "prompt_tokens_ours": ntok_o, "prompt_tokens_released": ntok_r,
            "header_ours_tail": repr(ohead[-40:]), "header_released_tail": repr(rhead[-40:]),
        })
    d = pd.DataFrame(rows)
    feat_fork = feat.groupby(["trajectory_id", "fork_idx"]).agg(
        n=("sample_idx", "size"), any_error=("has_error", "any"),
        all_fullcot_ok=("fullcot_endswith_rollout", "all"),
        n_prefix_variants=("prefix_sha256", "nunique"))
    summ = {
        "n_forks": int(len(d)),
        "released_files_all_100_samples": bool((feat_fork.n == 100).all()),
        "released_any_error_record": bool(feat_fork.any_error.any()),
        "released_full_cot_endswith_rollout_all": bool(feat_fork.all_fullcot_ok.all()),
        "released_one_prompt_per_fork": bool((rel.n_distinct_prompts == 1).all()),
        "released_one_prefix_variant_per_fork": bool((feat_fork.n_prefix_variants == 1).all()),
        "exact_equal": int(d.exact_equal.sum()),
        "header_equal": int(d.header_equal.sum()),
        "header_equal_ws_norm": int(d.header_equal_ws_norm.sum()),
        "think_equal_ws_norm": int(d.think_equal_ws_norm.sum()),
        "forks_released_missing_text": int((d.chars_missing_in_released_ws_norm > 0).sum()),
        "ours_trailing_ws": d.ours_trailing_ws.value_counts().to_dict(),
        "released_trailing_ws": d.released_trailing_ws.value_counts().to_dict(),
        "parabreaks_think_mean_ours": float(d.parabreaks_ours_think.mean()),
        "parabreaks_think_mean_released": float(d.parabreaks_released_think.mean()),
        "newlines_think_mean_ours": float(d.newlines_ours_think.mean()),
        "newlines_think_mean_released": float(d.newlines_released_think.mean()),
        "prompt_tokens_mean_ours": float(d.prompt_tokens_ours.mean()),
        "prompt_tokens_mean_released": float(d.prompt_tokens_released.mean()),
        "prompt_token_diff_mean_ours_minus_released":
            float((d.prompt_tokens_ours - d.prompt_tokens_released).mean()),
        "header_tail_examples": d.groupby("trajectory_id")[
            ["header_ours_tail", "header_released_tail"]].first().to_dict("index"),
        "by_trajectory": d.groupby("trajectory_id").agg(
            think_equal_ws_norm=("think_equal_ws_norm", "sum"),
            header_equal=("header_equal", "sum"),
            parabreaks_ours=("parabreaks_ours_think", "mean"),
            parabreaks_released=("parabreaks_released_think", "mean"),
            missing_chars=("chars_missing_in_released_ws_norm", "max")).to_dict("index"),
    }
    return d, summ


# ----------------------------------------------------------------------------- main

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=2000)
    ap.add_argument("--perms", type=int, default=2000)
    args = ap.parse_args()
    t0 = time.time()
    s = sample_table()
    res: dict = {"status": "EXPLORATORY, POST HOC", "task": TASK,
                 "incorrect_trajectories": list(INCORRECT),
                 "settings": {"reps": args.reps, "perms": args.perms,
                              "length_bins_traj": LENGTH_BINS_TRAJ,
                              "length_bins_fork": LENGTH_BINS_FORK,
                              "accuracy": "correct_canon over all samples unless stated",
                              "tokens": "tokens_retok = pinned tokenizer on the generated text, "
                                        "add_special_tokens=False, both sides"}}

    # --- Step 2 prompt diff (first: it informs the reading of Step 1)
    pdiff, psumm = prompt_diff()
    pdiff.to_csv(OUT / "prompt_diff_by_fork.csv", index=False)
    res["step2_prompt_diff"] = psumm

    # --- caps and lengths
    caps = {}
    for rep, d in s.groupby("replicate"):
        caps[f"r{rep}"] = {"max_tokens_native": int(d.tokens.max()),
                           "max_tokens_retok": int(d.tokens_retok.max()),
                           "min_tokens_truncated_native": int(d[d.truncated].tokens.min())
                           if d.truncated.any() else None,
                           "n_truncated": int(d.truncated.sum())}
    res["caps"] = caps
    allr = pd.read_parquet(SHARED / "released_canon.parquet", columns=["v2_status",
                                                                        "rollout_tokens"])
    res["caps"]["r0_all_647400"] = {
        "max_tokens_retok": int(allr.rollout_tokens.max()),
        "truncated_tokens_p1_p50_p99": [float(x) for x in np.percentile(
            allr[allr.v2_status == "length_truncated"].rollout_tokens, [1, 50, 99])],
        "n_truncated": int((allr.v2_status == "length_truncated").sum())}
    lengths = {}
    for grp in ("incorrect", "correct", "all"):
        dg = s if grp == "all" else s[s.group == grp]
        lengths[grp] = {f"r{r}": describe_lengths(d.tokens_retok) for r, d in dg.groupby("replicate")}
        lengths[grp]["new"] = describe_lengths(dg[dg.replicate > 0].tokens_retok)
        lengths[grp]["new_native_count"] = describe_lengths(dg[dg.replicate > 0].tokens)
        lengths[grp]["r0_valid"] = describe_lengths(dg[(dg.replicate == 0) & dg.valid].tokens_retok)
        lengths[grp]["new_valid"] = describe_lengths(dg[(dg.replicate > 0) & dg.valid].tokens_retok)
    res["lengths"] = lengths
    # retok vs native agreement on ours
    o = s[s.replicate > 0]
    res["token_count_check_ours"] = {
        "median_abs_diff_retok_vs_server": float((o.tokens_retok - o.tokens).abs().median()),
        "mean_diff_retok_minus_server": float((o.tokens_retok - o.tokens).mean())}
    # leading whitespace of the generated continuation
    res["continuation_lead_ws"] = {f"r{r}": d.lead_ws.value_counts(normalize=True).round(4).to_dict()
                                   for r, d in s.groupby("replicate")}
    res["parabreak_per_kchar"] = {
        grp: {f"r{r}": float(1000 * d.n_parabreak.sum() / d.n_chars.sum())
              for r, d in (s if grp == "all" else s[s.group == grp]).groupby("replicate")}
        for grp in ("incorrect", "correct", "all")}

    # --- per trajectory metrics and gap decomposition
    tm = traj_metrics(s)
    tm.to_csv(OUT / "trajectory_metrics.csv", index=False)
    res["length_support_dropped_frac_r0"] = tm.attrs["len_support_dropped"]
    gap_cols = ["gap_D0_all", "gap_D0_all_raw", "gap_D1_excl_trunc_noans",
                "gap_D2_lenmatch_traj", "gap_D2b_lenmatch_fork", "gap_D12_excl_and_lenmatch",
                "gap_D3_answered_only", "ref_r1_minus_r2", "ref_r1_minus_r2_answered",
                "dtrunc_r0_minus_new", "dnoans_r0_minus_new", "dtok_median_r0_minus_new"]
    show = ["acc_r0", "acc_r1", "acc_r2", "acc_new", "trunc_r0", "trunc_new", "noans_r0",
            "noans_new", "tok_median_r0", "tok_median_new", "acc_answered_r0",
            "acc_answered_new", "parabreak_per_kchar_r0", "parabreak_per_kchar_new"] + gap_cols
    res["per_trajectory"] = tm.set_index("trajectory_id")[["group"] + show].round(5) \
        .to_dict("index")
    groups = {}
    for grp in ("incorrect", "correct", "all"):
        dg = tm if grp == "all" else tm[tm.group == grp]
        groups[grp] = {c: boot_mean(dg, c, args.reps, grp) for c in gap_cols}
        g0 = groups[grp]["gap_D0_all"]["est"]
        groups[grp]["fraction_of_gap_removed"] = {
            c: (1 - groups[grp][c]["est"] / g0) if abs(g0) > 1e-12 else float("nan")
            for c in ("gap_D1_excl_trunc_noans", "gap_D2_lenmatch_traj",
                      "gap_D2b_lenmatch_fork", "gap_D12_excl_and_lenmatch",
                      "gap_D3_answered_only")}
    groups["incorrect_minus_correct"] = {c: boot_diff(tm, c, args.reps, "diff")
                                         for c in ("gap_D0_all", "gap_D3_answered_only",
                                                   "ref_r1_minus_r2")}
    res["gap_by_group"] = groups

    # --- fork-level TV
    tvres = {}
    for col in ("cat_canon", "cat_raw"):
        ft = fork_tv(s, col)
        ft = ft.round({c: 10 for c in ft.columns if c.startswith(("tv_", "floor", "excess"))})
        if col == "cat_canon":
            ft.to_csv(OUT / "fork_tv_canon.csv", index=False)
        per_t = ft.groupby(["trajectory_id", "problem_id", "group"], as_index=False)[
            ["tv_r0_r1", "tv_r0_r2", "tv_r1_r2", "tv_r0_new", "floor_r0_new", "floor_r1_r2",
             "excess"]].mean()
        out = {"per_trajectory": per_t.set_index("trajectory_id").drop(
            columns=["problem_id"]).round(5).to_dict("index")}
        for grp in ("incorrect", "correct", "all"):
            dg = per_t if grp == "all" else per_t[per_t.group == grp]
            out[grp] = {c: boot_mean(dg, c, args.reps, f"tv:{col}:{grp}")
                        for c in ("tv_r0_r1", "tv_r0_r2", "tv_r1_r2", "excess",
                                  "tv_r0_new", "floor_r0_new")}
        out["incorrect_minus_correct_excess"] = boot_diff(per_t, "excess", args.reps,
                                                          f"tv:{col}")
        # fork-level: is the accuracy gap uniform over the window?
        ft["gap"] = ft.acc_r0 - ft.acc_new
        out["fork_gap_spread"] = {
            tid: {"mean": float(g.gap.mean()), "sd": float(g.gap.std(ddof=1)),
                  "min": float(g.gap.min()), "max": float(g.gap.max()),
                  "frac_forks_r0_below": float((g.gap < 0).mean()),
                  "sd_ref_r1_minus_r2": float((g.acc_r1 - g.acc_r2).std(ddof=1))}
            for tid, g in ft.groupby("trajectory_id")}
        tvres[col] = out
    res["fork_tv"] = tvres

    # --- permutation tests (exchangeability of r0 with r1, r2 within fork)
    pt = perm_test(s, "cat_canon", args.perms, "canon")
    res["perm_exchangeability_canon"] = pt.set_index("trajectory_id").round(5).to_dict("index")

    # --- prompt feature vs per-fork gap (descriptive)
    ft = pd.read_csv(OUT / "fork_tv_canon.csv")
    ft["gap"] = ft.acc_r0 - ft.acc_new
    m = ft.merge(pdiff[["trajectory_id", "fork_idx", "ours_trailing_ws"]],
                 on=["trajectory_id", "fork_idx"])
    res["gap_by_our_trailing_ws"] = {
        grp: m[m.group == grp].groupby("ours_trailing_ws").agg(
            n=("gap", "size"), mean_gap=("gap", "mean"),
            mean_excess=("excess", "mean")).round(5).to_dict("index")
        for grp in ("incorrect", "correct")}
    lw = s.merge(pdiff[["trajectory_id", "fork_idx", "ours_trailing_ws"]],
                 on=["trajectory_id", "fork_idx"])
    res["r0_lead_ws_by_our_forced_sep"] = lw[lw.replicate == 0].groupby("ours_trailing_ws") \
        .lead_ws.value_counts(normalize=True).round(4).unstack(fill_value=0).to_dict("index")
    res["runtime_seconds"] = round(time.time() - t0, 1)
    write_json(OUT / "results.json", res)
    print(json.dumps({k: res[k] for k in ("caps", "gap_by_group")}, indent=1, default=str)[:6000])


if __name__ == "__main__":
    main()
