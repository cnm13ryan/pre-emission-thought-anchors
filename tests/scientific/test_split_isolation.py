"""SPEC §10.5: problem-level split hygiene and the absence of any row-wise splitter."""

import re
from pathlib import Path

from preanchor.config import load_config

ROOT = Path(__file__).resolve().parents[2]


def test_problem_sets_are_disjoint_and_complete():
    config = load_config(ROOT / "configs/mvp.yaml")
    train, val, test = (set(config.splits.train), set(config.splits.validation),
                        set(config.splits.test))
    assert not (train & val or train & test or val & test)
    assert (len(train), len(val), len(test)) == (12, 4, 4)
    assert train | val | test == set(config.dataset.problem_ids)
    assert sorted(config.dataset.trace_kinds) == ["correct", "incorrect"]


def test_no_row_wise_random_splitter_in_source():
    forbidden = re.compile(r"train_test_split|ShuffleSplit|(?<!StratifiedGroup)(?<!Group)KFold\(")
    hits = [str(p) for p in (ROOT / "src").rglob("*.py")
            if forbidden.search(p.read_text())]
    assert hits == []


def test_both_trace_kinds_share_the_problem_split():
    from preanchor.features.dataset import split_of

    config = load_config(ROOT / "configs/mvp.yaml")
    splits = split_of(config)
    assert all(splits[p] == splits[p] for p in config.dataset.problem_ids)  # keyed by problem
    assert set(splits.values()) == {"train", "validation", "test"}
