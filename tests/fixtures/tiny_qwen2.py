"""Network-independent fixtures: a character-level fast tokenizer and a tiny random Qwen2."""

from __future__ import annotations

import string

VOCAB_CHARS = string.printable  # 100 characters including whitespace and newline


def tiny_tokenizer():
    from tokenizers import Regex, Tokenizer, decoders, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast

    vocab = {c: i for i, c in enumerate(VOCAB_CHARS)}
    vocab["<unk>"] = len(vocab)
    tok = Tokenizer(models.WordLevel(vocab, unk_token="<unk>"))
    tok.pre_tokenizer = pre_tokenizers.Split(Regex("[\\s\\S]"), behavior="isolated")
    tok.decoder = decoders.Fuse()
    return PreTrainedTokenizerFast(tokenizer_object=tok, unk_token="<unk>")


def tiny_model(seed: int = 0, layers: int = 4, hidden: int = 32):
    import torch
    from transformers import Qwen2Config, Qwen2ForCausalLM

    torch.manual_seed(seed)
    config = Qwen2Config(vocab_size=128, hidden_size=hidden, intermediate_size=64,
                         num_hidden_layers=layers, num_attention_heads=4, num_key_value_heads=2,
                         max_position_embeddings=512, tie_word_embeddings=False,
                         attn_implementation="sdpa")
    model = Qwen2ForCausalLM(config)
    model.eval()
    return model


def tiny_loader(config):
    return tiny_tokenizer(), tiny_model(), 0.0


def synthetic_corpus(config) -> tuple[list[dict], list[dict], list[dict]]:
    """40 trajectories (Stage 3 grain) and matching Stage 4 rows with the real column set."""
    import hashlib

    trajectories, sentences, labels = [], [], []
    for problem in config.dataset.problem_ids:
        for kind in ("correct", "incorrect"):
            tid = f"{problem}:{kind}"
            prompt = f"Problem {problem} Solution: \n<think>\n"
            body = [f"Okay, so {problem} is {kind}.", "Let me see.", "So x = 2.",
                    "Wait, check that.", "Hmm, maybe 3.", "Yes, 3 works.", "Therefore 3.",
                    "But also 4?", "No, 3.", "Let me verify again.", "Fine.", "Done now.",
                    "Actually not.", "Recheck.", "Alright.", "So the answer is 3."]
            if tid == "4682:correct":   # every frozen Stage 5 fork (up to 47) needs a boundary
                body += [f"Step {i} holds." for i in range(16, 47)]
            body.append("Final: \\boxed{3}.")
            trace, spans = prompt, []
            for i, s in enumerate(body):
                if i:
                    trace += "\n" if i % 2 else " "
                spans.append((len(trace), len(trace) + len(s)))
                trace += s
            trajectories.append({"schema_version": 1, "trajectory_id": tid,
                                 "problem_id": problem, "trace_kind": kind,
                                 "problem_text": "p", "ground_truth_answer": "3",
                                 "prompt": prompt, "solution": "3", "full_cot": trace,
                                 "full_cot_sha256": hashlib.sha256(trace.encode()).hexdigest(),
                                 "normalized_final_answer": "3", "is_correct": kind == "correct",
                                 "source_path": "synthetic", "source_revision": "synthetic",
                                 "n_chars": len(trace)})
            for i, (a, b) in enumerate(spans):
                text = trace[a:b]
                row = {"schema_version": 1, "sentence_id": f"{tid}:{i}", "trajectory_id": tid,
                       "problem_id": problem, "trace_kind": kind, "sentence_idx": i,
                       "text": text, "text_sha256": hashlib.sha256(text.encode()).hexdigest(),
                       "char_start": a, "char_end": b, "alignment_method": "exact",
                       "alignment_ok": True, "eligible": i < 14, "exclusion_reason": None,
                       "importance_released": 0.1, "importance_legacy": None,
                       "importance_canonical": None, "importance_primary": 0.1,
                       "label_source": "released"}
                sentences.append(row)
                labels.append({**row, "exclusion_reason": None if i < 14 else "final_answer",
                               "importance_legacy": 0.1, "importance_canonical": 0.1,
                               "label_source": "canonical_fallback", "importance_log1p": 0.09,
                               "eligible_k4": i < 12, "anchor": i == 3})
    return trajectories, sentences, labels


def write_synthetic_root(root, config):
    import pyarrow as pa

    from preanchor.io import write_parquet

    trajectories, sentences, labels = synthetic_corpus(config)
    processed = root / "data/processed"
    write_parquet(processed / "stage3/trajectories.parquet", pa.Table.from_pylist(trajectories))
    write_parquet(processed / "stage3/sentences.parquet", pa.Table.from_pylist(sentences))
    write_parquet(processed / "stage4/labels.parquet", pa.Table.from_pylist(labels))
    return processed
