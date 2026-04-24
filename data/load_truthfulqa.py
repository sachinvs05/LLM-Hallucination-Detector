"""
data/load_truthfulqa.py

Downloads TruthfulQA from HuggingFace and builds labelled (question, answer, label)
triples suitable for training and benchmarking.

Label 0 = truthful / correct answer
Label 1 = hallucinated / incorrect answer
"""

from datasets import load_dataset
import pandas as pd
import random
from typing import Optional


def load_truthfulqa_raw(split: str = "validation"):
    """Return the raw HuggingFace TruthfulQA dataset split."""
    print(f"[TruthfulQA] Loading '{split}' split …")
    ds = load_dataset("truthful_qa", "generation", split=split)
    print(f"[TruthfulQA] {len(ds)} examples loaded.")
    return ds


def build_labelled_dataframe(
    split: str = "validation",
    max_examples: Optional[int] = None,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Build a flat DataFrame of (question, answer, label) triples.

    Each TruthfulQA example has a list of correct_answers and a list of
    incorrect_answers.  We pair every question with all of its answers and
    assign:
        label = 0   → correct / truthful answer
        label = 1   → incorrect / hallucinated answer

    Returns
    -------
    pd.DataFrame with columns:
        question, answer, label, source, best_answer
    """
    ds = load_truthfulqa_raw(split)
    rows = []

    for ex in ds:
        question    = ex["question"]
        best_answer = ex["best_answer"]
        source      = ex.get("source", "")

        for ans in ex["correct_answers"]:
            if ans.strip():
                rows.append({
                    "question":    question,
                    "answer":      ans.strip(),
                    "label":       0,
                    "source":      source,
                    "best_answer": best_answer,
                })

        for ans in ex["incorrect_answers"]:
            if ans.strip():
                rows.append({
                    "question":    question,
                    "answer":      ans.strip(),
                    "label":       1,
                    "source":      source,
                    "best_answer": best_answer,
                })

    df = pd.DataFrame(rows).sample(frac=1, random_state=seed).reset_index(drop=True)

    if max_examples:
        # Balanced subsample
        n = min(max_examples // 2, (df.label == 0).sum(), (df.label == 1).sum())
        df = pd.concat([
            df[df.label == 0].head(n),
            df[df.label == 1].head(n),
        ]).sample(frac=1, random_state=seed).reset_index(drop=True)

    pos = (df.label == 1).sum()
    neg = (df.label == 0).sum()
    print(f"[TruthfulQA] Built dataframe: {len(df)} rows  |  "
          f"{neg} truthful, {pos} hallucinated")
    return df


def get_benchmark_questions(n: int = 50, seed: int = 42):
    """
    Return a list of raw TruthfulQA examples (dicts) for benchmarking.
    Each dict keeps the full question / correct_answers / incorrect_answers.
    """
    ds = load_truthfulqa_raw("validation")
    indices = list(range(len(ds)))
    random.seed(seed)
    random.shuffle(indices)
    selected = [ds[i] for i in indices[:n]]
    print(f"[TruthfulQA] Selected {len(selected)} benchmark questions.")
    return selected


if __name__ == "__main__":
    df = build_labelled_dataframe(max_examples=200)
    print(df.head())
    print(df["label"].value_counts())
