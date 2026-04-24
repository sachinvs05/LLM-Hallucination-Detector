"""
benchmark.py — FREE VERSION

Benchmarks all free detectors on TruthfulQA.
Uses Ollama for LLM calls, HuggingFace for NLI/embeddings.
No API keys required.

Run:
    python benchmark.py --n 50
    python benchmark.py --n 50 --fast          # only logprob + nli
    python benchmark.py --n 50 --heavy         # include semantic + selfcheck
"""

import argparse, os, json, time
from datetime import datetime

import numpy as np
import pandas as pd
from tqdm import tqdm

import ollama
from sklearn.metrics import (
    classification_report, f1_score,
    precision_score, recall_score, roc_auc_score,
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, os.path.dirname(__file__))
import config
from data.load_truthfulqa import get_benchmark_questions
from detectors.logprob_detector import LogProbDetector
from detectors.nli_consistency  import NLIConsistencyDetector

API_COSTS = {
    "logprob":    1,
    "nli":        2,
    "semantic":   config.NUM_SAMPLES,
    "selfcheck":  config.NUM_SAMPLES,
    "classifier": 0,
}


import time

def generate_answer(question: str, retries: int = 3) -> str:
    for attempt in range(retries):
        try:
            resp = ollama.chat(
                model=config.OLLAMA_MODEL,
                messages=[{"role": "user", "content": question}],
                options={"temperature": 0.3},
            )
            return resp["message"]["content"].strip()
        except Exception as e:
            print(f"  [ollama error attempt {attempt+1}/{retries}] {e}")
            if attempt < retries - 1:
                time.sleep(3)   # give the runner time to recover
    return ""   # fallback: empty string gets scored as non-hallucination


def is_hallucinated(generated: str, correct_answers: list, incorrect_answers: list) -> int:
    gen = generated.lower()
    def overlap(text, candidates):
        scores = [len(set(c.lower().split()) & set(text.split())) / max(len(c.split()), 1)
                  for c in candidates if c.strip()]
        return max(scores) if scores else 0.0
    return int(overlap(gen, incorrect_answers) >= overlap(gen, correct_answers) + 0.05)


def run_benchmark(n=50, include_heavy=False):
    print(f"\n[Benchmark] TruthfulQA  n={n}  model={config.OLLAMA_MODEL}")
    examples    = get_benchmark_questions(n=n)
    logprob_det = LogProbDetector()
    nli_det     = NLIConsistencyDetector()

    if include_heavy:
        from detectors.semantic_entropy import SemanticEntropyDetector
        from detectors.selfcheck_gpt    import SelfCheckGPTDetector
        sem_det = SemanticEntropyDetector()
        sc_det  = SelfCheckGPTDetector()

    records = []
    for ex in tqdm(examples, desc="Evaluating"):
        q         = ex["question"]
        generated = generate_answer(q)
        gt        = is_hallucinated(generated, ex["correct_answers"], ex["incorrect_answers"])
        row       = {"question": q, "generated": generated, "gt_label": gt}

        for name, det in [("logprob", logprob_det), ("nli", nli_det)]:
            try:
                r = det.detect(q, generated)
                row[f"{name}_score"] = r.score
                row[f"{name}_pred"]  = int(r.score > config.HALLUCINATION_THRESH)
            except Exception as e:
                row[f"{name}_score"] = 0.5
                row[f"{name}_pred"]  = 0
                print(f"  [{name} error] {e}")

        if include_heavy:
            for name, det in [("semantic", sem_det), ("selfcheck", sc_det)]:
                try:
                    r = det.detect(q) if name == "semantic" else det.detect(q, generated)
                    row[f"{name}_score"] = r.score
                    row[f"{name}_pred"]  = int(r.score > config.HALLUCINATION_THRESH)
                except Exception as e:
                    row[f"{name}_score"] = 0.5
                    row[f"{name}_pred"]  = 0

        records.append(row)
        time.sleep(0.1)

    return pd.DataFrame(records)


def report(df, out=config.RESULTS_DIR):
    os.makedirs(out, exist_ok=True)
    gt           = df["gt_label"].tolist()
    method_cols  = [c for c in df.columns if c.endswith("_pred")]
    method_names = [c.replace("_pred", "") for c in method_cols]
    summary_rows = []

    print("\n" + "=" * 65 + "\n  RESULTS\n" + "=" * 65)
    for name, col in zip(method_names, method_cols):
        preds = df[col].tolist()
        print(f"\n── {name.upper()} ──")
        print(classification_report(gt, preds, target_names=["Truthful","Hallucinated"], digits=3))
        summary_rows.append({
            "Method":    name,
            "F1":        round(f1_score(gt, preds, zero_division=0), 3),
            "Precision": round(precision_score(gt, preds, zero_division=0), 3),
            "Recall":    round(recall_score(gt, preds), 3),
            "API_calls": API_COSTS.get(name, "?"),
        })

    summary = pd.DataFrame(summary_rows).sort_values("F1", ascending=False)
    print("\n" + summary.to_string(index=False))

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    df.to_csv(os.path.join(out, f"benchmark_{ts}.csv"), index=False)

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    x = np.arange(len(summary))
    w = 0.25
    ax = axes[0]
    ax.bar(x - w, summary["F1"],        width=w, label="F1",        color="#378ADD")
    ax.bar(x,     summary["Precision"], width=w, label="Precision",  color="#1D9E75")
    ax.bar(x + w, summary["Recall"],    width=w, label="Recall",     color="#D85A30")
    ax.set_xticks(x); ax.set_xticklabels(summary["Method"])
    ax.set_ylim(0, 1); ax.legend(); ax.set_title("Method Comparison")
    ax.grid(axis="y", alpha=0.3)

    ax = axes[1]
    colors = ["#378ADD","#1D9E75","#D85A30","#9F4AB7"]
    for i, row in summary.iterrows():
        calls = row["API_calls"] if row["API_calls"] != "?" else 0
        ax.scatter(calls, row["F1"], s=200, color=colors[i % 4], zorder=3)
        ax.annotate(row["Method"], (calls, row["F1"]),
                    textcoords="offset points", xytext=(6, 3))
    ax.set_xlabel("Ollama calls per query"); ax.set_ylabel("F1 score")
    ax.set_title("Cost vs Accuracy"); ax.grid(alpha=0.3)

    plt.tight_layout()
    plot_path = os.path.join(out, f"comparison_{ts}.png")
    plt.savefig(plot_path, dpi=150)
    print(f"\nPlot saved: {plot_path}")
    plt.close()
    return summary


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--n",     type=int, default=50)
    p.add_argument("--fast",  action="store_true")
    p.add_argument("--heavy", action="store_true")
    args = parse_args = p.parse_args()
    df   = run_benchmark(n=args.n, include_heavy=args.heavy)
    report(df)
