"""
train_custom.py — FREE VERSION

Fine-tunes DeBERTa-v3-base on TruthfulQA labelled pairs.
100% local — no internet calls during training (only HuggingFace model download).

Run:
    python train_custom.py
    python train_custom.py --examples 500 --epochs 3
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import argparse, os, json
from sklearn.model_selection import train_test_split

import sys
sys.path.insert(0, os.path.dirname(__file__))
import config
from data.load_truthfulqa     import build_labelled_dataframe
from models.custom_classifier import HallucinationClassifier


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--examples",   type=int,   default=1000)
    p.add_argument("--epochs",     type=int,   default=config.TRAIN_EPOCHS)
    p.add_argument("--batch_size", type=int,   default=config.TRAIN_BATCH_SIZE)
    p.add_argument("--lr",         type=float, default=config.TRAIN_LR)
    p.add_argument("--save_path",  type=str,   default=config.CLASSIFIER_SAVE_PATH)
    p.add_argument("--val_split",  type=float, default=0.15)
    p.add_argument("--seed",       type=int,   default=42)
    return p.parse_args()


def main():
    args = parse_args()
    print("=" * 60)
    print("  DeBERTa Hallucination Classifier — TruthfulQA Fine-tune")
    print("=" * 60)
    print("  No API keys. Runs entirely locally.\n")

    df = build_labelled_dataframe(max_examples=args.examples, seed=args.seed)
    print(f"Dataset: {len(df)} rows | "
          f"{(df.label==0).sum()} truthful | {(df.label==1).sum()} hallucinated\n")

    train_df, val_df = train_test_split(
        df, test_size=args.val_split,
        stratify=df["label"], random_state=args.seed,
    )

    clf = HallucinationClassifier(
        base_model=config.CLASSIFIER_BASE_MODEL,
        max_length=config.MAX_SEQ_LENGTH,
    )

    history = clf.fit(
        questions   = train_df["question"].tolist(),
        answers     = train_df["answer"].tolist(),
        labels      = train_df["label"].tolist(),
        val_questions = val_df["question"].tolist(),
        val_answers   = val_df["answer"].tolist(),
        val_labels    = val_df["label"].tolist(),
        epochs     = args.epochs,
        batch_size = args.batch_size,
        learning_rate = args.lr,
    )

    print("\nFinal evaluation:")
    metrics = clf.evaluate(val_df["question"].tolist(),
                           val_df["answer"].tolist(),
                           val_df["label"].tolist())
    print(metrics.report)
    print(f"ROC-AUC: {metrics.auc:.4f}")

    clf.save(args.save_path)

    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    with open(os.path.join(config.RESULTS_DIR, "training_history.json"), "w") as f:
        json.dump(history, f, indent=2)

    # Quick sanity checks
    print("\n── Sanity checks ──")
    for q, a, expected in [
        ("Does the Great Wall show up from space?",
         "Yes, it's the only man-made structure visible from the Moon.", "HALLUCINATED"),
        ("Does the Great Wall show up from space?",
         "No. It's too narrow to see from space without magnification.", "TRUTHFUL"),
    ]:
        prob = clf.predict_proba(q, a)
        pred = "HALLUCINATED" if prob > 0.5 else "TRUTHFUL"
        ok   = "✓" if pred == expected else "✗"
        print(f"  {ok} P(hallucinated)={prob:.3f} → {pred}  (expected {expected})")

    print(f"\nDone! Model saved to: {args.save_path}")


if __name__ == "__main__":
    main()
