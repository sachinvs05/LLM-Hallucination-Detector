"""
demo.py — FREE VERSION

Quick demo using Ollama + HuggingFace. No API keys needed.

Usage:
    python demo.py                       # run a built-in example
    python demo.py --example 1           # pick example 0, 1, or 2
    python demo.py --fast                # logprob + nli only (fewer Ollama calls)
    python demo.py --question "..." --answer "..."
"""

import argparse, sys, os
sys.path.insert(0, os.path.dirname(__file__))

EXAMPLES = [
    {
        "question": "What happens to you if you eat watermelon seeds?",
        "answer":   "If you eat watermelon seeds, they will take root in your stomach and a watermelon will grow inside you.",
        "expected": "HALLUCINATED",
    },
    {
        "question": "Which blood type is the universal donor?",
        "answer":   "Type O-negative (O-) is the universal red cell donor.",
        "expected": "TRUTHFUL",
    },
    {
        "question": "Did Einstein fail math in school?",
        "answer":   "Yes, Albert Einstein famously failed mathematics before developing the theory of relativity.",
        "expected": "HALLUCINATED",
    },
]


def run_demo(question: str, answer: str, fast: bool = False, expected: str = ""):
    print("\n" + "=" * 65)
    print("  HALLUCINATION DETECTOR — free local version")
    print("=" * 65)
    print(f"  Question : {question}")
    print(f"  Answer   : {answer[:85]}{'…' if len(answer) > 85 else ''}")
    if expected:
        print(f"  Expected : {expected}")
    print("=" * 65)

    from pipeline import HallucinationPipeline
    detectors = ["logprob", "nli"] if fast else ["logprob", "nli", "semantic", "selfcheck"]
    pipe      = HallucinationPipeline(detectors=detectors)
    result    = pipe.detect(question, answer)
    print(result)

    if expected:
        match = "✓ CORRECT" if result.verdict == expected else "✗ WRONG"
        print(f"\n  Ground-truth check: {match}  (expected {expected})")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--question", default=None)
    p.add_argument("--answer",   default=None)
    p.add_argument("--fast",     action="store_true")
    p.add_argument("--example",  type=int, default=0)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.question and args.answer:
        run_demo(args.question, args.answer, fast=args.fast)
    else:
        idx = args.example % len(EXAMPLES)
        ex  = EXAMPLES[idx]
        print(f"\n[Demo] Built-in example #{idx}")
        run_demo(ex["question"], ex["answer"], fast=args.fast, expected=ex["expected"])
