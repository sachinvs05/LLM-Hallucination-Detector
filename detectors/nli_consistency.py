"""
detectors/nli_consistency.py  — FREE VERSION (Ollama + HuggingFace NLI)

NLI Consistency Hallucination Detector
========================================
1. Generate a reference paragraph using a local Ollama model.
2. Use a free HuggingFace NLI model to test:
       response ENTAILED by reference   → consistent, likely truthful
       response NEUTRAL                 → uncertain
       response CONTRADICTS reference   → hallucinated
3. score = contradiction_prob + 0.5 * neutral_prob

No API keys required.
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional

import ollama
from transformers import pipeline as hf_pipeline

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import config


@dataclass
class NLIResult:
    score: float
    entailment:    float = 0.0
    neutral:       float = 0.0
    contradiction: float = 0.0
    reference:     str   = ""
    verdict:       str   = ""

    def __post_init__(self):
        self.verdict = "HALLUCINATED" if self.score > config.HALLUCINATION_THRESH else "TRUTHFUL"

    def __str__(self):
        return (f"NLI Consistency | score={self.score:.3f} | "
                f"entail={self.entailment:.2f} neutral={self.neutral:.2f} "
                f"contradict={self.contradiction:.2f} | {self.verdict}")


REFERENCE_PROMPT = (
    "Write a short, factually accurate paragraph (3-5 sentences) about this topic. "
    "Use only well-established facts. Be precise.\n\nTopic: {question}"
)


class NLIConsistencyDetector:
    """
    Parameters
    ----------
    model     : str   Ollama model for reference generation
    nli_model : str   HuggingFace zero-shot-classification model
    """

    def __init__(
        self,
        model: str = config.OLLAMA_MODEL,
        nli_model: str = config.NLI_MODEL,
        num_reference_samples: int = 2,
    ):
        self.model                  = model
        self.num_reference_samples  = num_reference_samples
        print(f"[NLI] Loading NLI model '{nli_model}' …")
        self._nli = hf_pipeline("zero-shot-classification", model=nli_model, device=-1)

    def _build_reference(self, question: str) -> str:
        refs = []
        for _ in range(self.num_reference_samples):
            resp = ollama.chat(
                model=self.model,
                messages=[{"role": "user",
                           "content": REFERENCE_PROMPT.format(question=question)}],
                options={"temperature": 0.1},
            )
            refs.append(resp["message"]["content"].strip())
        return " ".join(refs)

    def _nli_score(self, hypothesis: str, premise: str):
        result = self._nli(
            sequences=premise[:512],
            candidate_labels=["entailment", "neutral", "contradiction"],
        )
        d = dict(zip(result["labels"], result["scores"]))
        return d.get("entailment", 0.0), d.get("neutral", 0.0), d.get("contradiction", 0.0)

    def detect(self, question: str, response: str,
               reference: Optional[str] = None) -> NLIResult:
        if reference is None:
            print("\n[NLI] Generating reference with Ollama …")
            reference = self._build_reference(question)
            print(f"  reference: {reference[:80]}…")

        print("[NLI] Running NLI scoring …")
        entail, neutral, contradict = self._nli_score(response, reference)
        score = float(np.clip(contradict + 0.5 * neutral, 0.0, 1.0))
        return NLIResult(score=score, entailment=entail, neutral=neutral,
                         contradiction=contradict, reference=reference)


if __name__ == "__main__":
    det = NLIConsistencyDetector()
    r = det.detect(
        "Which blood type is the universal donor?",
        "Type AB positive is the universal donor because it has all antigens.",
    )
    print("\n" + str(r))
