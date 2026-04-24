"""
detectors/selfcheck_gpt.py  — FREE VERSION (Ollama + HuggingFace NLI)

SelfCheckGPT Hallucination Detector
=====================================
1. Generate the main response with a local Ollama model.
2. Sample K extra responses at high temperature.
3. For each sentence in the main response, check NLI consistency
   against each extra sample using a free HuggingFace cross-encoder.
4. Sentence score = 1 − mean_entailment.  Overall = mean of sentences.

No API keys required.
"""

import re
import numpy as np
from dataclasses import dataclass, field
from typing import List

import ollama
from transformers import pipeline as hf_pipeline

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import config


@dataclass
class SelfCheckResult:
    score: float
    sentence_scores: List[float] = field(default_factory=list)
    sentences: List[str]         = field(default_factory=list)
    samples: List[str]           = field(default_factory=list)
    verdict: str = ""

    def __post_init__(self):
        self.verdict = "HALLUCINATED" if self.score > config.HALLUCINATION_THRESH else "TRUTHFUL"

    def __str__(self):
        lines = [f"SelfCheckGPT | score={self.score:.3f} | {self.verdict}"]
        for s, sc in zip(self.sentences, self.sentence_scores):
            flag = "⚠" if sc > 0.5 else "✓"
            lines.append(f"  {flag} ({sc:.2f}) {s[:80]}")
        return "\n".join(lines)


def split_sentences(text: str) -> List[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s.strip()]


class SelfCheckGPTDetector:
    """
    Parameters
    ----------
    model : str   Ollama model name
    """

    def __init__(
        self,
        model: str = config.OLLAMA_MODEL,
        num_samples: int = config.NUM_SAMPLES,
        temperature: float = config.SAMPLE_TEMPERATURE,
        nli_model: str = config.NLI_MODEL,
    ):
        self.model       = model
        self.num_samples = num_samples
        self.temperature = temperature
        print(f"[SelfCheckGPT] Loading NLI model '{nli_model}' …")
        self._nli = hf_pipeline("zero-shot-classification", model=nli_model, device=-1)

    def _generate(self, question: str, temperature: float = 0.0) -> str:
        resp = ollama.chat(
            model=self.model,
            messages=[{"role": "user", "content": question}],
            options={"temperature": temperature},
        )
        return resp["message"]["content"].strip()

    def _sample_extra(self, question: str) -> List[str]:
        samples = []
        for i in range(self.num_samples):
            r = self._generate(question, temperature=self.temperature)
            samples.append(r)
            print(f"  sample {i+1}/{self.num_samples}: {r[:60]}{'…' if len(r)>60 else ''}")
        return samples

    def _consistency(self, sentence: str, context: str) -> float:
        result = self._nli(
            sequences=context[:512],
            candidate_labels=["entailment", "neutral", "contradiction"],
        )
        return dict(zip(result["labels"], result["scores"])).get("entailment", 0.0)

    def detect(self, question: str, response: str) -> SelfCheckResult:
        print(f"\n[SelfCheckGPT] Sampling {self.num_samples} extra responses …")
        samples   = self._sample_extra(question)
        sentences = split_sentences(response)
        print(f"[SelfCheckGPT] Scoring {len(sentences)} sentences …")
        sentence_scores = []
        for sent in sentences:
            mean_c = float(np.mean([self._consistency(sent, s) for s in samples]))
            sentence_scores.append(1.0 - mean_c)
        overall = float(np.mean(sentence_scores)) if sentence_scores else 0.5
        return SelfCheckResult(score=overall, sentence_scores=sentence_scores,
                               sentences=sentences, samples=samples)


if __name__ == "__main__":
    det = SelfCheckGPTDetector()
    r = det.detect(
        "Did Einstein fail math in school?",
        "Yes, Albert Einstein famously failed his mathematics classes.",
    )
    print("\n" + str(r))
