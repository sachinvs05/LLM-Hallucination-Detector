"""
detectors/logprob_detector.py  — FREE VERSION (Ollama native logprobs)

Log-Probability Hallucination Detector
========================================
Ollama exposes per-token log-probabilities natively via the `logprobs` option.
We use the mean token log-prob as a proxy for model uncertainty:
    low mean log-prob  →  model is "unsure"  →  higher hallucination risk.

Falls back to a hedge-word keyword heuristic if logprobs unavailable.

No API keys required.
"""

import re
import math
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional

import ollama

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import config


@dataclass
class LogProbResult:
    score: float
    mean_log_prob: float
    method: str = "logprob"
    flagged_tokens: List[str] = field(default_factory=list)
    verdict: str = ""

    def __post_init__(self):
        self.verdict = "HALLUCINATED" if self.score > config.HALLUCINATION_THRESH else "TRUTHFUL"

    def __str__(self):
        return (f"LogProbability | score={self.score:.3f} | "
                f"mean_log_prob={self.mean_log_prob:.4f} | "
                f"method={self.method} | {self.verdict}")


HEDGE_PATTERN = re.compile(
    r"\b(I believe|I think|I'm not sure|probably|possibly|might be|could be|"
    r"it is said|some sources|reportedly|allegedly|according to some|"
    r"may have|might have|I cannot be certain|I am not entirely sure)\b",
    re.IGNORECASE,
)


class LogProbDetector:
    """
    Uses Ollama's native logprob support.

    Parameters
    ----------
    model : str   Ollama model name. Must support logprobs (llama3, mistral do).
    """

    def __init__(self, model: str = config.OLLAMA_MODEL):
        self.model = model

    def _get_logprobs(self, question: str, response: str) -> Optional[List[float]]:
        """
        Re-score `response` conditioned on `question` using Ollama logprobs.
        Returns a list of per-token log-probabilities, or None if not supported.
        """
        try:
            result = ollama.generate(
                model=self.model,
                prompt=f"Question: {question}\nAnswer: {response}",
                options={"temperature": 0, "logprobs": True, "num_predict": 0},
            )
            # Ollama returns logprobs in result["logprobs"] as a list of dicts
            lp = result.get("logprobs")
            if lp and isinstance(lp, list):
                return [entry.get("logprob", 0.0) for entry in lp if "logprob" in entry]
        except Exception:
            pass
        return None

    def _confidence_prompt(self, question: str, response: str) -> float:
        """
        Fallback: ask the model to rate its own confidence 0-10.
        Returns a value in [0, 1].
        """
        prompt = (
            f"Rate how factually accurate this answer is on a scale 0-10.\n"
            f"Question: {question}\nAnswer: {response}\n"
            f"Reply with ONLY a single number."
        )
        try:
            resp = ollama.chat(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                options={"temperature": 0},
            )
            raw = resp["message"]["content"].strip()
            m = re.search(r"\d+(\.\d+)?", raw)
            if m:
                return min(float(m.group()), 10.0) / 10.0
        except Exception:
            pass
        return 0.5

    def detect(self, question: str, response: str) -> LogProbResult:
        print("\n[LogProbability] Scoring response …")

        # Try native logprobs first
        logprobs = self._get_logprobs(question, response)
        if logprobs:
            mean_lp    = float(np.mean(logprobs))
            # Normalise: typical range is roughly [-5, 0]; map to [0, 1] hallucination score
            # score=0 → confident (lp near 0), score=1 → uncertain (lp very negative)
            score      = float(np.clip(-mean_lp / 5.0, 0.0, 1.0))
            method     = "native_logprob"
        else:
            # Fallback: confidence prompt + hedge words
            confidence = self._confidence_prompt(question, response)
            hedges     = HEDGE_PATTERN.findall(response)
            penalty    = min(len(hedges) * 0.05, 0.2)
            score      = float(np.clip(1.0 - confidence + penalty, 0.0, 1.0))
            mean_lp    = -(score * 5.0)   # synthetic for display
            method     = "confidence_prompt"

        flagged = HEDGE_PATTERN.findall(response)
        print(f"  method={method}  mean_log_prob={mean_lp:.4f}  score={score:.3f}  hedges={flagged}")
        return LogProbResult(score=score, mean_log_prob=mean_lp,
                             method=method, flagged_tokens=flagged)


if __name__ == "__main__":
    det = LogProbDetector()
    r = det.detect(
        "Can humans breathe underwater with practice?",
        "With enough practice, free-divers can extract oxygen from water.",
    )
    print("\n" + str(r))
