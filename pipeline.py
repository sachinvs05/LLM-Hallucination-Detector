"""
pipeline.py — FREE VERSION

Weighted ensemble of all four free detectors.
No API keys. Uses Ollama + HuggingFace only.

Usage
-----
    from pipeline import HallucinationPipeline
    pipe   = HallucinationPipeline()
    result = pipe.detect("Did Einstein fail math?", "Yes, he failed math class.")
    print(result)
"""

from dataclasses import dataclass, field
from typing import Optional, List
import numpy as np

import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import config

from detectors.semantic_entropy import SemanticEntropyDetector
from detectors.selfcheck_gpt    import SelfCheckGPTDetector
from detectors.logprob_detector import LogProbDetector
from detectors.nli_consistency  import NLIConsistencyDetector


@dataclass
class PipelineResult:
    score: float
    verdict: str
    method_scores: dict = field(default_factory=dict)
    confidence:    float = 0.0

    def __str__(self):
        lines = [
            f"\n{'='*55}",
            f"  ENSEMBLE RESULT",
            f"  Score:   {self.score:.3f}  ({self.verdict})",
            f"  Agreement across detectors: {self.confidence:.0%}",
            f"{'─'*55}",
            f"  Per-method breakdown:",
        ]
        for method, score in self.method_scores.items():
            bar = "█" * int(score * 20) + "░" * (20 - int(score * 20))
            lines.append(f"    {method:<22} {bar} {score:.3f}")
        lines.append("=" * 55)
        return "\n".join(lines)


WEIGHT_MAP = {
    "semantic":   0.35,
    "selfcheck":  0.30,
    "logprob":    0.15,
    "nli":        0.20,
    "classifier": 0.40,
}


class HallucinationPipeline:
    """
    Parameters
    ----------
    detectors : list[str] | None
        Subset of ["semantic","selfcheck","logprob","nli","classifier"].
        None = all four free detectors.
    fast_mode : bool
        If True, only logprob + nli (fastest, ~2 Ollama calls).
    classifier_path : str | None
        Path to a saved HallucinationClassifier (optional).
    """

    def __init__(
        self,
        detectors: Optional[List[str]] = None,
        weights: Optional[List[float]] = None,
        classifier_path: Optional[str] = None,
        fast_mode: bool = False,
    ):
        if fast_mode:
            detectors = detectors or ["logprob", "nli"]
        else:
            detectors = detectors or ["semantic", "selfcheck", "logprob", "nli"]

        self.detector_names = detectors
        self._detectors     = {}

        for name in detectors:
            if name == "semantic":
                print("[Pipeline] Init SemanticEntropyDetector …")
                self._detectors["semantic"] = SemanticEntropyDetector()
            elif name == "selfcheck":
                print("[Pipeline] Init SelfCheckGPTDetector …")
                self._detectors["selfcheck"] = SelfCheckGPTDetector()
            elif name == "logprob":
                print("[Pipeline] Init LogProbDetector …")
                self._detectors["logprob"] = LogProbDetector()
            elif name == "nli":
                print("[Pipeline] Init NLIConsistencyDetector …")
                self._detectors["nli"] = NLIConsistencyDetector()
            elif name == "classifier":
                from models.custom_classifier import HallucinationClassifier
                self._detectors["classifier"] = HallucinationClassifier.load(classifier_path)
            else:
                raise ValueError(f"Unknown detector: {name!r}")

        if weights is not None:
            total = sum(weights)
            self._weights = {n: w / total for n, w in zip(detectors, weights)}
        else:
            raw = {n: WEIGHT_MAP.get(n, 1.0) for n in detectors}
            total = sum(raw.values())
            self._weights = {n: v / total for n, v in raw.items()}

        print(f"[Pipeline] Ready — detectors: {', '.join(detectors)}")

    def detect(self, question: str, response: str) -> PipelineResult:
        method_scores = {}
        for name, detector in self._detectors.items():
            print(f"\n[Pipeline] Running {name} …")
            try:
                if name == "semantic":
                    r = detector.detect(question)
                elif name == "classifier":
                    method_scores[name] = float(detector.predict_proba(question, response))
                    continue
                else:
                    r = detector.detect(question, response)
                method_scores[name] = float(r.score)
            except Exception as e:
                print(f"  [Warning] {name} failed: {e}")
                method_scores[name] = 0.5

        ensemble = float(sum(method_scores[n] * self._weights[n] for n in method_scores))

        if ensemble > 0.65:
            verdict = "HALLUCINATED"
        elif ensemble > 0.40:
            verdict = "UNCERTAIN"
        else:
            verdict = "TRUTHFUL"

        confidence = float(1.0 - np.std(list(method_scores.values())))
        return PipelineResult(score=ensemble, verdict=verdict,
                              method_scores=method_scores, confidence=confidence)


if __name__ == "__main__":
    pipe = HallucinationPipeline(fast_mode=True)
    examples = [
        ("Which blood type is the universal donor?",
         "Type AB positive is the universal donor."),
        ("Which blood type is the universal donor?",
         "Type O-negative (O-) is the universal red cell donor."),
    ]
    for q, r in examples:
        print(pipe.detect(q, r))
