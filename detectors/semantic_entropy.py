"""
detectors/semantic_entropy.py  — FREE VERSION (Ollama + sentence-transformers)

Semantic Entropy Hallucination Detector
========================================
1. Sample N responses from a local LLM (Ollama) at high temperature.
2. Cluster by semantic similarity using sentence-transformers embeddings.
3. Compute Shannon entropy over clusters — high entropy = likely hallucinating.

No API keys required.
"""

import math
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional

import ollama
from sentence_transformers import SentenceTransformer

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import config


@dataclass
class SemanticEntropyResult:
    score: float
    entropy: float
    num_clusters: int
    num_samples: int
    samples: List[str] = field(default_factory=list)
    cluster_labels: List[int] = field(default_factory=list)
    verdict: str = ""

    def __post_init__(self):
        self.verdict = "HALLUCINATED" if self.score > config.HALLUCINATION_THRESH else "TRUTHFUL"

    def __str__(self):
        return (f"SemanticEntropy | score={self.score:.3f} | entropy={self.entropy:.3f}"
                f" | clusters={self.num_clusters}/{self.num_samples} | {self.verdict}")


class SemanticEntropyDetector:
    """
    Parameters
    ----------
    model : str   Ollama model name (llama3, mistral, phi3, tinyllama…)
    """

    def __init__(
        self,
        model: str = config.OLLAMA_MODEL,
        num_samples: int = config.NUM_SAMPLES,
        temperature: float = config.SAMPLE_TEMPERATURE,
        similarity_threshold: float = config.SIMILARITY_THRESHOLD,
        embedding_model: str = config.EMBEDDING_MODEL,
    ):
        self.model               = model
        self.num_samples         = num_samples
        self.temperature         = temperature
        self.similarity_threshold = similarity_threshold
        print(f"[SemanticEntropy] Loading embedding model '{embedding_model}' …")
        self.embedder = SentenceTransformer(embedding_model)

    def _sample(self, question: str) -> List[str]:
        responses = []
        for i in range(self.num_samples):
            resp = ollama.chat(
                model=self.model,
                messages=[{"role": "user", "content": question}],
                options={"temperature": self.temperature},
            )
            text = resp["message"]["content"].strip()
            responses.append(text)
            print(f"  sample {i+1}/{self.num_samples}: {text[:70]}{'…' if len(text)>70 else ''}")
        return responses

    def _cluster(self, responses: List[str]) -> List[int]:
        embeddings = self.embedder.encode(responses, normalize_embeddings=True)
        labels, centroids = [-1] * len(responses), []
        for i, emb in enumerate(embeddings):
            assigned = False
            for c_id, centroid in enumerate(centroids):
                if float(np.dot(emb, centroid)) >= self.similarity_threshold:
                    labels[i] = c_id
                    n = labels[:i].count(c_id) + 1
                    centroids[c_id] = (centroid * (n - 1) + emb) / n
                    centroids[c_id] /= np.linalg.norm(centroids[c_id])
                    assigned = True
                    break
            if not assigned:
                labels[i] = len(centroids)
                centroids.append(emb.copy())
        return labels

    @staticmethod
    def _entropy(labels: List[int]) -> float:
        counts = np.bincount(labels)
        probs  = counts / counts.sum()
        return float(-np.sum(p * math.log(p + 1e-12) for p in probs if p > 0))

    def detect(self, question: str, response: Optional[str] = None) -> SemanticEntropyResult:
        print(f"\n[SemanticEntropy] Sampling {self.num_samples} responses via Ollama …")
        samples        = self._sample(question)
        cluster_labels = self._cluster(samples)
        num_clusters   = max(cluster_labels) + 1
        entropy        = self._entropy(cluster_labels)
        max_entropy    = math.log(self.num_samples + 1e-12)
        score          = float(np.clip(entropy / max_entropy, 0.0, 1.0))
        return SemanticEntropyResult(
            score=score, entropy=entropy, num_clusters=num_clusters,
            num_samples=self.num_samples, samples=samples, cluster_labels=cluster_labels,
        )


if __name__ == "__main__":
    det = SemanticEntropyDetector()
    r   = det.detect("What is the largest animal that has ever existed?")
    print("\n" + str(r))
