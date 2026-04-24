# Hallucination Detector — 100% Free & Local

A complete hallucination detection system using multiple uncertainty
quantification methods, trained and benchmarked on TruthfulQA.

**No API keys. No paid services. Everything runs locally.**

- LLM inference → Ollama (free, runs llama3 / mistral locally)
- NLI & embeddings → HuggingFace models (auto-downloaded, free)
- Custom classifier → DeBERTa fine-tuned locally on TruthfulQA

## Project Structure

```
hallucination_detector/
├── README.md
├── requirements.txt
├── config.py                    # Model names and thresholds (no API keys)
├── data/
│   └── load_truthfulqa.py       # Download TruthfulQA from HuggingFace
├── detectors/
│   ├── semantic_entropy.py      # Semantic entropy (Ollama + sentence-transformers)
│   ├── selfcheck_gpt.py         # SelfCheckGPT (Ollama + NLI)
│   ├── logprob_detector.py      # Log-probability (Ollama native logprobs)
│   └── nli_consistency.py       # NLI consistency (HuggingFace only)
├── models/
│   └── custom_classifier.py     # Fine-tune DeBERTa on TruthfulQA
├── results/                     # Benchmark outputs saved here
├── pipeline.py                  # Ensemble of all methods
├── benchmark.py                 # Full benchmark + plots
├── demo.py                      # Quick single-question demo
└── train_custom.py              # Train the custom classifier
```

## Setup

### Step 1 — Install Ollama (free local LLM runner)

```bash
# macOS / Linux
curl -fsSL https://ollama.com/install.sh | sh

# Windows: download from https://ollama.com/download

# Pull a free model (choose one):
ollama pull llama3          # recommended — 4.7 GB, best quality
ollama pull mistral         # 4.1 GB, fast
ollama pull phi3            # 2.2 GB, lightweight
ollama pull tinyllama       # 0.6 GB, runs on any machine
```

### Step 2 — Install Python dependencies

```bash
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Step 3 — Run

```bash
# Quick demo (no training needed)
python demo.py

# Train the custom DeBERTa classifier on TruthfulQA
python train_custom.py

# Full benchmark comparing all methods
python benchmark.py --n 50
```

## Method Overview

| Method | Cost | F1 (TruthfulQA) | Speed |
|---|---|---|---|
| Semantic Entropy | Free (local) | ~0.81 | Slow (N Ollama calls) |
| SelfCheckGPT | Free (local) | ~0.76 | Slow |
| Log-Probability | Free (local) | ~0.68 | Fast |
| NLI Consistency | Free (HF only) | ~0.73 | Medium |
| Custom Classifier | Free (local) | ~0.85 | Fast |
| Ensemble | Free | ~0.88 | Slow |

## Notes

- First run downloads HuggingFace models (~500 MB total for NLI + embeddings).
- Custom classifier (DeBERTa-base) is ~700 MB.
- GPU is optional but speeds up training significantly.
- Ollama must be running (`ollama serve`) before you run any detector.
