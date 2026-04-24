"""
config.py — shared configuration. No API keys needed.

The LLM runs locally via Ollama. Change OLLAMA_MODEL to any model
you have pulled: llama3, mistral, phi3, tinyllama, etc.
"""

# ── Ollama (free local LLM) ────────────────────────────────────────────────
OLLAMA_MODEL      = "tinyllama"      # change to mistral / phi3 / tinyllama
OLLAMA_BASE_URL   = "http://localhost:11434"

# ── Detector defaults ──────────────────────────────────────────────────────
NUM_SAMPLES          = 5       # responses sampled per query
SAMPLE_TEMPERATURE   = 0.8
SIMILARITY_THRESHOLD = 0.85
HALLUCINATION_THRESH = 0.50

# ── HuggingFace models (downloaded automatically, free) ───────────────────
NLI_MODEL       = "cross-encoder/nli-MiniLM2-L6-H768"   # ~85 MB
EMBEDDING_MODEL = "all-MiniLM-L6-v2"                    # ~80 MB

# ── Custom classifier (DeBERTa fine-tuned on TruthfulQA) ──────────────────
CLASSIFIER_BASE_MODEL = "distilbert-base-uncased"   
CLASSIFIER_SAVE_PATH  = "models/saved_classifier"
TRAIN_EPOCHS          = 3
TRAIN_LR              = 2e-5
TRAIN_BATCH_SIZE      = 2
MAX_SEQ_LENGTH        = 128

# ── Benchmark ─────────────────────────────────────────────────────────────
BENCHMARK_N_EXAMPLES = 50
RESULTS_DIR          = "results"
