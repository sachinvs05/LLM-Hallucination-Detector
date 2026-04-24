import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

methods = ["LogProb", "NLI", "Classifier"]
f1s     = [0.410,    0.552, 0.760]
precs   = [0.571,    0.485, 0.762]
recs    = [0.320,    0.640, 0.760]
calls   = [1,        2,     0]

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

x = np.arange(len(methods))
w = 0.25
ax = axes[0]
ax.bar(x - w, f1s,   width=w, label="F1",        color="#378ADD")
ax.bar(x,     precs, width=w, label="Precision",  color="#1D9E75")
ax.bar(x + w, recs,  width=w, label="Recall",     color="#D85A30")
ax.set_xticks(x)
ax.set_xticklabels(methods)
ax.set_ylim(0, 1)
ax.legend()
ax.set_title("Method Comparison")
ax.grid(axis="y", alpha=0.3)

ax = axes[1]
colors = ["#378ADD", "#1D9E75", "#D85A30"]
for i, (m, c, f) in enumerate(zip(methods, calls, f1s)):
    ax.scatter(c, f, s=200, color=colors[i], zorder=3)
    ax.annotate(m, (c, f), textcoords="offset points", xytext=(6, 3))
ax.set_xlabel("Ollama calls per query")
ax.set_ylabel("F1 score")
ax.set_title("Cost vs Accuracy")
ax.set_xlim(-0.5, 3)
ax.set_ylim(0.3, 0.9)
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("results/comparison_with_classifier.png", dpi=150)
print("Saved: results/comparison_with_classifier.png")
plt.close()
