#!/bin/bash
#SBATCH --job-name=gmm_segments
#SBATCH --array=0-7
#SBATCH --output=gmm_segment_%a.out
#SBATCH --error=gmm_segment_%a.err
#SBATCH --time=06:00:00
#SBATCH --mem=200G
#SBATCH --cpus-per-task=4
#SBATCH --partition=pibu_el8
#SBATCH --output=/data/users/ltucker/influenzaData/H5N1_pipeline/log/gmmSegment_%A_%a.out
#SBATCH --error=/data/users/ltucker/influenzaData/H5N1_pipeline/log/gmmSegment_%A_%a.err


SEGMENT_IDX=$SLURM_ARRAY_TASK_ID
SEGMENT_NAMES=("PB2" "PB1" "PA" "HA" "NP" "NA" "MP" "NS")
SEG_NAME=${SEGMENT_NAMES[$SEGMENT_IDX]}

INPUT_DIR="/data/users/ltucker/influenzaData/H5N1_pipeline/output/famsa_distances"
INPUT_FILE="${INPUT_DIR}/symmetric_distances_$((SEGMENT_IDX + 1)).parquet"
OUTPUT_DIR="/data/users/ltucker/influenzaData/H5N1_pipeline/output/famsa_per_segment_analysis/gmm"
mkdir -p "$OUTPUT_DIR"

CONTAINER="/data/users/ltucker/influenzaData/pipeline/jupyter-tensor_15.sif"

singularity exec "$CONTAINER" python3 << PYEOF
import numpy as np
import pandas as pd
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from scipy.stats import norm

seg_idx = ${SEGMENT_IDX}
seg_name = "${SEG_NAME}"
input_file = "${INPUT_FILE}"
output_dir = "${OUTPUT_DIR}"

print(f"Processing segment {seg_idx}: {seg_name}")
print(f"Reading {input_file}")

# Load distance matrix
df = pd.read_parquet(input_file)
upper_tri = df.values[np.triu_indices_from(df.values, k=1)].reshape(-1, 1)
print(f"Upper triangle size: {upper_tri.shape[0]} pairs")

# BIC search for best k
bics = []
models = []
for k in range(2, 8):
    gmm = GaussianMixture(n_components=k, random_state=42, n_init=5)
    gmm.fit(upper_tri)
    bic = gmm.bic(upper_tri)
    bics.append((k, bic))
    models.append(gmm)
    print(f"  k={k}, BIC={bic:.0f}")

best_idx = np.argmin([b[1] for b in bics])
best_k = bics[best_idx][0]
best_gmm = models[best_idx]
print(f"Best k={best_k}")

# Save model parameters as JSON (no pickle needed)
model_data = {
    "segment": seg_name,
    "best_k": int(best_k),
    "bics": [{"k": int(k), "bic": float(b)} for k, b in bics],
    "weights": best_gmm.weights_.tolist(),
    "means": best_gmm.means_.flatten().tolist(),
    "covariances": best_gmm.covariances_.flatten().tolist(),
}
with open(f"{output_dir}/gmm_{seg_name}.json", "w") as f:
    json.dump(model_data, f, indent=2)

# Assign each pair to its most likely component
labels = best_gmm.predict(upper_tri)

# === Plot 1: Histogram with GMM overlay ===
fig, axes = plt.subplots(2, 1, figsize=(12, 10))

colors = plt.cm.tab10(np.arange(best_k))
x_range = np.linspace(upper_tri.min(), upper_tri.max(), 1000).reshape(-1, 1)

# Top panel: density curves on histogram
axes[0].hist(upper_tri, bins=150, density=True, alpha=0.3, color="gray", edgecolor="none")
for i in range(best_k):
    weight = best_gmm.weights_[i]
    mean = best_gmm.means_[i, 0]
    std = np.sqrt(best_gmm.covariances_[i, 0, 0])
    curve = weight * norm.pdf(x_range, mean, std)
    axes[0].plot(x_range, curve, color=colors[i], linewidth=2,
                 label=f"k={i+1} (mu={mean:.3f}, s={std:.3f}, w={weight:.2f})")

total = np.exp(best_gmm.score_samples(x_range))
axes[0].plot(x_range, total, color="black", linewidth=2, linestyle="--", label="Total mixture")
axes[0].set_title(f"{seg_name} — GMM density overlay (k={best_k})")
axes[0].set_xlabel("Normalized distance")
axes[0].set_ylabel("Density")
axes[0].legend(fontsize=8)

# Bottom panel: stacked histogram by component
for i in range(best_k):
    subset = upper_tri[labels == i]
    axes[1].hist(subset, bins=150, alpha=0.6, color=colors[i],
                 label=f"Component {i+1} (n={len(subset)})", edgecolor="none")

axes[1].set_title(f"{seg_name} — Histogram colored by assignment (k={best_k})")
axes[1].set_xlabel("Normalized distance")
axes[1].set_ylabel("Count")
axes[1].legend(fontsize=8)

plt.tight_layout()
plt.savefig(f"{output_dir}/gmm_{seg_name}.png", dpi=150, bbox_inches="tight")
print(f"Saved plot to {output_dir}/gmm_{seg_name}.png")

# === Plot 2: BIC curve ===
fig, ax = plt.subplots(figsize=(8, 5))
ks, bic_vals = zip(*bics)
ax.plot(ks, bic_vals, "o-", color="steelblue")
ax.axvline(best_k, color="red", linestyle="--", label=f"Best k={best_k}")
ax.set_xlabel("Number of components")
ax.set_ylabel("BIC")
ax.set_title(f"{seg_name} — BIC vs k")
ax.legend()
plt.tight_layout()
plt.savefig(f"{output_dir}/gmm_bic_{seg_name}.png", dpi=150, bbox_inches="tight")
print(f"Saved BIC plot to {output_dir}/gmm_bic_{seg_name}.png")

PYEOF