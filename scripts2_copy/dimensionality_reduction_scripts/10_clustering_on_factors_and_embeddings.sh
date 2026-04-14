#!/bin/bash

#SBATCH --job-name=cluster_tucker
#SBATCH --time=04:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --partition=pibu_el8
#SBATCH --output=/data/users/ltucker/influenzaData/H5N1_pipeline/log/tensor/cluster_tucker_%j.out
#SBATCH --error=/data/users/ltucker/influenzaData/H5N1_pipeline/log/tensor/cluster_tucker_%j.err

DR_DIR="/data/users/ltucker/influenzaData/H5N1_pipeline/output/famsa_tensor_analysis/dimensionality_reduction"
TENSOR_DIR="/data/users/ltucker/influenzaData/H5N1_pipeline/output/famsa_tensor_analysis/tensor"
OUTPUT_DIR="/data/users/ltucker/influenzaData/H5N1_pipeline/output/famsa_tensor_analysis/clustering"
LOG_DIR="/data/users/ltucker/influenzaData/H5N1_pipeline/log/tensor"
mkdir -p "$OUTPUT_DIR"
mkdir -p "$LOG_DIR"

CONTAINER="/data/users/ltucker/influenzaData/pipeline/jupyter-tensor_15.sif"

export DR_DIR
export TENSOR_DIR
export OUTPUT_DIR

# Must match tucker decomposition params
export RANK="4-5-5"
export SEED=42

echo "Starting HDBSCAN + KMeans clustering on Tucker factor DR"
echo "DR_DIR: $DR_DIR"
echo "OUTPUT_DIR: $OUTPUT_DIR"

apptainer exec "$CONTAINER" python3 << 'PYTHON_SCRIPT'

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', buffering=1)
sys.stderr = os.fdopen(sys.stderr.fileno(), 'w', buffering=1)

try:
    from sklearn.cluster import HDBSCAN
    print("Using sklearn.cluster.HDBSCAN", flush=True)
    USE_SKLEARN_HDBSCAN = True
except ImportError:
    try:
        import hdbscan
        print("Using hdbscan package", flush=True)
        USE_SKLEARN_HDBSCAN = False
    except ImportError:
        print("ERROR: No HDBSCAN available.", flush=True)
        print("Either upgrade scikit-learn>=1.3 or install hdbscan package.", flush=True)
        sys.exit(1)

seed = int(os.environ["SEED"])
rank_str = os.environ["RANK"]
dr_dir = Path(os.environ["DR_DIR"])
tensor_dir = Path(os.environ["TENSOR_DIR"])
output_dir = Path(os.environ["OUTPUT_DIR"])
output_dir.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------
# 1. Load data
# ---------------------------------------------------------------
print("=" * 60, flush=True)
print("Loading data...", flush=True)
print("=" * 60, flush=True)

# DR embeddings (2D) for both factors
dr1 = pd.read_parquet(dr_dir / "dr_all_factor1_samples_row.parquet")
dr2 = pd.read_parquet(dr_dir / "dr_all_factor2_samples_col.parquet")

# Raw factor matrices (high-D)
base_name = f"tucker_full_rank{rank_str}_seed{seed}"
factor1_raw = np.load(tensor_dir / f"{base_name}_factor1.npy")
factor2_raw = np.load(tensor_dir / f"{base_name}_factor2.npy")

print(f"  DR factor1: {dr1.shape}", flush=True)
print(f"  DR factor2: {dr2.shape}", flush=True)
print(f"  Raw factor1: {factor1_raw.shape}", flush=True)
print(f"  Raw factor2: {factor2_raw.shape}", flush=True)

# Parse metadata
metadata = dr1[["sample_id", "country", "continent", "year"]].copy()

# ---------------------------------------------------------------
# 2. Define inputs to cluster
# ---------------------------------------------------------------
inputs = {
    # 2D DR embeddings
    "factor1_tsne_2d": dr1[["tsne_1", "tsne_2"]].values,
    "factor1_umap_2d": dr1[["umap_1", "umap_2"]].values,
    "factor1_mds_2d":  dr1[["mds_1", "mds_2"]].values,
    "factor2_tsne_2d": dr2[["tsne_1", "tsne_2"]].values,
    "factor2_umap_2d": dr2[["umap_1", "umap_2"]].values,
    "factor2_mds_2d":  dr2[["mds_1", "mds_2"]].values,
    # Raw high-D factor matrices
    "factor1_raw":     factor1_raw,
    "factor2_raw":     factor2_raw,
}

# ---------------------------------------------------------------
# 3. HDBSCAN
# ---------------------------------------------------------------
print("\n" + "=" * 60, flush=True)
print("Running HDBSCAN (min_cluster_size=50, min_samples=30)", flush=True)
print("=" * 60, flush=True)

hdbscan_results = {}

for name, X in inputs.items():
    print(f"\n  {name} (shape: {X.shape})...", flush=True)

    if USE_SKLEARN_HDBSCAN:
        clusterer = HDBSCAN(
            min_cluster_size=50,
            min_samples=30,
            n_jobs=4,
        )
    else:
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=50,
            min_samples=30,
            core_dist_n_jobs=4,
        )
    labels = clusterer.fit_predict(X)

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = (labels == -1).sum()
    pct_noise = 100 * n_noise / len(labels)

    print(f"    Clusters: {n_clusters}, Noise: {n_noise} ({pct_noise:.1f}%)", flush=True)

    # Silhouette on non-noise points
    mask = labels >= 0
    if len(set(labels[mask])) >= 2:
        sil = silhouette_score(X[mask], labels[mask])
        print(f"    Silhouette (excl. noise): {sil:.3f}", flush=True)
    else:
        sil = np.nan
        print(f"    Silhouette: N/A (fewer than 2 clusters)", flush=True)

    hdbscan_results[name] = {
        "labels": labels,
        "n_clusters": n_clusters,
        "n_noise": n_noise,
        "pct_noise": pct_noise,
        "silhouette": sil,
    }

    # Save labels
    out_df = metadata.copy()
    out_df[f"hdbscan_cluster"] = labels
    out_df.to_parquet(output_dir / f"hdbscan_{name}.parquet", index=False)

# ---------------------------------------------------------------
# 4. KMeans (K=2..30)
# ---------------------------------------------------------------
print("\n" + "=" * 60, flush=True)
print("Running KMeans (K=2..30)", flush=True)
print("=" * 60, flush=True)

k_range = range(2, 31)
kmeans_results = {}

for name, X in inputs.items():
    print(f"\n  {name} (shape: {X.shape})...", flush=True)

    scores = []
    all_labels = {}

    for k in k_range:
        km = KMeans(n_clusters=k, random_state=seed, n_init=10, max_iter=300)
        labels = km.fit_predict(X)
        all_labels[k] = labels

        sil = silhouette_score(X, labels)
        ch = calinski_harabasz_score(X, labels)
        db = davies_bouldin_score(X, labels)

        scores.append({
            "k": k,
            "silhouette": sil,
            "calinski_harabasz": ch,
            "davies_bouldin": db,
            "inertia": km.inertia_,
        })

    scores_df = pd.DataFrame(scores)
    best_k_sil = scores_df.loc[scores_df["silhouette"].idxmax(), "k"]
    best_k_db = scores_df.loc[scores_df["davies_bouldin"].idxmin(), "k"]

    print(f"    Best K (silhouette): {int(best_k_sil)} "
          f"(score: {scores_df['silhouette'].max():.3f})", flush=True)
    print(f"    Best K (Davies-Bouldin): {int(best_k_db)} "
          f"(score: {scores_df['davies_bouldin'].min():.3f})", flush=True)

    kmeans_results[name] = {
        "scores": scores_df,
        "all_labels": all_labels,
        "best_k_sil": int(best_k_sil),
        "best_k_db": int(best_k_db),
    }

    # Save scores
    scores_df.to_parquet(output_dir / f"kmeans_scores_{name}.parquet", index=False)

    # Save best labels
    for best_k, metric_name in [(int(best_k_sil), "sil"), (int(best_k_db), "db")]:
        out_df = metadata.copy()
        out_df[f"kmeans_k{best_k}_cluster"] = all_labels[best_k]
        out_df.to_parquet(
            output_dir / f"kmeans_best_{metric_name}_{name}_k{best_k}.parquet",
            index=False
        )

    # Save all K labels
    out_df = metadata.copy()
    for k in k_range:
        out_df[f"kmeans_k{k}"] = all_labels[k]
    out_df.to_parquet(output_dir / f"kmeans_all_k_{name}.parquet", index=False)

# ---------------------------------------------------------------
# 5. Summary table
# ---------------------------------------------------------------
print("\n" + "=" * 60, flush=True)
print("Summary", flush=True)
print("=" * 60, flush=True)

summary_rows = []
for name in inputs:
    h = hdbscan_results[name]
    km = kmeans_results[name]
    summary_rows.append({
        "input": name,
        "dims": inputs[name].shape[1],
        "hdbscan_clusters": h["n_clusters"],
        "hdbscan_noise_pct": round(h["pct_noise"], 1),
        "hdbscan_silhouette": round(h["silhouette"], 3) if not np.isnan(h["silhouette"]) else "N/A",
        "kmeans_best_k_sil": km["best_k_sil"],
        "kmeans_best_sil": round(km["scores"]["silhouette"].max(), 3),
        "kmeans_best_k_db": km["best_k_db"],
        "kmeans_best_db": round(km["scores"]["davies_bouldin"].min(), 3),
    })

summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv(output_dir / "clustering_summary.csv", index=False)
print(summary_df.to_string(index=False), flush=True)

# ---------------------------------------------------------------
# 6. Plots
# ---------------------------------------------------------------
print("\nCreating plots...", flush=True)

# --- KMeans elbow + silhouette plots ---
fig, axes = plt.subplots(len(inputs), 3, figsize=(18, 4 * len(inputs)))
for row, name in enumerate(inputs):
    scores_df = kmeans_results[name]["scores"]

    # Elbow (inertia)
    ax = axes[row, 0]
    ax.plot(scores_df["k"], scores_df["inertia"], "o-", markersize=3)
    ax.set_title(f"{name}", fontsize=10)
    ax.set_xlabel("K")
    ax.set_ylabel("Inertia")
    ax.grid(True, alpha=0.3)
    best_k = kmeans_results[name]["best_k_sil"]
    ax.axvline(best_k, color="red", linestyle="--", alpha=0.7, label=f"Best K={best_k}")
    ax.legend(fontsize=8)

    # Silhouette
    ax = axes[row, 1]
    ax.plot(scores_df["k"], scores_df["silhouette"], "o-", markersize=3, color="green")
    ax.set_xlabel("K")
    ax.set_ylabel("Silhouette")
    ax.grid(True, alpha=0.3)
    ax.axvline(best_k, color="red", linestyle="--", alpha=0.7)

    # Davies-Bouldin
    ax = axes[row, 2]
    ax.plot(scores_df["k"], scores_df["davies_bouldin"], "o-", markersize=3, color="purple")
    ax.set_xlabel("K")
    ax.set_ylabel("Davies-Bouldin")
    ax.grid(True, alpha=0.3)
    best_k_db = kmeans_results[name]["best_k_db"]
    ax.axvline(best_k_db, color="red", linestyle="--", alpha=0.7, label=f"Best K={best_k_db}")
    ax.legend(fontsize=8)

plt.suptitle("KMeans: Elbow, Silhouette, Davies-Bouldin", fontsize=16, y=1.01)
plt.tight_layout()
plt.savefig(output_dir / "kmeans_metrics.png", dpi=300, bbox_inches="tight")
plt.close()
print("  Saved kmeans_metrics.png", flush=True)

# --- HDBSCAN scatter plots (2D inputs only) ---
dr_inputs_2d = [k for k in inputs if "_2d" in k]
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
for idx, name in enumerate(dr_inputs_2d):
    row, col = divmod(idx, 3)
    ax = axes[row, col]
    X = inputs[name]
    labels = hdbscan_results[name]["labels"]
    n_cl = hdbscan_results[name]["n_clusters"]

    noise_mask = labels == -1
    ax.scatter(X[noise_mask, 0], X[noise_mask, 1], c="#cccccc", s=1, alpha=0.2, label="noise")
    for cl in range(n_cl):
        mask = labels == cl
        ax.scatter(X[mask, 0], X[mask, 1], s=2, alpha=0.4, label=f"C{cl}")
    ax.set_title(f"{name}\n{n_cl} clusters, {hdbscan_results[name]['pct_noise']:.0f}% noise",
                 fontsize=10)
    ax.tick_params(labelsize=7)
    if n_cl <= 15:
        ax.legend(fontsize=6, markerscale=3, ncol=2)

plt.suptitle("HDBSCAN Clusters (2D embeddings)", fontsize=16)
plt.tight_layout()
plt.savefig(output_dir / "hdbscan_scatter_2d.png", dpi=300, bbox_inches="tight")
plt.close()
print("  Saved hdbscan_scatter_2d.png", flush=True)

# --- KMeans scatter plots (2D, best K by silhouette) ---
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
for idx, name in enumerate(dr_inputs_2d):
    row, col = divmod(idx, 3)
    ax = axes[row, col]
    X = inputs[name]
    best_k = kmeans_results[name]["best_k_sil"]
    labels = kmeans_results[name]["all_labels"][best_k]

    for cl in range(best_k):
        mask = labels == cl
        ax.scatter(X[mask, 0], X[mask, 1], s=2, alpha=0.4, label=f"C{cl}")
    ax.set_title(f"{name}\nK={best_k} (best silhouette)", fontsize=10)
    ax.tick_params(labelsize=7)
    if best_k <= 15:
        ax.legend(fontsize=6, markerscale=3, ncol=2)

plt.suptitle("KMeans Clusters (2D embeddings, best K by silhouette)", fontsize=16)
plt.tight_layout()
plt.savefig(output_dir / "kmeans_scatter_2d.png", dpi=300, bbox_inches="tight")
plt.close()
print("  Saved kmeans_scatter_2d.png", flush=True)

# --- Comparison: HDBSCAN vs KMeans by continent ---
CONTINENT_COLORS = {
    "africa": "#e6194b", "asia": "#3cb44b", "europe": "#4363d8",
    "americas": "#f58231", "oceania": "#911eb4",
}

fig, axes = plt.subplots(3, len(dr_inputs_2d), figsize=(6 * len(dr_inputs_2d), 18))
for col_idx, name in enumerate(dr_inputs_2d):
    X = inputs[name]

    # Row 0: continent
    ax = axes[0, col_idx]
    for cont in sorted(metadata["continent"].dropna().unique()):
        mask = metadata["continent"] == cont
        color = CONTINENT_COLORS.get(cont, "#cccccc")
        ax.scatter(X[mask, 0], X[mask, 1], c=color, s=2, alpha=0.3, label=cont)
    ax.set_title(f"{name}\nContinent", fontsize=10)
    if col_idx == 0:
        ax.legend(fontsize=7, markerscale=3)

    # Row 1: HDBSCAN
    ax = axes[1, col_idx]
    labels = hdbscan_results[name]["labels"]
    n_cl = hdbscan_results[name]["n_clusters"]
    noise_mask = labels == -1
    ax.scatter(X[noise_mask, 0], X[noise_mask, 1], c="#cccccc", s=1, alpha=0.1)
    for cl in range(n_cl):
        mask = labels == cl
        ax.scatter(X[mask, 0], X[mask, 1], s=2, alpha=0.3, label=f"C{cl}")
    ax.set_title(f"HDBSCAN ({n_cl} clusters)", fontsize=10)

    # Row 2: KMeans best
    ax = axes[2, col_idx]
    best_k = kmeans_results[name]["best_k_sil"]
    labels = kmeans_results[name]["all_labels"][best_k]
    for cl in range(best_k):
        mask = labels == cl
        ax.scatter(X[mask, 0], X[mask, 1], s=2, alpha=0.3, label=f"C{cl}")
    ax.set_title(f"KMeans (K={best_k})", fontsize=10)

plt.suptitle("Comparison: Continent vs HDBSCAN vs KMeans", fontsize=16)
plt.tight_layout()
plt.savefig(output_dir / "comparison_continent_hdbscan_kmeans.png", dpi=300, bbox_inches="tight")
plt.close()
print("  Saved comparison_continent_hdbscan_kmeans.png", flush=True)

print("\n" + "=" * 60, flush=True)
print("All done.", flush=True)
print(f"Results saved to: {output_dir}", flush=True)
print("=" * 60, flush=True)

PYTHON_SCRIPT

echo "Clustering on Tucker factors complete."