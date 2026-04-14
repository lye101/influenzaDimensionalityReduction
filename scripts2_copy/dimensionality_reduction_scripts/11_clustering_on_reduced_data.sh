#!/bin/bash

#SBATCH --job-name=contracted_cluster
#SBATCH --time=1-04:00:00
#SBATCH --mem=120G
#SBATCH --cpus-per-task=4
#SBATCH --partition=pibu_el8
#SBATCH --output=/data/users/ltucker/influenzaData/H5N1_pipeline/log/tensor/contracted_cluster_%j.out
#SBATCH --error=/data/users/ltucker/influenzaData/H5N1_pipeline/log/tensor/contracted_cluster_%j.err

TENSOR_DIR="/data/users/ltucker/influenzaData/H5N1_pipeline/output/famsa_tensor_analysis/tensor"
OUTPUT_DIR="/data/users/ltucker/influenzaData/H5N1_pipeline/output/famsa_tensor_analysis/contracted_clustering"
LOG_DIR="/data/users/ltucker/influenzaData/H5N1_pipeline/log/tensor"
mkdir -p "$OUTPUT_DIR"
mkdir -p "$LOG_DIR"

CONTAINER="/data/users/ltucker/influenzaData/pipeline/jupyter-tensor_15.sif"

export TENSOR_DIR
export OUTPUT_DIR
export RANK="4-5-5"
export SEED=42

echo "Starting core tensor contraction + clustering"
echo "TENSOR_DIR: $TENSOR_DIR"
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
from sklearn.manifold import TSNE, MDS
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', buffering=1)
sys.stderr = os.fdopen(sys.stderr.fileno(), 'w', buffering=1)

try:
    from sklearn.cluster import HDBSCAN
    USE_SKLEARN_HDBSCAN = True
    print("Using sklearn.cluster.HDBSCAN", flush=True)
except ImportError:
    try:
        import hdbscan as hdbscan_lib
        USE_SKLEARN_HDBSCAN = False
        print("Using hdbscan package", flush=True)
    except ImportError:
        print("ERROR: No HDBSCAN available.", flush=True)
        sys.exit(1)

try:
    from umap import UMAP
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False
    print("UMAP not available, skipping", flush=True)

seed = int(os.environ["SEED"])
rank_str = os.environ["RANK"]
tensor_dir = Path(os.environ["TENSOR_DIR"])
output_dir = Path(os.environ["OUTPUT_DIR"])
output_dir.mkdir(parents=True, exist_ok=True)

segment_names = ["PB2", "PB1", "PA", "HA", "NP", "NA", "MP", "NS"]

# ---------------------------------------------------------------
# 1. Load Tucker components
# ---------------------------------------------------------------
print("=" * 60, flush=True)
print("Loading Tucker decomposition components...", flush=True)
print("=" * 60, flush=True)

base_name = f"tucker_full_rank{rank_str}_seed{seed}"
core    = np.load(tensor_dir / f"{base_name}_core.npy")      # (R0=4, R1=5, R2=5)
factor0 = np.load(tensor_dir / f"{base_name}_factor0.npy")   # (8, R0=4)
factor1 = np.load(tensor_dir / f"{base_name}_factor1.npy")   # (N=10009, R1=5)
factor2 = np.load(tensor_dir / f"{base_name}_factor2.npy")   # (N=10009, R2=5)

print(f"  Core:    {core.shape}    (R0 x R1 x R2)", flush=True)
print(f"  Factor0: {factor0.shape}  (segments x R0)", flush=True)
print(f"  Factor1: {factor1.shape}  (samples x R1)", flush=True)
print(f"  Factor2: {factor2.shape}  (samples x R2)", flush=True)

# Load metadata
sample_ids = pd.read_parquet(tensor_dir / "sample_ids.parquet")["sample_id"].tolist()
metadata = pd.DataFrame({"sample_id": sample_ids})
split = metadata["sample_id"].str.split("|")
metadata["country"]   = split.str[5]
metadata["continent"] = split.str[6]
metadata["year"]      = pd.to_numeric(split.str[7], errors="coerce")
print(f"  Samples: {len(metadata)}", flush=True)

# ---------------------------------------------------------------
# 2. Core tensor contractions
# ---------------------------------------------------------------
print("\n" + "=" * 60, flush=True)
print("Performing core tensor contractions...", flush=True)
print("=" * 60, flush=True)

# --- Contraction A: Factor1 x Core along R1 ---
# Each sample: 5D -> (R0 x R2) = (4 x 5) = 20D
# Encodes how each sample interacts across segment-mode components
contracted_f1 = np.einsum('nj, ijk -> nik', factor1, core)   # (N, 4, 5)
contracted_f1_flat = contracted_f1.reshape(len(factor1), -1)  # (N, 20)
print(f"  Contraction A (factor1 x core):  {factor1.shape} -> {contracted_f1_flat.shape}  (20D)", flush=True)

# --- Contraction B: Factor2 x Core along R2 ---
# Same idea for the column mode
contracted_f2 = np.einsum('nk, ijk -> nij', factor2, core)   # (N, 4, 5)
contracted_f2_flat = contracted_f2.reshape(len(factor2), -1)  # (N, 20)
print(f"  Contraction B (factor2 x core):  {factor2.shape} -> {contracted_f2_flat.shape}  (20D)", flush=True)

# --- Contraction C: Factor0 x Core, then Factor1 ---
# Fold segment loadings into core first, then contract with samples
# Result: per-sample vector that knows about specific segments
core_x_f0 = np.einsum('si, ijk -> sjk', factor0, core)       # (8, 5, 5)
contracted_f0f1 = np.einsum('nj, sjk -> nsk', factor1, core_x_f0)  # (N, 8, 5)
contracted_f0f1_flat = contracted_f0f1.reshape(len(factor1), -1)     # (N, 40)
print(f"  Contraction C (factor0 x core x factor1): {factor1.shape} -> {contracted_f0f1_flat.shape}  (40D)", flush=True)

# --- Contraction D: Factor0 x Core, then Factor2 ---
contracted_f0f2 = np.einsum('nk, sjk -> nsj', factor2, core_x_f0)  # (N, 8, 5)
contracted_f0f2_flat = contracted_f0f2.reshape(len(factor2), -1)     # (N, 40)
print(f"  Contraction D (factor0 x core x factor2): {factor2.shape} -> {contracted_f0f2_flat.shape}  (40D)", flush=True)

# --- Contraction E: All three — Factor1 x Core x Factor2 ---
# Per-sample scalar per segment-component: how much each sample
# contributes to each segment latent component
contracted_all = np.einsum('nj, ijk, nk -> ni', factor1, core, factor2)  # (N, 4)
print(f"  Contraction E (factor1 x core x factor2): -> {contracted_all.shape}  (4D)", flush=True)

# Save all contracted representations
contractions = {
    "contracted_f1_core_20d":     contracted_f1_flat,
    "contracted_f2_core_20d":     contracted_f2_flat,
    "contracted_f0f1_core_40d":   contracted_f0f1_flat,
    "contracted_f0f2_core_40d":   contracted_f0f2_flat,
    "contracted_all_4d":          contracted_all,
}

for name, arr in contractions.items():
    out_df = metadata.copy()
    for col_idx in range(arr.shape[1]):
        out_df[f"dim_{col_idx}"] = arr[:, col_idx]
    out_df.to_parquet(output_dir / f"{name}.parquet", index=False)
    print(f"  Saved {name}.parquet", flush=True)

# ---------------------------------------------------------------
# 3. Dimensionality reduction on contracted representations
# ---------------------------------------------------------------
print("\n" + "=" * 60, flush=True)
print("Dimensionality reduction on contracted vectors...", flush=True)
print("=" * 60, flush=True)

# Only run DR on the higher-D contractions (20D and 40D)
dr_inputs = {
    "contracted_f1_core_20d":   contracted_f1_flat,
    "contracted_f2_core_20d":   contracted_f2_flat,
    "contracted_f0f1_core_40d": contracted_f0f1_flat,
    "contracted_f0f2_core_40d": contracted_f0f2_flat,
}

dr_results = {}
for name, X in dr_inputs.items():
    n = X.shape[0]
    print(f"\n  {name} ({X.shape})...", flush=True)

    # t-SNE
    print(f"    t-SNE...", flush=True)
    tsne = TSNE(n_components=2, random_state=seed,
                perplexity=min(200, max(30, n // 50)),
                learning_rate=max(200, n // 12),
                max_iter=2000)
    tsne_emb = tsne.fit_transform(X)

    # UMAP
    if HAS_UMAP:
        print(f"    UMAP...", flush=True)
        umap_model = UMAP(n_components=2, random_state=seed,
                          n_neighbors=min(100, max(15, n // 100)),
                          min_dist=0.1)
        umap_emb = umap_model.fit_transform(X)
    else:
        umap_emb = np.zeros((n, 2))

    # MDS
    print(f"    MDS...", flush=True)
    mds = MDS(n_components=2, random_state=seed, n_init=4)
    mds_emb = mds.fit_transform(X)

    dr_results[name] = {"tsne": tsne_emb, "umap": umap_emb, "mds": mds_emb}

    # Save combined DR
    out_df = metadata.copy()
    out_df["tsne_1"] = tsne_emb[:, 0]
    out_df["tsne_2"] = tsne_emb[:, 1]
    out_df["umap_1"] = umap_emb[:, 0]
    out_df["umap_2"] = umap_emb[:, 1]
    out_df["mds_1"]  = mds_emb[:, 0]
    out_df["mds_2"]  = mds_emb[:, 1]
    out_df.to_parquet(output_dir / f"dr_{name}.parquet", index=False)
    print(f"    Saved dr_{name}.parquet", flush=True)

# ---------------------------------------------------------------
# 4. HDBSCAN clustering
# ---------------------------------------------------------------
print("\n" + "=" * 60, flush=True)
print("HDBSCAN clustering (min_cluster_size=50, min_samples=30)", flush=True)
print("=" * 60, flush=True)

hdbscan_results = {}
for name, X in contractions.items():
    print(f"\n  {name} ({X.shape})...", flush=True)

    if USE_SKLEARN_HDBSCAN:
        clusterer = HDBSCAN(min_cluster_size=50, min_samples=30, n_jobs=4)
    else:
        clusterer = hdbscan_lib.HDBSCAN(min_cluster_size=50, min_samples=30, core_dist_n_jobs=4)

    labels = clusterer.fit_predict(X)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = (labels == -1).sum()
    pct_noise = 100 * n_noise / len(labels)

    mask = labels >= 0
    if len(set(labels[mask])) >= 2:
        sil = silhouette_score(X[mask], labels[mask])
    else:
        sil = np.nan

    print(f"    Clusters: {n_clusters}, Noise: {n_noise} ({pct_noise:.1f}%), Silhouette: {sil:.3f}" if not np.isnan(sil) else f"    Clusters: {n_clusters}, Noise: {n_noise} ({pct_noise:.1f}%)", flush=True)

    hdbscan_results[name] = {
        "labels": labels, "n_clusters": n_clusters,
        "n_noise": n_noise, "pct_noise": pct_noise, "silhouette": sil,
    }

    out_df = metadata.copy()
    out_df["hdbscan_cluster"] = labels
    out_df.to_parquet(output_dir / f"hdbscan_{name}.parquet", index=False)

# ---------------------------------------------------------------
# 5. KMeans clustering (K=2..30)
# ---------------------------------------------------------------
print("\n" + "=" * 60, flush=True)
print("KMeans clustering (K=2..30)", flush=True)
print("=" * 60, flush=True)

k_range = range(2, 31)
kmeans_results = {}

for name, X in contractions.items():
    print(f"\n  {name} ({X.shape})...", flush=True)

    scores = []
    all_labels = {}

    for k in k_range:
        km = KMeans(n_clusters=k, random_state=seed, n_init=10, max_iter=300)
        labels = km.fit_predict(X)
        all_labels[k] = labels

        sil = silhouette_score(X, labels)
        ch = calinski_harabasz_score(X, labels)
        db = davies_bouldin_score(X, labels)
        scores.append({"k": k, "silhouette": sil, "calinski_harabasz": ch,
                        "davies_bouldin": db, "inertia": km.inertia_})

    scores_df = pd.DataFrame(scores)
    best_k_sil = int(scores_df.loc[scores_df["silhouette"].idxmax(), "k"])
    best_k_db = int(scores_df.loc[scores_df["davies_bouldin"].idxmin(), "k"])

    print(f"    Best K (silhouette): {best_k_sil} ({scores_df['silhouette'].max():.3f})", flush=True)
    print(f"    Best K (Davies-Bouldin): {best_k_db} ({scores_df['davies_bouldin'].min():.3f})", flush=True)

    kmeans_results[name] = {
        "scores": scores_df, "all_labels": all_labels,
        "best_k_sil": best_k_sil, "best_k_db": best_k_db,
    }

    scores_df.to_parquet(output_dir / f"kmeans_scores_{name}.parquet", index=False)

    # Save best labels
    for best_k, metric in [(best_k_sil, "sil"), (best_k_db, "db")]:
        out_df = metadata.copy()
        out_df[f"kmeans_k{best_k}_cluster"] = all_labels[best_k]
        out_df.to_parquet(output_dir / f"kmeans_best_{metric}_{name}_k{best_k}.parquet", index=False)

    # Save all K labels
    out_df = metadata.copy()
    for k in k_range:
        out_df[f"kmeans_k{k}"] = all_labels[k]
    out_df.to_parquet(output_dir / f"kmeans_all_k_{name}.parquet", index=False)

# ---------------------------------------------------------------
# 6. Summary table
# ---------------------------------------------------------------
print("\n" + "=" * 60, flush=True)
print("Summary", flush=True)
print("=" * 60, flush=True)

summary_rows = []
for name in contractions:
    h = hdbscan_results[name]
    km = kmeans_results[name]
    summary_rows.append({
        "input": name,
        "dims": contractions[name].shape[1],
        "hdbscan_clusters": h["n_clusters"],
        "hdbscan_noise_pct": round(h["pct_noise"], 1),
        "hdbscan_silhouette": round(h["silhouette"], 3) if not np.isnan(h["silhouette"]) else "N/A",
        "kmeans_best_k_sil": km["best_k_sil"],
        "kmeans_best_sil": round(km["scores"]["silhouette"].max(), 3),
        "kmeans_best_k_db": km["best_k_db"],
        "kmeans_best_db": round(km["scores"]["davies_bouldin"].min(), 3),
    })

summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv(output_dir / "contracted_clustering_summary.csv", index=False)
print(summary_df.to_string(index=False), flush=True)

# ---------------------------------------------------------------
# 7. Plots
# ---------------------------------------------------------------
print("\nCreating plots...", flush=True)

# --- KMeans metrics ---
fig, axes = plt.subplots(len(contractions), 3, figsize=(18, 4 * len(contractions)))
for row, name in enumerate(contractions):
    scores_df = kmeans_results[name]["scores"]
    best_k = kmeans_results[name]["best_k_sil"]
    best_k_db = kmeans_results[name]["best_k_db"]

    ax = axes[row, 0]
    ax.plot(scores_df["k"], scores_df["inertia"], "o-", markersize=3)
    ax.set_title(f"{name}", fontsize=10)
    ax.set_xlabel("K"); ax.set_ylabel("Inertia"); ax.grid(True, alpha=0.3)
    ax.axvline(best_k, color="red", linestyle="--", alpha=0.7, label=f"Best K={best_k}")
    ax.legend(fontsize=8)

    ax = axes[row, 1]
    ax.plot(scores_df["k"], scores_df["silhouette"], "o-", markersize=3, color="green")
    ax.set_xlabel("K"); ax.set_ylabel("Silhouette"); ax.grid(True, alpha=0.3)
    ax.axvline(best_k, color="red", linestyle="--", alpha=0.7)

    ax = axes[row, 2]
    ax.plot(scores_df["k"], scores_df["davies_bouldin"], "o-", markersize=3, color="purple")
    ax.set_xlabel("K"); ax.set_ylabel("Davies-Bouldin"); ax.grid(True, alpha=0.3)
    ax.axvline(best_k_db, color="red", linestyle="--", alpha=0.7, label=f"Best K={best_k_db}")
    ax.legend(fontsize=8)

plt.suptitle("KMeans Metrics — Contracted Representations", fontsize=16, y=1.01)
plt.tight_layout()
plt.savefig(output_dir / "kmeans_metrics_contracted.png", dpi=300, bbox_inches="tight")
plt.close()
print("  Saved kmeans_metrics_contracted.png", flush=True)

# --- DR scatter plots coloured by HDBSCAN and KMeans (20D contractions) ---
CONTINENT_COLORS = {
    "africa": "#e6194b", "asia": "#3cb44b", "europe": "#4363d8",
    "americas": "#f58231", "oceania": "#911eb4",
}

for dr_name in dr_results:
    dr = dr_results[dr_name]
    h_labels = hdbscan_results[dr_name]["labels"]
    km_best_k = kmeans_results[dr_name]["best_k_sil"]
    km_labels = kmeans_results[dr_name]["all_labels"][km_best_k]

    fig, axes = plt.subplots(4, 3, figsize=(18, 24))

    methods = [("tsne", "t-SNE"), ("umap", "UMAP"), ("mds", "MDS")]

    for col, (mkey, mlabel) in enumerate(methods):
        emb = dr[mkey]

        # Row 0: continent
        ax = axes[0, col]
        for cont in sorted(metadata["continent"].dropna().unique()):
            mask = metadata["continent"] == cont
            ax.scatter(emb[mask, 0], emb[mask, 1], c=CONTINENT_COLORS.get(cont, "#ccc"),
                       s=2, alpha=0.3, label=cont)
        ax.set_title(f"{mlabel} — Continent", fontsize=11)
        if col == 0:
            ax.legend(fontsize=7, markerscale=3)

        # Row 1: year
        ax = axes[1, col]
        years = metadata["year"].values
        valid = ~np.isnan(years)
        sc = ax.scatter(emb[valid, 0], emb[valid, 1], c=years[valid],
                        cmap="viridis", s=2, alpha=0.3)
        ax.set_title(f"{mlabel} — Year", fontsize=11)

        # Row 2: HDBSCAN
        ax = axes[2, col]
        cmap_tab = plt.get_cmap("tab20")
        noise_mask = h_labels == -1
        ax.scatter(emb[noise_mask, 0], emb[noise_mask, 1], c="#cccccc", s=1, alpha=0.1)
        n_hcl = hdbscan_results[dr_name]["n_clusters"]
        for cl in range(n_hcl):
            mask = h_labels == cl
            ax.scatter(emb[mask, 0], emb[mask, 1], c=[cmap_tab(cl % 20)],
                       s=2, alpha=0.3, label=f"C{cl}")
        ax.set_title(f"{mlabel} — HDBSCAN ({n_hcl} cl.)", fontsize=11)

        # Row 3: KMeans
        ax = axes[3, col]
        for cl in range(km_best_k):
            mask = km_labels == cl
            ax.scatter(emb[mask, 0], emb[mask, 1], c=[cmap_tab(cl % 20)],
                       s=2, alpha=0.3, label=f"C{cl}")
        ax.set_title(f"{mlabel} — KMeans (K={km_best_k})", fontsize=11)

    plt.suptitle(f"{dr_name}\nContinent / Year / HDBSCAN / KMeans", fontsize=14)
    plt.tight_layout()
    plt.savefig(output_dir / f"dr_grid_{dr_name}.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved dr_grid_{dr_name}.png", flush=True)

print("\n" + "=" * 60, flush=True)
print("All done.", flush=True)
print(f"Results saved to: {output_dir}", flush=True)
print("=" * 60, flush=True)

PYTHON_SCRIPT

echo "Contracted clustering complete."