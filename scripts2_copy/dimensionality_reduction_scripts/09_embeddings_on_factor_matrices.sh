#!/bin/bash

#SBATCH --job-name=tucker_dr
#SBATCH --time=04:00:00
#SBATCH --mem=100G
#SBATCH --cpus-per-task=4
#SBATCH --partition=pibu_el8
#SBATCH --output=/data/users/ltucker/influenzaData/H5N1_pipeline/log/tensor/tucker_dr_%j.out
#SBATCH --error=/data/users/ltucker/influenzaData/H5N1_pipeline/log/tensor/tucker_dr_%j.err

INPUT_DIR="/data/users/ltucker/influenzaData/H5N1_pipeline/output/famsa_tensor_analysis/tensor"
OUTPUT_DIR="/data/users/ltucker/influenzaData/H5N1_pipeline/output/famsa_tensor_analysis/dimensionality_reduction"
mkdir -p "$OUTPUT_DIR"
mkdir -p "/data/users/ltucker/influenzaData/H5N1_pipeline/log/tensor"

CONTAINER="/data/users/ltucker/influenzaData/pipeline/jupyter-tensor_15.sif"

# Must match the rank and seed used in 08_tucker_decomposition.py
export RANK="4-5-5"
export SEED=42
export INPUT_DIR
export OUTPUT_DIR

echo "Starting dimensionality reduction on Tucker factor matrices"
echo "Using: tucker_full_rank${RANK}_seed${SEED}"

apptainer exec "$CONTAINER" python3 << 'PYTHON_SCRIPT'

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE, MDS
from umap import UMAP
from pathlib import Path

sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', buffering=1)
sys.stderr = os.fdopen(sys.stderr.fileno(), 'w', buffering=1)

seed = int(os.environ["SEED"])
rank_str = os.environ["RANK"]
input_dir = Path(os.environ["INPUT_DIR"])
output_dir = Path(os.environ["OUTPUT_DIR"])

base_name = f"tucker_full_rank{rank_str}_seed{seed}"
output_dir.mkdir(parents=True, exist_ok=True)

segment_names = ["PB2", "PB1", "PA", "HA", "NP", "NA", "MP", "NS"]

# ---------------------------------------------------------------
# 1. Load Tucker factor matrices
# ---------------------------------------------------------------
print("Loading Tucker factor matrices...", flush=True)
factor0 = np.load(input_dir / f"{base_name}_factor0.npy")
factor1 = np.load(input_dir / f"{base_name}_factor1.npy")
factor2 = np.load(input_dir / f"{base_name}_factor2.npy")
core    = np.load(input_dir / f"{base_name}_core.npy")

print(f"  Factor 0 — Segments:     {factor0.shape}  (8 segments x R0 components)", flush=True)
print(f"  Factor 1 — Samples_row:  {factor1.shape}  (N samples x R1 components)", flush=True)
print(f"  Factor 2 — Samples_col:  {factor2.shape}  (N samples x R2 components)", flush=True)
print(f"  Core tensor:             {core.shape}", flush=True)

# Load sample IDs for metadata
sample_ids = pd.read_parquet(input_dir / "sample_ids.parquet")["sample_id"].tolist()

# Parse metadata
metadata = pd.DataFrame({"sample_id": sample_ids})
split = metadata["sample_id"].str.split("|")
metadata["country"]   = split.str[5]
metadata["continent"] = split.str[6]
metadata["year"]      = pd.to_numeric(split.str[7], errors="coerce")
print(f"\n  Samples: {len(metadata)}, Countries: {metadata['country'].nunique()}, "
      f"Continents: {metadata['continent'].nunique()}, "
      f"Years: {int(metadata['year'].min())}-{int(metadata['year'].max())}", flush=True)

# ---------------------------------------------------------------
# 2. DR on Factor 1 (samples_row) and Factor 2 (samples_col)
# ---------------------------------------------------------------
factors_to_reduce = {
    "factor1_samples_row": factor1,
    "factor2_samples_col": factor2,
}

def make_tsne(n):
    return TSNE(n_components=2, random_state=seed,
                perplexity=min(200, max(30, n // 50)),
                learning_rate=max(200, n // 12),
                max_iter=2000)

def make_umap(n):
    return UMAP(n_components=2, random_state=seed,
                n_neighbors=min(100, max(15, n // 100)),
                min_dist=0.1)

def make_mds(n):
    return MDS(n_components=2, random_state=seed, n_init=4)

methods = {
    "tsne": make_tsne,
    "umap": make_umap,
    "mds":  make_mds,
}

results = {}

for factor_label, factor_matrix in factors_to_reduce.items():
    n = factor_matrix.shape[0]
    print(f"\n{'='*60}", flush=True)
    print(f"DR on {factor_label}  (shape: {factor_matrix.shape})", flush=True)
    print(f"{'='*60}", flush=True)

    for method_name, method_fn in methods.items():
        print(f"  Running {method_name.upper()}...", flush=True)
        model = method_fn(n)
        emb = model.fit_transform(factor_matrix)
        results[f"{factor_label}_{method_name}"] = emb
        print(f"  {method_name.upper()} done — shape: {emb.shape}", flush=True)

        # Save individual embedding with metadata
        emb_df = metadata.copy()
        emb_df[f"{method_name}_1"] = emb[:, 0]
        emb_df[f"{method_name}_2"] = emb[:, 1]
        out_file = output_dir / f"dr_{method_name}_{factor_label}.parquet"
        emb_df.to_parquet(out_file, index=False)

    # Save combined parquet per factor (all methods in one file)
    combined = metadata.copy()
    for method_name in methods:
        key = f"{factor_label}_{method_name}"
        combined[f"{method_name}_1"] = results[key][:, 0]
        combined[f"{method_name}_2"] = results[key][:, 1]
    combined.to_parquet(output_dir / f"dr_all_{factor_label}.parquet", index=False)
    print(f"  Saved combined: dr_all_{factor_label}.parquet", flush=True)

# ---------------------------------------------------------------
# 3. Visualisation grids
# ---------------------------------------------------------------
print("\nCreating visualisation grids...", flush=True)

method_list = ["tsne", "umap", "mds"]
method_labels = ["t-SNE", "UMAP", "MDS"]
factor_list = ["factor1_samples_row", "factor2_samples_col"]
factor_labels_short = ["Factor 1 (samples x [seg * samples])", "Factor 2 (samples x [seg * samples])"]

# --- Grid colored by year ---
fig, axes = plt.subplots(len(factor_list), len(method_list),
                         figsize=(6 * len(method_list), 6 * len(factor_list)))

for row, (flabel, fshort) in enumerate(zip(factor_list, factor_labels_short)):
    for col, (mname, mlabel) in enumerate(zip(method_list, method_labels)):
        ax = axes[row, col]
        key = f"{flabel}_{mname}"
        emb = results[key]
        sc = ax.scatter(emb[:, 0], emb[:, 1],
                        c=metadata["year"], cmap="viridis",
                        s=3, alpha=0.4, rasterized=True)
        if row == 0:
            ax.set_title(mlabel, fontsize=13)
        if col == 0:
            ax.set_ylabel(fshort, fontsize=11)
        ax.set_xticks([])
        ax.set_yticks([])

fig.colorbar(sc, ax=axes, shrink=0.5, label="Year")
plt.suptitle("Tucker Factor DR — colored by year", fontsize=16, y=1.00)
plt.tight_layout()
plt.savefig(output_dir / "dr_grid_by_year.png", dpi=300, bbox_inches="tight")
plt.close()
print("  Saved dr_grid_by_year.png", flush=True)

# --- Grid colored by continent ---
continents = sorted(metadata["continent"].dropna().unique())
cmap_cont = plt.get_cmap("tab10")
color_map = {c: cmap_cont(i / max(len(continents) - 1, 1)) for i, c in enumerate(continents)}
colors = metadata["continent"].map(color_map)

fig, axes = plt.subplots(len(factor_list), len(method_list),
                         figsize=(6 * len(method_list), 6 * len(factor_list)))

for row, (flabel, fshort) in enumerate(zip(factor_list, factor_labels_short)):
    for col, (mname, mlabel) in enumerate(zip(method_list, method_labels)):
        ax = axes[row, col]
        key = f"{flabel}_{mname}"
        emb = results[key]
        ax.scatter(emb[:, 0], emb[:, 1],
                   c=colors, s=3, alpha=0.4, rasterized=True)
        if row == 0:
            ax.set_title(mlabel, fontsize=13)
        if col == 0:
            ax.set_ylabel(fshort, fontsize=11)
        ax.set_xticks([])
        ax.set_yticks([])

from matplotlib.lines import Line2D
handles = [Line2D([0], [0], marker="o", color="w", markerfacecolor=color_map[c],
                  markersize=8, label=c) for c in continents]
fig.legend(handles=handles, loc="center right", title="Continent",
           bbox_to_anchor=(1.12, 0.5), fontsize=9)
plt.suptitle("Tucker Factor DR — colored by continent", fontsize=16, y=1.00)
plt.tight_layout()
plt.savefig(output_dir / "dr_grid_by_continent.png", dpi=300, bbox_inches="tight")
plt.close()
print("  Saved dr_grid_by_continent.png", flush=True)

# --- Factor 0 bar chart (segments — only 8 rows, no DR needed) ---
n_components = factor0.shape[1]
fig, axes = plt.subplots(1, n_components, figsize=(5 * n_components, 4))
if n_components == 1:
    axes = [axes]

for comp in range(n_components):
    ax = axes[comp]
    vals = factor0[:, comp]
    colors_bar = ["salmon" if v > 0 else "skyblue" for v in vals]
    ax.bar(segment_names, vals, color=colors_bar, alpha=0.9)
    ax.axhline(y=0, color="black", linewidth=0.5)
    ax.set_ylabel("Loading")
    ax.set_title(f"Factor 0 — Component {comp+1}")
    ax.grid(True, alpha=0.3, axis="y")
    ax.tick_params(axis="x", rotation=45)

plt.suptitle("Tucker Factor 0: Segment Loadings", fontsize=14)
plt.tight_layout()
plt.savefig(output_dir / "factor0_segment_loadings.png", dpi=300, bbox_inches="tight")
plt.close()
print("  Saved factor0_segment_loadings.png", flush=True)

print("\nAll done.", flush=True)

PYTHON_SCRIPT

echo "Dimensionality reduction on Tucker factors complete."