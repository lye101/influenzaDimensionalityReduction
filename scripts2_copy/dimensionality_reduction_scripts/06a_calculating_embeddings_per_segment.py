#!/bin/bash
#SBATCH --job-name=embeddings_segments
#SBATCH --array=0-7
#SBATCH --time=06:00:00
#SBATCH --mem=200G
#SBATCH --cpus-per-task=4
#SBATCH --partition=pibu_el8
#SBATCH --output=/data/users/ltucker/influenzaData/H5N1_pipeline/log/embeddings_%A_%a.out
#SBATCH --error=/data/users/ltucker/influenzaData/H5N1_pipeline/log/embeddings_%A_%a.err

SEGMENT_IDX=$SLURM_ARRAY_TASK_ID
SEGMENT_NAMES=("PB2" "PB1" "PA" "HA" "NP" "NA" "MP" "NS")
SEG_NAME=${SEGMENT_NAMES[$SEGMENT_IDX]}

INPUT_DIR="/data/users/ltucker/influenzaData/H5N1_pipeline/output/famsa_distances"
INPUT_FILE="${INPUT_DIR}/symmetric_distances_$((SEGMENT_IDX + 1)).parquet"
OUTPUT_DIR="/data/users/ltucker/influenzaData/H5N1_pipeline/output/famsa_per_segment_analysis/embeddings"
mkdir -p "$OUTPUT_DIR"

CONTAINER="/data/users/ltucker/influenzaData/pipeline/jupyter-tensor_15.sif"

echo "Processing segment ${SEG_NAME} (index ${SEGMENT_IDX})"
echo "Input: ${INPUT_FILE}"
echo "Output dir: ${OUTPUT_DIR}"

apptainer exec "$CONTAINER" python3 << 'PYTHON_SCRIPT'

import numpy as np
import pandas as pd
import umap
import phate
from sklearn.manifold import TSNE, MDS
import os

seed = 42
segment_idx = int(os.environ["SLURM_ARRAY_TASK_ID"])
segment_names = ["PB2", "PB1", "PA", "HA", "NP", "NA", "MP", "NS"]
seg_name = segment_names[segment_idx]

input_file = f"/data/users/ltucker/influenzaData/H5N1_pipeline/output/famsa_distances/symmetric_distances_{segment_idx + 1}.parquet"
output_dir = "/data/users/ltucker/influenzaData/H5N1_pipeline/output/famsa_per_segment_analysis/embeddings"

print(f"Loading distance matrix from {input_file}")
df = pd.read_parquet(input_file)
dist_matrix = df.values
sample_ids = df.index.tolist()
print(f"Distance matrix shape: {dist_matrix.shape}")

# Ensure symmetry and zero diagonal
dist_matrix = (dist_matrix + dist_matrix.T) / 2.0
np.fill_diagonal(dist_matrix, 0.0)

results = pd.DataFrame({"sample_id": sample_ids})

# --- t-SNE ---
print(f"[{seg_name}] Running t-SNE...")
tsne = TSNE(n_components=2, metric="precomputed", init="random", random_state=seed)
tsne_emb = tsne.fit_transform(dist_matrix)
results["tsne_1"] = tsne_emb[:, 0]
results["tsne_2"] = tsne_emb[:, 1]
print(f"[{seg_name}] t-SNE done.")

# --- UMAP ---
print(f"[{seg_name}] Running UMAP...")
umap_model = umap.UMAP(n_components=2, metric="precomputed", random_state=seed)
umap_emb = umap_model.fit_transform(dist_matrix)
results["umap_1"] = umap_emb[:, 0]
results["umap_2"] = umap_emb[:, 1]
print(f"[{seg_name}] UMAP done.")

# --- MDS ---
print(f"[{seg_name}] Running MDS...")
mds = MDS(n_components=2, dissimilarity="precomputed", random_state=seed, n_init=4)
mds_emb = mds.fit_transform(dist_matrix)
results["mds_1"] = mds_emb[:, 0]
results["mds_2"] = mds_emb[:, 1]
print(f"[{seg_name}] MDS done.")

# --- PHATE ---
print(f"[{seg_name}] Running PHATE...")
phate_op = phate.PHATE(
    n_components=2,
    knn=15,
    decay=40,
    t="auto",
    n_landmark=2000,
    random_state=seed
)
phate_emb = phate_op.fit_transform(dist_matrix)
results["phate_1"] = phate_emb[:, 0]
results["phate_2"] = phate_emb[:, 1]
print(f"[{seg_name}] PHATE done.")

# --- Save ---
out_path = os.path.join(output_dir, f"embeddings_{seg_name}.parquet")
results.to_parquet(out_path, index=False)
print(f"[{seg_name}] Saved embeddings to {out_path}")
print(f"[{seg_name}] Output shape: {results.shape}")

PYTHON_SCRIPT

echo "Segment ${SEG_NAME} complete."