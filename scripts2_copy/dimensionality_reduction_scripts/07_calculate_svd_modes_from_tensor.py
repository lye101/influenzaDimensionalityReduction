#!/bin/bash
#SBATCH --job-name=svd_tensor
#SBATCH --time=06:00:00
#SBATCH --mem=200G
#SBATCH --cpus-per-task=4
#SBATCH --partition=pibu_el8
#SBATCH --output=/data/users/ltucker/influenzaData/H5N1_pipeline/log/svd/svd_tensor_%j.out
#SBATCH --error=/data/users/ltucker/influenzaData/H5N1_pipeline/log/svd/svd_tensor_%j.err

CONTAINER="/data/users/ltucker/influenzaData/pipeline/jupyter-tensor_15.sif"

OUTPUT_DIR="/data/users/ltucker/influenzaData/H5N1_pipeline/output/famsa_tensor_analysis/svd"
mkdir -p "$OUTPUT_DIR"

echo "Starting SVD tensor analysis"

apptainer exec "$CONTAINER" python3 << 'PYTHON_SCRIPT'

import os
import sys
import numpy as np
import pandas as pd
import tensorly as tl
from pathlib import Path

# Force unbuffered output
sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', buffering=1)
sys.stderr = os.fdopen(sys.stderr.fileno(), 'w', buffering=1)

input_dir = Path("/data/users/ltucker/influenzaData/H5N1_pipeline/output/famsa_distances")
output_dir = Path("/data/users/ltucker/influenzaData/H5N1_pipeline/output/famsa_tensor_analysis/svd")
output_dir.mkdir(parents=True, exist_ok=True)

segment_names = ["PB2", "PB1", "PA", "HA", "NP", "NA", "MP", "NS"]

# Load all 8 symmetric distance matrices
print("Loading distance matrices...", flush=True)
all_segments = []
for i in range(1, 9):
    f = input_dir / f"symmetric_distances_{i}.parquet"
    df = pd.read_parquet(f)
    print(f"  Loaded {f.name} — shape {df.shape}", flush=True)
    all_segments.append(df.values)

# Store sample IDs from the first segment (assumed consistent across segments)
sample_ids = pd.read_parquet(input_dir / "symmetric_distances_1.parquet").index.tolist()

# Create tensor: (segments, samples, samples)
print("Stacking into tensor...", flush=True)
segment_tensor = tl.tensor(np.stack(all_segments, axis=0))
print(f"Tensor shape: {segment_tensor.shape}", flush=True)

# Unfold along each mode
print("Unfolding tensor per mode...", flush=True)
segment_tensor_mode0 = tl.unfold(segment_tensor, mode=0)  # segments × (samples*samples)
print(f"  Mode 0 unfolded: {segment_tensor_mode0.shape}", flush=True)

segment_tensor_mode1 = tl.unfold(segment_tensor, mode=1)  # samples × (segments*samples)
print(f"  Mode 1 unfolded: {segment_tensor_mode1.shape}", flush=True)

segment_tensor_mode2 = tl.unfold(segment_tensor, mode=2)  # samples × (segments*samples)
print(f"  Mode 2 unfolded: {segment_tensor_mode2.shape}", flush=True)

# SVD on each unfolding
print("\nPerforming SVD on each mode...", flush=True)

U0, S0, Vt0 = np.linalg.svd(segment_tensor_mode0, full_matrices=False)
print(f"  Mode 0 (segments): {len(S0)} singular values", flush=True)

U1, S1, Vt1 = np.linalg.svd(segment_tensor_mode1, full_matrices=False)
print(f"  Mode 1 (samples-rows): {len(S1)} singular values", flush=True)

U2, S2, Vt2 = np.linalg.svd(segment_tensor_mode2, full_matrices=False)
print(f"  Mode 2 (samples-cols): {len(S2)} singular values", flush=True)

# Cumulative variance
def cumulative_variance(singular_values):
    variance = (singular_values ** 2) / np.sum(singular_values ** 2)
    cumulative = np.cumsum(variance)
    return variance, cumulative

var0, cum0 = cumulative_variance(S0)
var1, cum1 = cumulative_variance(S1)
var2, cum2 = cumulative_variance(S2)

# --- Save U, S, Vt for each mode ---
print("\nSaving SVD results...", flush=True)

for mode, (U, S, Vt, var, cum) in enumerate(
    [(U0, S0, Vt0, var0, cum0),
     (U1, S1, Vt1, var1, cum1),
     (U2, S2, Vt2, var2, cum2)]
):
    np.save(output_dir / f"U_mode{mode}.npy", U)
    np.save(output_dir / f"S_mode{mode}.npy", S)
    np.save(output_dir / f"Vt_mode{mode}.npy", Vt)

    # Save variance info as a DataFrame
    var_df = pd.DataFrame({
        "component": np.arange(1, len(S) + 1),
        "singular_value": S,
        "variance_explained": var,
        "cumulative_variance": cum
    })
    var_df.to_parquet(output_dir / f"variance_mode{mode}.parquet", index=False)
    print(f"  Mode {mode}: U{U.shape}, S({len(S)},), Vt{Vt.shape} — saved", flush=True)

# Save sample IDs for reference
pd.Series(sample_ids, name="sample_id").to_frame().to_parquet(
    output_dir / "sample_ids.parquet", index=False
)

# Save segment names for mode 0 reference
pd.Series(segment_names, name="segment").to_frame().to_parquet(
    output_dir / "segment_names.parquet", index=False
)

print("\nAll SVD outputs saved to:", output_dir, flush=True)
print("Done.", flush=True)

PYTHON_SCRIPT

echo "SVD tensor job complete."