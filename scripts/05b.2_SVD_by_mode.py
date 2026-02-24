#!/bin/bash/python3
#!/bin/bash/python3

import os
import sys
from pathlib import Path
import glob
import numpy as np
import pandas as pd
import tensorly as tl
import matplotlib.pyplot as plt

# Force unbuffered output
sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', buffering=1)
sys.stderr = os.fdopen(sys.stderr.fileno(), 'w', buffering=1)


# Create output directory if it doesn't exist
output_dir = Path('/data/users/ltucker/influenzaData/pipeline/output/rank_selection_plots')
output_dir.mkdir(parents=True, exist_ok=True)

# Path to files 
folder = Path("/data/users/ltucker/influenzaData/pipeline/output/1_distances")
files = sorted(folder.glob("*.parquet"))

# open files 
print("Opening files...", flush=True)

all_segments = []
for index, file in enumerate(files):
    #open da file 
    df = pd.read_parquet(file)
    df = df.set_index("column00000")
    all_segments.append(df)
print("Opened all files succesfully.", flush=True)

# Create tensor
print("Concatenating to tensor", flush=True)
segment_tensor = tl.tensor(np.stack(all_segments, axis=0))
print(f"Tensor created with shape: {segment_tensor.shape}")

# Unfold tensor along each mode
print("unfold tensor per mode.", flush=True)
segment_tensor_mode0 = tl.unfold(segment_tensor, mode=0)  # segments × (samples*samples)
print("segment_tensor_mode 0 is done", flush=True)
segment_tensor_mode1 = tl.unfold(segment_tensor, mode=1)  # samples × (segments*samples)
print("segment_tensor_mode 1 is done", flush=True)
segment_tensor_mode2 = tl.unfold(segment_tensor, mode=2)  # samples × (segments*samples)
print("segment_tensor_mode 2 is done", flush=True)

# Perform SVD on each unfolding
print("\nPerforming SVD on each mode...", flush=True)
U0, S0, Vt0 = np.linalg.svd(segment_tensor_mode0, full_matrices=False)
print(f"Mode 0 (segments): {len(S0)} singular values", flush=True)

U1, S1, Vt1 = np.linalg.svd(segment_tensor_mode1, full_matrices=False)
print(f"Mode 1 (samples-rows): {len(S1)} singular values", flush=True)

U2, S2, Vt2 = np.linalg.svd(segment_tensor_mode2, full_matrices=False)
print(f"Mode 2 (samples-cols): {len(S2)} singular values", flush=True)

# Calculate cumulative variance explained
def cumulative_variance(singular_values):
    variance = (singular_values ** 2) / np.sum(singular_values ** 2)
    cumulative = np.cumsum(variance)
    return variance, cumulative

var0, cum0 = cumulative_variance(S0)
var1, cum1 = cumulative_variance(S1)
var2, cum2 = cumulative_variance(S2)

# Create scree plots
fig, axes = plt.subplots(2, 3, figsize=(18, 10))

# Mode 0 (segments)
axes[0, 0].plot(range(1, len(S0) + 1), S0, 'o-')
axes[0, 0].set_xlabel('Component')
axes[0, 0].set_ylabel('Singular Value')
axes[0, 0].set_title('Mode 0: Segments - Singular Values')
axes[0, 0].grid(True)

axes[1, 0].plot(range(1, len(cum0) + 1), cum0, 'o-')
axes[1, 0].axhline(y=0.90, color='r', linestyle='--', label='90%')
axes[1, 0].axhline(y=0.95, color='orange', linestyle='--', label='95%')
axes[1, 0].set_xlabel('Number of Components')
axes[1, 0].set_ylabel('Cumulative Variance Explained')
axes[1, 0].set_title('Mode 0: Segments - Cumulative Variance')
axes[1, 0].legend()
axes[1, 0].grid(True)

# Mode 1 (samples - rows)
# Show first 50 components for readability
max_show = min(50, len(S1))
axes[0, 1].plot(range(1, max_show + 1), S1[:max_show], 'o-')
axes[0, 1].set_xlabel('Component')
axes[0, 1].set_ylabel('Singular Value')
axes[0, 1].set_title(f'Mode 1: Samples (Rows) - Singular Values (first {max_show})')
axes[0, 1].grid(True)

axes[1, 1].plot(range(1, max_show + 1), cum1[:max_show], 'o-')
axes[1, 1].axhline(y=0.90, color='r', linestyle='--', label='90%')
axes[1, 1].axhline(y=0.95, color='orange', linestyle='--', label='95%')
axes[1, 1].set_xlabel('Number of Components')
axes[1, 1].set_ylabel('Cumulative Variance Explained')
axes[1, 1].set_title(f'Mode 1: Samples (Rows) - Cumulative Variance (first {max_show})')
axes[1, 1].legend()
axes[1, 1].grid(True)

# Mode 2 (samples - cols)
axes[0, 2].plot(range(1, max_show + 1), S2[:max_show], 'o-')
axes[0, 2].set_xlabel('Component')
axes[0, 2].set_ylabel('Singular Value')
axes[0, 2].set_title(f'Mode 2: Samples (Cols) - Singular Values (first {max_show})')
axes[0, 2].grid(True)

axes[1, 2].plot(range(1, max_show + 1), cum2[:max_show], 'o-')
axes[1, 2].axhline(y=0.90, color='r', linestyle='--', label='90%')
axes[1, 2].axhline(y=0.95, color='orange', linestyle='--', label='95%')
axes[1, 2].set_xlabel('Number of Components')
axes[1, 2].set_ylabel('Cumulative Variance Explained')
axes[1, 2].set_title(f'Mode 2: Samples (Cols) - Cumulative Variance (first {max_show})')
axes[1, 2].legend()
axes[1, 2].grid(True)

plt.tight_layout()
plt.savefig(output_dir / 'tucker_scree_plots.png', dpi=300, bbox_inches='tight')
print("\nScree plots saved to 'tucker_scree_plots.png'", flush=True)

# Print recommendations
print("\n" + "="*60, flush=True)
print("RANK RECOMMENDATIONS", flush=True)
print("="*60, flush=True)

# Find ranks for 90% and 95% variance
def find_rank_for_variance(cumulative, threshold):
    return np.argmax(cumulative >= threshold) + 1

rank_90_mode0 = find_rank_for_variance(cum0, 0.90)
rank_95_mode0 = find_rank_for_variance(cum0, 0.95)
print(f"\nMode 0 (Segments - {len(S0)} total):", flush=True)
print(f"  90% variance: rank = {rank_90_mode0}", flush=True)
print(f"  95% variance: rank = {rank_95_mode0}", flush=True)

rank_90_mode1 = find_rank_for_variance(cum1, 0.90)
rank_95_mode1 = find_rank_for_variance(cum1, 0.95)
print(f"\nMode 1 (Samples-Rows - {len(S1)} total):")
print(f"  90% variance: rank = {rank_90_mode1}")
print(f"  95% variance: rank = {rank_95_mode1}")

rank_90_mode2 = find_rank_for_variance(cum2, 0.90)
rank_95_mode2 = find_rank_for_variance(cum2, 0.95)
print(f"\nMode 2 (Samples-Cols - {len(S2)} total):", flush=True)
print(f"  90% variance: rank = {rank_90_mode2}", flush=True)
print(f"  95% variance: rank = {rank_95_mode2}", flush=True)

print("\n" + "="*60, flush=True)
print("SUGGESTED RANKS TO TRY", flush=True)
print("="*60)
print(f"\nConservative (95% variance): [{rank_95_mode0}, {rank_95_mode1}, {rank_95_mode2}]", flush=True)
print(f"Moderate (90% variance):     [{rank_90_mode0}, {rank_90_mode1}, {rank_90_mode2}]", flush=True)
print(f"Aggressive (heuristic):      [{max(2, len(S0)//4)}, {max(5, len(S1)//10)}, {max(5, len(S2)//10)}]", flush=True)