import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import tensorly as tl
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import argparse

def load_tensor_with_indices(indices_file, parquet_folder):
    """Load full tensor and subsample using indices"""
    indices_path = Path(indices_file)
    selected_indices = np.load(indices_path)
    
    files = sorted(Path(parquet_folder).glob("*.parquet"))
    all_segments = []
    for file in files:
        df = pd.read_parquet(file)
        df = df.set_index("column00000")
        all_segments.append(df.values)
    
    full_tensor = tl.tensor(np.stack(all_segments, axis=0))
    subsampled_tensor = full_tensor[:, selected_indices, :][:, :, selected_indices]
    
    return subsampled_tensor, selected_indices

def cumulative_variance(singular_values):
    """Calculate variance and cumulative variance from singular values"""
    variance = (singular_values ** 2) / np.sum(singular_values ** 2)
    cumulative = np.cumsum(variance)
    return variance, cumulative

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--percentage', type=int, required=True, help='Subsample percentage (5, 10, 20, etc.)')
    parser.add_argument('--iteration', type=int, required=True, help='Iteration number (0-indexed)')
    parser.add_argument('--seed', type=int, required=True, help='Random seed')
    args = parser.parse_args()
    
    # Paths
    indices_folder = Path("/data/users/ltucker/influenzaData/pipeline/output/tensor/subsample_indices")
    parquet_folder = Path("/data/users/ltucker/influenzaData/pipeline/output/1_distances")
    output_folder = Path("/data/users/ltucker/influenzaData/pipeline/output/tensor/svd_scree_plots")
    output_folder.mkdir(parents=True, exist_ok=True)
    
    # Load tensor
    indices_file = indices_folder / f"indices_pct{args.percentage:03d}_iter{args.iteration:02d}_seed{args.seed:04d}.npy"
    print(f"Loading indices from: {indices_file}")
    tensor, selected_indices = load_tensor_with_indices(indices_file, parquet_folder)
    print(f"Tensor shape: {tensor.shape}")
    
    print("\n" + "="*60)
    print(f"Processing {args.percentage}% subsample (iter {args.iteration}, seed {args.seed})")
    print("="*60)
    
    print("\nUnfolding tensor along each mode...")
    # Unfold tensor along each mode
    tensor_mode1 = tl.unfold(tensor, mode=0)  # I × (J*K)
    print(f"segment_tensor_mode1 is done - shape: {tensor_mode1.shape}")
    
    tensor_mode2 = tl.unfold(tensor, mode=1)  # J × (I*K)
    print(f"segment_tensor_mode2 is done - shape: {tensor_mode2.shape}")
    
    tensor_mode3 = tl.unfold(tensor, mode=2)  # K × (I*J)
    print(f"segment_tensor_mode3 is done - shape: {tensor_mode3.shape}")
    
    print("\nPerforming SVD on each unfolding...")
    # SVD on each unfolding
    U1, W1, V1 = np.linalg.svd(tensor_mode1, full_matrices=False)
    print(f"mode1 svd is done - {len(W1)} singular values")
    
    U2, W2, V2 = np.linalg.svd(tensor_mode2, full_matrices=False)
    print(f"mode2 svd is done - {len(W2)} singular values")
    
    U3, W3, V3 = np.linalg.svd(tensor_mode3, full_matrices=False)
    print(f"mode3 svd is done - {len(W3)} singular values")
    
    # Calculate cumulative variance
    var1, cum1 = cumulative_variance(W1)
    var2, cum2 = cumulative_variance(W2)
    var3, cum3 = cumulative_variance(W3)
    
    # Print values to log
    print("\nVariance explained by mode:")
    print(f"Mode 0 variance: {var1}")
    print(f"Mode 0 cumulative: {cum1}")
    print(f"\nMode 1 variance (first 10): {var2[:10]}")
    print(f"Mode 1 cumulative (first 10): {cum2[:10]}")
    print(f"\nMode 2 variance (first 10): {var3[:10]}")
    print(f"Mode 2 cumulative (first 10): {cum3[:10]}")
    
    # Print variance thresholds
    print(f"\nMode 0 - 90% variance at component: {np.argmax(cum1 >= 0.90) + 1}")
    print(f"Mode 0 - 95% variance at component: {np.argmax(cum1 >= 0.95) + 1}")
    print(f"Mode 1 - 90% variance at component: {np.argmax(cum2 >= 0.90) + 1}")
    print(f"Mode 1 - 95% variance at component: {np.argmax(cum2 >= 0.95) + 1}")
    print(f"Mode 2 - 90% variance at component: {np.argmax(cum3 >= 0.90) + 1}")
    print(f"Mode 2 - 95% variance at component: {np.argmax(cum3 >= 0.95) + 1}")
    
    # Save singular values and variance to CSV
    max_len = max(len(W1), len(W2), len(W3))
    
    sv_df = pd.DataFrame({
        'component': range(1, max_len + 1),
        'mode0_singular_value': np.pad(W1, (0, max_len - len(W1)), constant_values=np.nan),
        'mode0_variance': np.pad(var1, (0, max_len - len(var1)), constant_values=np.nan),
        'mode0_cumulative_variance': np.pad(cum1, (0, max_len - len(cum1)), constant_values=np.nan),
        'mode1_singular_value': np.pad(W2, (0, max_len - len(W2)), constant_values=np.nan),
        'mode1_variance': np.pad(var2, (0, max_len - len(var2)), constant_values=np.nan),
        'mode1_cumulative_variance': np.pad(cum2, (0, max_len - len(cum2)), constant_values=np.nan),
        'mode2_singular_value': np.pad(W3, (0, max_len - len(W3)), constant_values=np.nan),
        'mode2_variance': np.pad(var3, (0, max_len - len(var3)), constant_values=np.nan),
        'mode2_cumulative_variance': np.pad(cum3, (0, max_len - len(cum3)), constant_values=np.nan),
    })
    
    csv_filename = f"svd_values_pct{args.percentage:03d}_iter{args.iteration:02d}_seed{args.seed:04d}.csv"
    sv_df.to_csv(output_folder / csv_filename, index=False)
    print(f"\nSaved SVD values to: {csv_filename}")
    
    # Create scree plots
    print(f"\nFor random state : {args.seed}")
    print("Creating scree plots...")
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    # Mode 0 (segments)
    axes[0, 0].plot(range(1, len(W1) + 1), W1, 'o-', markersize=4)
    axes[0, 0].set_xlabel('Component')
    axes[0, 0].set_ylabel('Singular Value')
    axes[0, 0].set_title('Mode 0: Segments - Singular Values')
    axes[0, 0].grid(True)
    
    axes[1, 0].plot(range(1, len(cum1) + 1), cum1, 'o-', markersize=4)
    axes[1, 0].axhline(y=0.90, color='r', linestyle='--', label='90%')
    axes[1, 0].axhline(y=0.95, color='orange', linestyle='--', label='95%')
    axes[1, 0].set_xlabel('Number of Components')
    axes[1, 0].set_ylabel('Cumulative Variance Explained')
    axes[1, 0].set_title('Mode 0: Segments - Cumulative Variance')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # Mode 1 (samples - rows)
    max_show = min(50, len(W2))
    axes[0, 1].plot(range(1, max_show + 1), W2[:max_show], 'o-', markersize=4)
    axes[0, 1].set_xlabel('Component')
    axes[0, 1].set_ylabel('Singular Value')
    axes[0, 1].set_title(f'Mode 1: Samples (Rows) - Singular Values (first {max_show})')
    axes[0, 1].grid(True)
    
    axes[1, 1].plot(range(1, max_show + 1), cum2[:max_show], 'o-', markersize=4)
    axes[1, 1].axhline(y=0.90, color='r', linestyle='--', label='90%')
    axes[1, 1].axhline(y=0.95, color='orange', linestyle='--', label='95%')
    axes[1, 1].set_xlabel('Number of Components')
    axes[1, 1].set_ylabel('Cumulative Variance Explained')
    axes[1, 1].set_title(f'Mode 1: Samples (Rows) - Cumulative Variance (first {max_show})')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    # Mode 2 (samples - cols)
    axes[0, 2].plot(range(1, max_show + 1), W3[:max_show], 'o-', markersize=4)
    axes[0, 2].set_xlabel('Component')
    axes[0, 2].set_ylabel('Singular Value')
    axes[0, 2].set_title(f'Mode 2: Samples (Cols) - Singular Values (first {max_show})')
    axes[0, 2].grid(True)
    
    axes[1, 2].plot(range(1, max_show + 1), cum3[:max_show], 'o-', markersize=4)
    axes[1, 2].axhline(y=0.90, color='r', linestyle='--', label='90%')
    axes[1, 2].axhline(y=0.95, color='orange', linestyle='--', label='95%')
    axes[1, 2].set_xlabel('Number of Components')
    axes[1, 2].set_ylabel('Cumulative Variance Explained')
    axes[1, 2].set_title(f'Mode 2: Samples (Cols) - Cumulative Variance (first {max_show})')
    axes[1, 2].legend()
    axes[1, 2].grid(True)
    
    plt.suptitle(f'Scree Plots - {args.percentage}% Subsample (Iteration {args.iteration}, Seed {args.seed})', 
                 fontsize=14, y=1.00)
    plt.tight_layout()
    
    plot_filename = f"scree_plot_pct{args.percentage:03d}_iter{args.iteration:02d}_seed{args.seed:04d}.png"
    plt.savefig(output_folder / plot_filename, dpi=300, bbox_inches='tight')
    print(f"\nScree plots saved to '{plot_filename}'", flush=True)
    
    print("\nComplete!")

if __name__ == "__main__":
    main()