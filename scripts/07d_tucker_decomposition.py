import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import tensorly as tl
from tensorly.decomposition import tucker
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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--percentage', type=int, required=True)
    parser.add_argument('--iteration', type=int, required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--rank', type=str, default='2,3,3', help='Comma-separated ranks, e.g., "2,3,3"')
    args = parser.parse_args()
    
    # Parse rank
    R1, R2, R3 = map(int, args.rank.split(','))
    
    # Paths
    indices_folder = Path("/data/users/ltucker/influenzaData/pipeline/output/tensor/subsample_indices")
    parquet_folder = Path("/data/users/ltucker/influenzaData/pipeline/output/1_distances")
    output_folder = Path("/data/users/ltucker/influenzaData/pipeline/output/tensor/tucker_decomposition_ranks_2_3_3")
    output_folder.mkdir(parents=True, exist_ok=True)
    
    # Load tensor
    indices_file = indices_folder / f"indices_pct{args.percentage:03d}_iter{args.iteration:02d}_seed{args.seed:04d}.npy"
    print(f"Loading indices from: {indices_file}")
    segment_tensor, selected_indices = load_tensor_with_indices(indices_file, parquet_folder)
    print(f"Tensor shape: {segment_tensor.shape}")
    
    print(f"\n{'='*60}")
    print(f"Performing Tucker decomposition")
    print(f"Rank: [{R1}, {R2}, {R3}]")
    print(f"{'='*60}\n")
    
    # Tucker decomposition
    core_hosvd, factors_hosvd = tucker(
        segment_tensor,
        rank=[R1, R2, R3],
        init='random',
        random_state=args.seed
    )
    
    print(f"Core tensor shape: {core_hosvd.shape}")
    print(f"Factor 0 (segments) shape: {factors_hosvd[0].shape}")
    print(f"Factor 1 (rows) shape: {factors_hosvd[1].shape}")
    print(f"Factor 2 (cols) shape: {factors_hosvd[2].shape}")
    
    # Remake the tensor from decomposition
    reconstructed_tensor = tl.tucker_to_tensor((core_hosvd, factors_hosvd))
    
    # Get them metrics
    # 1. Relative error
    relative_error = tl.norm(segment_tensor - reconstructed_tensor) / tl.norm(segment_tensor)
    print(f"\nRank: [{R1}, {R2}, {R3}]")
    
    # 2. Fit score is 1 - relative error
    fit_score = 1 - relative_error
    print(f"Relative Error: {relative_error:.6f}")
    print(f"Fit Score (1 - relative error): {fit_score:.6f}")
    
    # 3. Explained variances
    total_variance = tl.norm(segment_tensor) ** 2
    residual_variance = tl.norm(segment_tensor - reconstructed_tensor) ** 2
    explained_variance = 1 - (residual_variance / total_variance)
    print(f"Explained Variance: {explained_variance:.6f}")
    print(f"Compression Ratio: {np.prod(segment_tensor.shape) / (np.prod(core_hosvd.shape) + sum(f.size for f in factors_hosvd)):.2f}x")
    
    # Save results
    base_name = f"tucker_pct{args.percentage:03d}_iter{args.iteration:02d}_seed{args.seed:04d}_rank{R1}-{R2}-{R3}"
    
    # Save core tensor
    np.save(output_folder / f"{base_name}_core.npy", core_hosvd)
    print(f"\nSaved core tensor to: {base_name}_core.npy")
    
    # Save factor matrices
    for i, factor in enumerate(factors_hosvd):
        np.save(output_folder / f"{base_name}_factor{i}.npy", factor)
        print(f"Saved factor {i} to: {base_name}_factor{i}.npy")
    
    # Save metrics
    metrics_df = pd.DataFrame({
        'percentage': [args.percentage],
        'iteration': [args.iteration],
        'seed': [args.seed],
        'rank1': [R1],
        'rank2': [R2],
        'rank3': [R3],
        'tensor_shape_0': [segment_tensor.shape[0]],
        'tensor_shape_1': [segment_tensor.shape[1]],
        'tensor_shape_2': [segment_tensor.shape[2]],
        'core_shape_0': [core_hosvd.shape[0]],
        'core_shape_1': [core_hosvd.shape[1]],
        'core_shape_2': [core_hosvd.shape[2]],
        'relative_error': [relative_error],
        'fit_score': [fit_score],
        'explained_variance': [explained_variance],
        'compression_ratio': [np.prod(segment_tensor.shape) / (np.prod(core_hosvd.shape) + sum(f.size for f in factors_hosvd))]
    })
    
    metrics_df.to_csv(output_folder / f"{base_name}_metrics.csv", index=False)
    print(f"Saved metrics to: {base_name}_metrics.csv")
    
    print("\nComplete!")

if __name__ == "__main__":
    main()
    