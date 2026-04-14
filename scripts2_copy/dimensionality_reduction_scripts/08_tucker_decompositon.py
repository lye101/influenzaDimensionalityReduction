import numpy as np
import pandas as pd
import tensorly as tl
from tensorly.decomposition import tucker
from pathlib import Path
import argparse


def load_tensor(parquet_folder):
    """Load all segment symmetric distance matrices and stack into a 3D tensor."""
    files = sorted(parquet_folder.glob("symmetric_distances_*.parquet"))
    all_segments = []
    for file in files:
        print(f"  Loading {file.name}")
        df = pd.read_parquet(file)
        all_segments.append(df.values)
    return tl.tensor(np.stack(all_segments, axis=0)), df.index.tolist()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--rank', type=str, default='4,5,5', help='Comma-separated ranks, e.g., "2,3,3"')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    R1, R2, R3 = map(int, args.rank.split(','))

    # Paths
    parquet_folder = Path("/data/users/ltucker/influenzaData/H5N1_pipeline/output/famsa_distances")
    output_folder = Path("/data/users/ltucker/influenzaData/H5N1_pipeline/output/famsa_tensor_analysis/tensor")
    output_folder.mkdir(parents=True, exist_ok=True)

    segment_names = ["PB2", "PB1", "PA", "HA", "NP", "NA", "MP", "NS"]

    # Load full tensor (no subsampling)
    print("Loading tensor...")
    segment_tensor, sample_ids = load_tensor(parquet_folder)
    print(f"Tensor shape: {segment_tensor.shape}")

    print(f"\n{'='*60}")
    print(f"Performing Tucker decomposition")
    print(f"Rank: [{R1}, {R2}, {R3}]")
    print(f"{'='*60}\n")

    # Tucker decomposition (SVD init for stability and reproducibility)
    core_hosvd, factors_hosvd = tucker(
        segment_tensor,
        rank=[R1, R2, R3],
        init='svd',
        random_state=args.seed
    )

    print(f"Core tensor shape: {core_hosvd.shape}")
    print(f"Factor 0 (segments) shape: {factors_hosvd[0].shape}")
    print(f"Factor 1 (rows) shape: {factors_hosvd[1].shape}")
    print(f"Factor 2 (cols) shape: {factors_hosvd[2].shape}")

    # Reconstruct and compute metrics
    reconstructed_tensor = tl.tucker_to_tensor((core_hosvd, factors_hosvd))

    relative_error = tl.norm(segment_tensor - reconstructed_tensor) / tl.norm(segment_tensor)
    fit_score = 1 - relative_error
    total_variance = tl.norm(segment_tensor) ** 2
    residual_variance = tl.norm(segment_tensor - reconstructed_tensor) ** 2
    explained_variance = 1 - (residual_variance / total_variance)
    compression_ratio = np.prod(segment_tensor.shape) / (np.prod(core_hosvd.shape) + sum(f.size for f in factors_hosvd))

    print(f"\nRelative Error: {relative_error:.6f}")
    print(f"Fit Score (1 - relative error): {fit_score:.6f}")
    print(f"Explained Variance: {explained_variance:.6f}")
    print(f"Compression Ratio: {compression_ratio:.2f}x")

    # Save results
    base_name = f"tucker_full_rank{R1}-{R2}-{R3}_seed{args.seed}"

    np.save(output_folder / f"{base_name}_core.npy", core_hosvd)
    print(f"\nSaved core tensor to: {base_name}_core.npy")

    for i, factor in enumerate(factors_hosvd):
        np.save(output_folder / f"{base_name}_factor{i}.npy", factor)
        print(f"Saved factor {i} to: {base_name}_factor{i}.npy")

    # Save sample IDs and segment names as reference parquets
    pd.Series(sample_ids, name="sample_id").to_frame().to_parquet(
        output_folder / "sample_ids.parquet", index=False
    )
    pd.Series(segment_names, name="segment").to_frame().to_parquet(
        output_folder / "segment_names.parquet", index=False
    )
    print("Saved sample_ids.parquet")
    print("Saved segment_names.parquet")

    metrics_df = pd.DataFrame({
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
        'relative_error': [float(relative_error)],
        'fit_score': [float(fit_score)],
        'explained_variance': [float(explained_variance)],
        'compression_ratio': [compression_ratio]
    })

    metrics_df.to_csv(output_folder / f"{base_name}_metrics.csv", index=False)
    print(f"Saved metrics to: {base_name}_metrics.csv")

    print("\nComplete!")


if __name__ == "__main__":
    main()