import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE, MDS
from umap import UMAP
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--percentage', type=int, required=True)
    parser.add_argument('--iteration', type=int, required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--rank', type=str, default='1,2,2', help='Comma-separated ranks used in Tucker')
    args = parser.parse_args()
    
    R1, R2, R3 = map(int, args.rank.split(','))
    
    # Paths
    tucker_folder = Path("/data/users/ltucker/influenzaData/pipeline/output/tucker_decomposition")
    output_folder = Path("/data/users/ltucker/influenzaData/pipeline/output/dimensionality_reduction")
    output_folder.mkdir(parents=True, exist_ok=True)
    
    # Load factor matrices
    base_name = f"tucker_pct{args.percentage:03d}_iter{args.iteration:02d}_seed{args.seed:04d}_rank{R1}-{R2}-{R3}"
    
    print(f"Loading factor matrices from: {base_name}")
    factor1 = np.load(tucker_folder / f"{base_name}_factor1.npy")  # R2 - rows
    factor2 = np.load(tucker_folder / f"{base_name}_factor2.npy")  # R3 - cols
    
    print(f"Factor 1 (R2) shape: {factor1.shape}")
    print(f"Factor 2 (R3) shape: {factor2.shape}")
    
    # Combine factors for dimensionality reduction
    # We'll do DR on both factors separately
    
    results = {}
    
    for factor_idx, factor_matrix in enumerate([factor1, factor2], start=1):
        factor_name = f"factor{factor_idx}"
        print(f"\n{'='*60}")
        print(f"Processing {factor_name} (shape: {factor_matrix.shape})")
        print(f"{'='*60}")
        
        # t-SNE
        print("\nRunning t-SNE...")
        tsne = TSNE(n_components=2, random_state=args.seed, perplexity=min(30, factor_matrix.shape[0]-1))
        tsne_result = tsne.fit_transform(factor_matrix)
        results[f'{factor_name}_tsne'] = tsne_result
        print(f"t-SNE complete - shape: {tsne_result.shape}")
        
        # UMAP
        print("Running UMAP...")
        umap = UMAP(n_components=2, random_state=args.seed, n_neighbors=min(15, factor_matrix.shape[0]-1))
        umap_result = umap.fit_transform(factor_matrix)
        results[f'{factor_name}_umap'] = umap_result
        print(f"UMAP complete - shape: {umap_result.shape}")
        
        # MDS
        print("Running MDS...")
        mds = MDS(n_components=2, random_state=args.seed)
        mds_result = mds.fit_transform(factor_matrix)
        results[f'{factor_name}_mds'] = mds_result
        print(f"MDS complete - shape: {mds_result.shape}")
        
        # Save results
        for method in ['tsne', 'umap', 'mds']:
            result_key = f'{factor_name}_{method}'
            result_data = results[result_key]
            
            # Save as CSV
            result_df = pd.DataFrame({
                'component_1': result_data[:, 0],
                'component_2': result_data[:, 1],
                'sample_index': range(result_data.shape[0])
            })
            
            csv_filename = f"dimred_{method}_{factor_name}_pct{args.percentage:03d}_iter{args.iteration:02d}_seed{args.seed:04d}.csv"
            result_df.to_csv(output_folder / csv_filename, index=False)
            print(f"Saved {method.upper()} results to: {csv_filename}")
            
            # Save as numpy
            npy_filename = f"dimred_{method}_{factor_name}_pct{args.percentage:03d}_iter{args.iteration:02d}_seed{args.seed:04d}.npy"
            np.save(output_folder / npy_filename, result_data)
    
    # Create grid plots
    print("\nCreating visualization grid...")
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    methods = ['tsne', 'umap', 'mds']
    method_names = ['t-SNE', 'UMAP', 'MDS']
    
    for col_idx, (method, method_name) in enumerate(zip(methods, method_names)):
        for row_idx, factor_idx in enumerate([1, 2]):
            factor_name = f"factor{factor_idx}"
            result_key = f'{factor_name}_{method}'
            result_data = results[result_key]
            
            ax = axes[row_idx, col_idx]
            scatter = ax.scatter(result_data[:, 0], result_data[:, 1], 
                               c=range(result_data.shape[0]), 
                               cmap='viridis', 
                               alpha=0.6, 
                               s=20)
            ax.set_xlabel('Component 1')
            ax.set_ylabel('Component 2')
            ax.set_title(f'{method_name} - Factor {factor_idx}')
            ax.grid(True, alpha=0.3)
            plt.colorbar(scatter, ax=ax, label='Sample Index')
    
    plt.suptitle(f'Dimensionality Reduction - {args.percentage}% (Iter {args.iteration}, Seed {args.seed})', 
                 fontsize=16, y=0.995)
    plt.tight_layout()
    
    plot_filename = f"dimred_grid_pct{args.percentage:03d}_iter{args.iteration:02d}_seed{args.seed:04d}.png"
    plt.savefig(output_folder / plot_filename, dpi=300, bbox_inches='tight')
    print(f"\nGrid plot saved to: {plot_filename}")
    
    print("\nComplete!")

if __name__ == "__main__":
    main()