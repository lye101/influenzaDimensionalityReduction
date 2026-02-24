#!/bin/bash
#SBATCH --job-name=embed_scale
#SBATCH --output=/data/users/ltucker/influenzaData/pipeline/log/embeddings_nanduri_params/embedding_scaling_%A_%a.log
#SBATCH --error=/data/users/ltucker/influenzaData/pipeline/log/embeddings_nanduri_params/embedding_scaling_%A_%a.err
#SBATCH --open-mode=append
#SBATCH --time=7-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=100G
#SBATCH --partition=pibu_el8
#SBATCH --array=0-7  # Adjust based on number of segments

# Get segment index from array task ID
SEGMENT_IDX=${SLURM_ARRAY_TASK_ID}

echo "Processing segment ${SEGMENT_IDX}"

# Create Python script
cat > /tmp/embedding_scaling_${SLURM_JOB_ID}_seg${SEGMENT_IDX}.py << EOF
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE, MDS
import umap
from sklearn.metrics import pairwise_distances
from scipy.stats import spearmanr
from sklearn.manifold import trustworthiness
from pathlib import Path

# Segment to process
SEGMENT_IDX = ${SEGMENT_IDX}

# Load data
print(f"Loading distance matrix for segment {SEGMENT_IDX}...", flush=True)
folder = Path("/data/users/ltucker/influenzaData/pipeline/output/1_distances")
files = sorted(folder.glob("*.parquet"))

if SEGMENT_IDX >= len(files):
    print(f"Error: Segment index {SEGMENT_IDX} out of range (max {len(files)-1})", flush=True)
    exit(1)

# Load only the segment we need
file = files[SEGMENT_IDX]
print(f"Loading file: {file}", flush=True)
df = pd.read_parquet(file)
df = df.set_index("column00000")
print(f"Shape: {df.shape}", flush=True)

# Create output directories
output_dir = '/data/users/ltucker/influenzaData/pipeline/output/jupyter_outputs/nanduri_params'
embeddings_dir = os.path.join(output_dir, 'embeddings')
os.makedirs(output_dir, exist_ok=True)
os.makedirs(embeddings_dir, exist_ok=True)

# Define percentages to test
percentages = [0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0]

# Number of repeats for robustness
n_repeats = 5

# Storage for results
segment_results = []

print(f"\\nProcessing segment {SEGMENT_IDX}...", flush=True)

for pct_idx, pct in enumerate(percentages):
    n_samples = int(pct * len(df))
    print(f"\\n{'='*80}", flush=True)
    print(f"Testing {pct*100}% ({n_samples} samples)...", flush=True)
    print(f"{'='*80}", flush=True)
    
    for repeat in range(n_repeats):
        # Set seed for reproducibility
        seed = pct_idx * 1000 + repeat
        np.random.seed(seed)
        print(f"Repeat {repeat + 1}/{n_repeats} - Random seed: {seed}", flush=True)
        
        # Sample positions
        sample_positions = np.random.choice(len(df), size=n_samples, replace=False)
        
        # Subsample the matrix
        subsampled = df.iloc[sample_positions, sample_positions]
        sampled_indices = df.index[sample_positions]
        
        print(f"  Running embeddings...", end=' ', flush=True)
        
        # Run t-SNE
        tsne = TSNE(n_components=2,
                    perplexity=200,
                    learning_rate=100, 
                    metric='precomputed', 
                    init='random', 
                    random_state=seed)

        tsne_embedding = tsne.fit_transform(subsampled)
        
        # Run UMAP
        umap_model = umap.UMAP(n_components=2, 
                               metric='precomputed', 
                               random_state=seed,
                               n_neighbors=100,
                               min_dist=0.1)

        umap_embedding = umap_model.fit_transform(subsampled)
        
        # Run MDS (symmetrize matrix first)
        subsampled_symmetric = (subsampled + subsampled.T) / 2
        mds = MDS(n_components=2, 
                  dissimilarity='precomputed', 
                  random_state=seed, 
                  n_init=4, 
                  init='random')

        mds_embedding = mds.fit_transform(subsampled_symmetric)
        
        print("Saving...", end=' ', flush=True)
        
        # Save embeddings as CSVs
        pct_str = f"{int(pct*100)}pct"
        
        # Save t-SNE embedding
        tsne_df = pd.DataFrame(
            tsne_embedding, 
            columns=['tsne_dim1', 'tsne_dim2'],
            index=sampled_indices
        )
        tsne_filename = os.path.join(embeddings_dir, f'segment_{SEGMENT_IDX}_tsne_{pct_str}_seed_{seed}.csv')
        tsne_df.to_csv(tsne_filename)
        
        # Save UMAP embedding
        umap_df = pd.DataFrame(
            umap_embedding, 
            columns=['umap_dim1', 'umap_dim2'],
            index=sampled_indices
        )
        umap_filename = os.path.join(embeddings_dir, f'segment_{SEGMENT_IDX}_umap_{pct_str}_seed_{seed}.csv')
        umap_df.to_csv(umap_filename)
        
        # Save MDS embedding
        mds_df = pd.DataFrame(
            mds_embedding, 
            columns=['mds_dim1', 'mds_dim2'],
            index=sampled_indices
        )
        mds_filename = os.path.join(embeddings_dir, f'segment_{SEGMENT_IDX}_mds_{pct_str}_seed_{seed}.csv')
        mds_df.to_csv(mds_filename)
        
        print("Computing metrics...", end=' ', flush=True)
        
        # Calculate distances in 2D embeddings
        tsne_distances = pairwise_distances(tsne_embedding)
        umap_distances = pairwise_distances(umap_embedding)
        mds_distances = pairwise_distances(mds_embedding)
        
        # Flatten distance matrices for correlation
        orig_flat = subsampled.values[np.triu_indices_from(subsampled.values, k=1)]
        tsne_flat = tsne_distances[np.triu_indices_from(tsne_distances, k=1)]
        umap_flat = umap_distances[np.triu_indices_from(umap_distances, k=1)]
        mds_flat = mds_distances[np.triu_indices_from(mds_distances, k=1)]
        
        # Spearman correlation
        tsne_corr, _ = spearmanr(orig_flat, tsne_flat)
        umap_corr, _ = spearmanr(orig_flat, umap_flat)
        mds_corr, _ = spearmanr(orig_flat, mds_flat)
        
        # MDS stress
        stress = mds.stress_
        
        # Trustworthiness
        tsne_trust = trustworthiness(subsampled, tsne_embedding, n_neighbors=15, metric='precomputed')
        umap_trust = trustworthiness(subsampled, umap_embedding, n_neighbors=15, metric='precomputed')
        mds_trust = trustworthiness(subsampled, mds_embedding, n_neighbors=15, metric='precomputed')
        
        # Store results
        result = {
            'segment': SEGMENT_IDX,
            'percentage': pct * 100,
            'n_samples': n_samples,
            'repeat': repeat,
            'seed': seed,
            'tsne_correlation': tsne_corr,
            'umap_correlation': umap_corr,
            'mds_correlation': mds_corr,
            'tsne_trustworthiness': tsne_trust,
            'umap_trustworthiness': umap_trust,
            'mds_trustworthiness': mds_trust,
            'mds_stress': stress,
            'tsne_file': tsne_filename,
            'umap_file': umap_filename,
            'mds_file': mds_filename
        }
        segment_results.append(result)
        
        print("Done", flush=True)

# Save results
print(f"\\n{'='*80}", flush=True)
print(f"Saving results for segment {SEGMENT_IDX}...", flush=True)
print(f"{'='*80}", flush=True)

df_all = pd.DataFrame(segment_results)

# Group by percentage and calculate statistics
results_summary = []
for pct in percentages:
    pct_data = df_all[df_all['percentage'] == pct * 100]
    
    results_summary.append({
        'segment': SEGMENT_IDX,
        'percentage': pct * 100,
        'n_samples': pct_data['n_samples'].iloc[0],
        'tsne_correlation_mean': pct_data['tsne_correlation'].mean(),
        'tsne_correlation_std': pct_data['tsne_correlation'].std(),
        'umap_correlation_mean': pct_data['umap_correlation'].mean(),
        'umap_correlation_std': pct_data['umap_correlation'].std(),
        'mds_correlation_mean': pct_data['mds_correlation'].mean(),
        'mds_correlation_std': pct_data['mds_correlation'].std(),
        'tsne_trustworthiness_mean': pct_data['tsne_trustworthiness'].mean(),
        'tsne_trustworthiness_std': pct_data['tsne_trustworthiness'].std(),
        'umap_trustworthiness_mean': pct_data['umap_trustworthiness'].mean(),
        'umap_trustworthiness_std': pct_data['umap_trustworthiness'].std(),
        'mds_trustworthiness_mean': pct_data['mds_trustworthiness'].mean(),
        'mds_trustworthiness_std': pct_data['mds_trustworthiness'].std(),
        'mds_stress_mean': pct_data['mds_stress'].mean(),
        'mds_stress_std': pct_data['mds_stress'].std()
    })

df_results = pd.DataFrame(results_summary)

# Save summary DataFrame
csv_filename = os.path.join(output_dir, f'embedding_scaling_segment_{SEGMENT_IDX}.csv')
df_results.to_csv(csv_filename, index=False)

# Save detailed DataFrame
csv_detailed = os.path.join(output_dir, f'embedding_scaling_segment_{SEGMENT_IDX}_detailed.csv')
df_all.to_csv(csv_detailed, index=False)

print(f"Saved summary to {csv_filename}", flush=True)
print(f"Saved detailed results to {csv_detailed}", flush=True)

# Create plots
fig = plt.figure(figsize=(20, 12))

# Plot 1: t-SNE Correlation
ax1 = plt.subplot(2, 4, 1)
ax1.errorbar(df_results['percentage'], df_results['tsne_correlation_mean'], 
             yerr=df_results['tsne_correlation_std'], marker='o', capsize=5, linewidth=2)
ax1.set_xlabel('Percentage of Data (%)', fontsize=12)
ax1.set_ylabel('Distance Correlation', fontsize=12)
ax1.set_title(f't-SNE Distance Preservation', fontsize=14)
ax1.grid(True, alpha=0.3)

# Plot 2: UMAP Correlation
ax2 = plt.subplot(2, 4, 2)
ax2.errorbar(df_results['percentage'], df_results['umap_correlation_mean'], 
             yerr=df_results['umap_correlation_std'], marker='o', capsize=5, 
             linewidth=2, color='orange')
ax2.set_xlabel('Percentage of Data (%)', fontsize=12)
ax2.set_ylabel('Distance Correlation', fontsize=12)
ax2.set_title(f'UMAP Distance Preservation', fontsize=14)
ax2.grid(True, alpha=0.3)

# Plot 3: MDS Correlation
ax3 = plt.subplot(2, 4, 3)
ax3.errorbar(df_results['percentage'], df_results['mds_correlation_mean'], 
             yerr=df_results['mds_correlation_std'], marker='o', capsize=5, 
             linewidth=2, color='purple')
ax3.set_xlabel('Percentage of Data (%)', fontsize=12)
ax3.set_ylabel('Distance Correlation', fontsize=12)
ax3.set_title(f'MDS Distance Preservation', fontsize=14)
ax3.grid(True, alpha=0.3)

# Plot 4: MDS Stress
ax4 = plt.subplot(2, 4, 4)
ax4.errorbar(df_results['percentage'], df_results['mds_stress_mean'], 
             yerr=df_results['mds_stress_std'], marker='o', capsize=5, 
             linewidth=2, color='red')
ax4.set_xlabel('Percentage of Data (%)', fontsize=12)
ax4.set_ylabel('Stress (lower is better)', fontsize=12)
ax4.set_title(f'MDS Stress', fontsize=14)
ax4.grid(True, alpha=0.3)

# Plot 5: t-SNE Trustworthiness
ax5 = plt.subplot(2, 4, 5)
ax5.errorbar(df_results['percentage'], df_results['tsne_trustworthiness_mean'], 
             yerr=df_results['tsne_trustworthiness_std'], marker='o', capsize=5, 
             linewidth=2, color='green')
ax5.set_xlabel('Percentage of Data (%)', fontsize=12)
ax5.set_ylabel('Trustworthiness', fontsize=12)
ax5.set_title(f't-SNE Trustworthiness', fontsize=14)
ax5.grid(True, alpha=0.3)

# Plot 6: UMAP Trustworthiness
ax6 = plt.subplot(2, 4, 6)
ax6.errorbar(df_results['percentage'], df_results['umap_trustworthiness_mean'], 
             yerr=df_results['umap_trustworthiness_std'], marker='o', capsize=5, 
             linewidth=2, color='brown')
ax6.set_xlabel('Percentage of Data (%)', fontsize=12)
ax6.set_ylabel('Trustworthiness', fontsize=12)
ax6.set_title(f'UMAP Trustworthiness', fontsize=14)
ax6.grid(True, alpha=0.3)

# Plot 7: MDS Trustworthiness
ax7 = plt.subplot(2, 4, 7)
ax7.errorbar(df_results['percentage'], df_results['mds_trustworthiness_mean'], 
             yerr=df_results['mds_trustworthiness_std'], marker='o', capsize=5, 
             linewidth=2, color='teal')
ax7.set_xlabel('Percentage of Data (%)', fontsize=12)
ax7.set_ylabel('Trustworthiness', fontsize=12)
ax7.set_title(f'MDS Trustworthiness', fontsize=14)
ax7.grid(True, alpha=0.3)

# Plot 8: Comparison
ax8 = plt.subplot(2, 4, 8)
ax8.plot(df_results['percentage'], df_results['tsne_correlation_mean'], 
         marker='o', label='t-SNE', linewidth=2)
ax8.plot(df_results['percentage'], df_results['umap_correlation_mean'], 
         marker='s', label='UMAP', linewidth=2)
ax8.plot(df_results['percentage'], df_results['mds_correlation_mean'], 
         marker='^', label='MDS', linewidth=2)
ax8.set_xlabel('Percentage of Data (%)', fontsize=12)
ax8.set_ylabel('Distance Correlation', fontsize=12)
ax8.set_title(f'Method Comparison', fontsize=14)
ax8.legend()
ax8.grid(True, alpha=0.3)

plt.suptitle(f'Embedding Quality vs Data Size - Segment {SEGMENT_IDX}', fontsize=16, y=0.995)
plt.tight_layout()

# Save plot
plot_filename = os.path.join(output_dir, f'embedding_scaling_segment_{SEGMENT_IDX}.png')
plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
plt.close()
print(f"Saved plot to {plot_filename}", flush=True)

print(f"\\nSegment {SEGMENT_IDX} completed successfully!", flush=True)
print(f"Individual embeddings saved to: {embeddings_dir}", flush=True)
EOF

# Run the Python script in the container
singularity exec --cleanenv \
    --bind /data \
    /data/users/ltucker/influenzaData/pipeline/jupyter-tensor_12.sif \
    python -u /tmp/embedding_scaling_${SLURM_JOB_ID}_seg${SEGMENT_IDX}.py

# Clean up
rm /tmp/embedding_scaling_${SLURM_JOB_ID}_seg${SEGMENT_IDX}.py

echo "Segment ${SEGMENT_IDX} job completed!"