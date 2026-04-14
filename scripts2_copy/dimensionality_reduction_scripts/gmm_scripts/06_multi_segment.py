#!/bin/bash
#SBATCH --job-name=gmm_joint_8d
#SBATCH --time=06:00:00
#SBATCH --mem=200G
#SBATCH --cpus-per-task=4
#SBATCH --partition=pibu_el8
#SBATCH --output=/data/users/ltucker/influenzaData/H5N1_pipeline/log/gmmJoint8D_%A.out
#SBATCH --error=/data/users/ltucker/influenzaData/H5N1_pipeline/log/gmmJoint8D_%A.err

INPUT_DIR="/data/users/ltucker/influenzaData/H5N1_pipeline/output/famsa_distances"
OUTPUT_DIR="/data/users/ltucker/influenzaData/H5N1_pipeline/output/famsa_per_segment_analysis/gmm_joint_8d"
mkdir -p "$OUTPUT_DIR"

CONTAINER="/data/users/ltucker/influenzaData/pipeline/jupyter-tensor_15.sif"

singularity exec --bind /data "$CONTAINER" python3 -c "
import numpy as np
import pandas as pd
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from scipy.stats import norm
from pathlib import Path

# ── Configuration ──────────────────────────────────────────────
SEGMENT_NAMES = ['PB2', 'PB1', 'PA', 'HA', 'NP', 'NA', 'MP', 'NS']
input_dir = Path('${INPUT_DIR}')
output_dir = Path('${OUTPUT_DIR}')

K_MIN, K_MAX = 2, 7
N_INIT = 5
SUBSAMPLE_N = 100_000       # more pairs needed for 8D
RANDOM_STATE = 42
MOD_BIC_PENALTY = 3.0

# ── Helper: strip segment-specific fields from sample name ─────
# Header format: subtype|isolate_name|SEGMENT|SEG_NUM|accession|country|continent|year
# Fields 2 and 3 are segment-specific, so we drop them for cross-segment matching
def make_common_id(full_name):
    parts = full_name.split('|')
    if len(parts) >= 5:
        return '|'.join(parts[:2] + parts[4:])
    return full_name

# ── Load all 8 distance matrices ──────────────────────────────
# Sample names live in distance_*.parquet (column00000)
# Numeric matrices live in symmetric_distances_*.parquet (pure numeric)
print('Loading all 8 segment distance matrices...')
sample_names_per_seg = {}
common_ids_per_seg = {}
np_matrices = {}

for seg_idx, seg_name in enumerate(SEGMENT_NAMES):
    # Load sample names from original distance file
    name_file = input_dir / f'distance_{seg_idx + 1}.parquet'
    if not name_file.exists():
        print(f'ERROR: {name_file} not found (needed for sample names)')
        exit(1)
    df_names = pd.read_parquet(name_file)
    # First column is sample names
    full_names = df_names.iloc[:, 0].astype(str).tolist()

    # Load numeric distance matrix from symmetric file
    dist_file = input_dir / f'symmetric_distances_{seg_idx + 1}.parquet'
    if not dist_file.exists():
        print(f'ERROR: {dist_file} not found')
        exit(1)
    df_dist = pd.read_parquet(dist_file)
    # Drop first column (numeric index), keep square matrix
    mat = df_dist.iloc[:, 1:].values.astype(float)
    n_min = min(mat.shape[0], mat.shape[1])
    mat = mat[:n_min, :n_min]
    full_names = full_names[:n_min]

    np_matrices[seg_name] = mat
    sample_names_per_seg[seg_name] = full_names
    common_ids_per_seg[seg_name] = [make_common_id(n) for n in full_names]
    print(f'  {seg_name}: {n_min} samples, matrix {mat.shape}')
    print(f'    Example full name:  {full_names[0]}')
    print(f'    Example common ID:  {common_ids_per_seg[seg_name][0]}')

# ── Find shared samples across all segments ────────────────────
all_id_sets = [set(common_ids_per_seg[s]) for s in SEGMENT_NAMES]
shared_ids = sorted(set.intersection(*all_id_sets))
n_shared = len(shared_ids)
print(f'Samples shared across all 8 segments: {n_shared}')

if n_shared == 0:
    print('ERROR: No shared samples found. Check sample name format.')
    print('First 3 common IDs per segment:')
    for s in SEGMENT_NAMES:
        print(f'  {s}: {common_ids_per_seg[s][:3]}')
    exit(1)

# Align all matrices to shared samples (by common ID lookup)
aligned_matrices = {}
for seg_name in SEGMENT_NAMES:
    cids = common_ids_per_seg[seg_name]
    cid_to_pos = {cid: i for i, cid in enumerate(cids)}
    idx_map = [cid_to_pos[cid] for cid in shared_ids]
    mat = np_matrices[seg_name]
    aligned_matrices[seg_name] = mat[np.ix_(idx_map, idx_map)]
    print(f'  {seg_name} aligned: {aligned_matrices[seg_name].shape}')

# ── Build 8D feature matrix (one row per pair) ─────────────────
print('Extracting upper triangles and building 8D feature matrix...')
triu_idx = np.triu_indices(n_shared, k=1)
n_pairs = len(triu_idx[0])
print(f'Total pairs: {n_pairs:,}')

# Stack: each column is one segment's pairwise distances
feature_matrix = np.column_stack([
    aligned_matrices[seg_name][triu_idx]
    for seg_name in SEGMENT_NAMES
])
print(f'Feature matrix shape: {feature_matrix.shape}  (pairs x segments)')

# ── Filter: keep only pairs where ALL distances are in [0, 0.15] 
mask = np.all((feature_matrix >= 0) & (feature_matrix <= 0.15), axis=1)
feature_matrix = feature_matrix[mask]
triu_idx = (triu_idx[0][mask], triu_idx[1][mask])
n_pairs_raw = n_pairs
n_pairs = len(triu_idx[0])
print(f'After filtering to [0, 0.15]: {n_pairs:,} pairs ({n_pairs/n_pairs_raw*100:.1f}% kept)')
print(f'Range per segment:')
for s_idx, s_name in enumerate(SEGMENT_NAMES):
    col = feature_matrix[:, s_idx]
    print(f'  {s_name}: [{col.min():.6f}, {col.max():.6f}]')

# ── Optional: standardise so all segments contribute equally ───
scaler = StandardScaler()
feature_matrix_scaled = scaler.fit_transform(feature_matrix)
print('Standardised features (zero mean, unit variance per segment)')

# Save scaler parameters for reproducibility
scaler_params = {
    'means': scaler.mean_.tolist(),
    'stds': scaler.scale_.tolist(),
    'segment_names': SEGMENT_NAMES,
}

# ── Subsample for model selection ──────────────────────────────
np.random.seed(RANDOM_STATE)
if n_pairs > SUBSAMPLE_N:
    sub_idx = np.random.choice(n_pairs, size=SUBSAMPLE_N, replace=False)
    subset = feature_matrix_scaled[sub_idx]
    print(f'Subsampled to {SUBSAMPLE_N:,} pairs for model selection')
else:
    subset = feature_matrix_scaled
    sub_idx = np.arange(n_pairs)
    print(f'Using all {n_pairs:,} pairs (below subsample threshold)')

# ── Helper functions ───────────────────────────────────────────
def compute_icl(gmm, data):
    bic = gmm.bic(data)
    probs = gmm.predict_proba(data)
    entropy = -np.sum(probs * np.log(probs + 1e-10))
    return bic + 2 * entropy

def compute_modified_bic(gmm, data, penalty_factor=3.0):
    n = len(data)
    p = gmm._n_parameters()
    log_likelihood = gmm.score(data) * n
    return -2 * log_likelihood + penalty_factor * p * np.log(n)

# ── Fit GMMs for k = K_MIN..K_MAX on subsample ────────────────
print(f'Fitting GMMs for k={K_MIN}..{K_MAX} on subsample...')
all_results = []

for k in range(K_MIN, K_MAX + 1):
    gmm = GaussianMixture(n_components=k, random_state=RANDOM_STATE,
                           n_init=N_INIT, covariance_type='full')
    gmm.fit(subset)
    bic_val = gmm.bic(subset)
    aic_val = gmm.aic(subset)
    icl_val = compute_icl(gmm, subset)
    mod_bic_val = compute_modified_bic(gmm, subset, penalty_factor=MOD_BIC_PENALTY)
    row = {
        'k': int(k),
        'bic': float(bic_val),
        'aic': float(aic_val),
        'icl': float(icl_val),
        'mod_bic': float(mod_bic_val),
        'converged': bool(gmm.converged_),
        'n_iter': int(gmm.n_iter_),
        # Subsample model params
        'weights': gmm.weights_.tolist(),
        'means': gmm.means_.tolist(),           # list of lists (k x 8)
        'covariances': gmm.covariances_.tolist(),  # list of 8x8 matrices
    }
    all_results.append(row)
    print(f'  k={k}  BIC={bic_val:.0f}  AIC={aic_val:.0f}  ICL={icl_val:.0f}  modBIC={mod_bic_val:.0f}  converged={gmm.converged_}')

# ── Determine best k per criterion ────────────────────────────
criteria = {'bic': 'bic', 'aic': 'aic', 'icl': 'icl', 'mod_bic': 'mod_bic'}
best_k_per_criterion = {}
for name, col in criteria.items():
    vals = [r[col] for r in all_results]
    best_idx = int(np.argmin(vals))
    best_k_per_criterion[name] = all_results[best_idx]['k']

print(f'Best k per criterion: {best_k_per_criterion}')
primary_k = best_k_per_criterion['icl']
print(f'Primary selection (ICL): k={primary_k}')

# ── Refit ALL k on FULL data ──────────────────────────────────
print(f'Refitting all k on full data ({n_pairs:,} pairs)...')
full_data_models = {}
for k in range(K_MIN, K_MAX + 1):
    gmm_full = GaussianMixture(n_components=k, random_state=RANDOM_STATE,
                                n_init=N_INIT, covariance_type='full')
    gmm_full.fit(feature_matrix_scaled)
    full_data_models[k] = gmm_full
    print(f'  Refit k={k} on full data, converged={gmm_full.converged_}')

# Enrich iteration rows with full-data params
for row in all_results:
    k = row['k']
    gf = full_data_models[k]
    row['full_data_weights'] = gf.weights_.tolist()
    row['full_data_means'] = gf.means_.tolist()
    row['full_data_covariances'] = gf.covariances_.tolist()
    row['full_data_converged'] = bool(gf.converged_)
    row['full_data_n_iter'] = int(gf.n_iter_)

best_gmm = full_data_models[primary_k]
labels = best_gmm.predict(feature_matrix_scaled)
print(f'Assigned {n_pairs:,} pairs to {primary_k} components')

# ── Save pair-level labels ─────────────────────────────────────
# Build a DataFrame with (sample_i, sample_j, label, distances...)
print('Saving pair-level assignments...')
pair_df = pd.DataFrame({
    'sample_i': [shared_ids[i] for i in triu_idx[0]],
    'sample_j': [shared_ids[j] for j in triu_idx[1]],
    'cluster': labels,
})
for seg_idx, seg_name in enumerate(SEGMENT_NAMES):
    pair_df[f'dist_{seg_name}'] = feature_matrix[:, seg_idx]

pair_df.to_parquet(output_dir / 'pair_assignments.parquet', index=False)
print(f'Saved pair assignments ({len(pair_df):,} rows)')

# ── Save component membership counts per sample ───────────────
# For each sample, count how many of its pairs fall in each component
print('Computing per-sample component profiles...')
sample_profiles = np.zeros((n_shared, primary_k), dtype=int)
for p_idx in range(n_pairs):
    c = labels[p_idx]
    sample_profiles[triu_idx[0][p_idx], c] += 1
    sample_profiles[triu_idx[1][p_idx], c] += 1

profile_df = pd.DataFrame(sample_profiles,
                           index=shared_ids,
                           columns=[f'cluster_{c}' for c in range(primary_k)])
profile_df.to_csv(output_dir / 'sample_cluster_profiles.csv')
print('Saved per-sample cluster profiles')

# ── Save all results as JSON ──────────────────────────────────
output_data = {
    'n_shared_samples': n_shared,
    'n_pairs_total': n_pairs,
    'n_pairs_subsample': int(len(subset)),
    'subsample_n': SUBSAMPLE_N,
    'mod_bic_penalty': MOD_BIC_PENALTY,
    'segment_names': SEGMENT_NAMES,
    'scaler': scaler_params,
    'best_k_per_criterion': best_k_per_criterion,
    'primary_criterion': 'icl',
    'primary_k': primary_k,
    'iterations': all_results,
}
with open(output_dir / 'gmm_joint_8d.json', 'w') as f:
    json.dump(output_data, f, indent=2)
print('Saved JSON')

# Save criteria CSV
criteria_df = pd.DataFrame(all_results)[['k', 'bic', 'aic', 'icl', 'mod_bic']]
criteria_df.to_csv(output_dir / 'gmm_joint_8d_criteria.csv', index=False)
print('Saved criteria CSV')

# ── Plot 1: Four criteria vs k ────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.flatten()
criterion_info = [
    ('bic', 'BIC', 'steelblue'),
    ('aic', 'AIC', 'darkorange'),
    ('icl', 'ICL', 'forestgreen'),
    ('mod_bic', f'Modified BIC (penalty={MOD_BIC_PENALTY})', 'firebrick'),
]
ks = [r['k'] for r in all_results]
for ax, (col, label, color) in zip(axes, criterion_info):
    vals = [r[col] for r in all_results]
    ax.plot(ks, vals, 'o-', color=color, linewidth=2, markersize=8)
    best = best_k_per_criterion[col]
    ax.axvline(best, color='red', linestyle='--', alpha=0.7, label=f'Best k={best}')
    ax.set_xlabel('Number of components (k)')
    ax.set_ylabel(label)
    ax.set_title(f'Joint 8D GMM -- {label}')
    ax.set_xticks(ks)
    ax.legend()
    ax.grid(True, alpha=0.3)
plt.suptitle('Joint 8D GMM: model selection criteria', fontsize=14, weight='bold')
plt.tight_layout()
plt.savefig(output_dir / 'gmm_joint_8d_criteria.png', dpi=150, bbox_inches='tight')
plt.close()
print('Saved criteria plot')

# ── Plot 2: Per-segment marginal histograms colored by cluster ─
fig, axes = plt.subplots(2, 4, figsize=(24, 10))
axes = axes.flatten()
cluster_colors = plt.cm.tab10(np.arange(primary_k))

for seg_idx, seg_name in enumerate(SEGMENT_NAMES):
    ax = axes[seg_idx]
    for c in range(primary_k):
        mask = labels == c
        ax.hist(feature_matrix[mask, seg_idx], bins=100, alpha=0.5,
                color=cluster_colors[c], edgecolor='none',
                label=f'Cluster {c} (n={mask.sum():,})', density=True)
    ax.set_title(seg_name, fontsize=12, weight='bold')
    ax.set_xlabel('Distance (raw)')
    if seg_idx % 4 == 0:
        ax.set_ylabel('Density')
    if seg_idx == 0:
        ax.legend(fontsize=7)

plt.suptitle(f'Joint 8D GMM (k={primary_k}): per-segment marginal distributions by cluster',
             fontsize=14, weight='bold')
plt.tight_layout()
plt.savefig(output_dir / 'gmm_joint_8d_marginals.png', dpi=150, bbox_inches='tight')
plt.close()
print('Saved marginal histograms')

# ── Plot 3: Component mean profiles (radar-like bar chart) ─────
fig, axes = plt.subplots(1, primary_k, figsize=(5 * primary_k, 5), sharey=True)
if primary_k == 1:
    axes = [axes]

# Use full-data means (in scaled space), transform back to raw
full_means_scaled = best_gmm.means_  # shape (k, 8)
full_means_raw = scaler.inverse_transform(full_means_scaled)

x_pos = np.arange(len(SEGMENT_NAMES))
for c in range(primary_k):
    ax = axes[c]
    bars = ax.bar(x_pos, full_means_raw[c], color=cluster_colors[c],
                  edgecolor='black', alpha=0.7)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(SEGMENT_NAMES, rotation=45, ha='right')
    ax.set_title(f'Cluster {c} (w={best_gmm.weights_[c]:.2f})', fontsize=11)
    ax.set_ylabel('Mean distance (raw)' if c == 0 else '')
    ax.grid(True, alpha=0.3, axis='y')

plt.suptitle(f'Joint 8D GMM (k={primary_k}): mean distance profile per cluster',
             fontsize=14, weight='bold')
plt.tight_layout()
plt.savefig(output_dir / 'gmm_joint_8d_cluster_profiles.png', dpi=150, bbox_inches='tight')
plt.close()
print('Saved cluster profile plot')

# ── Plot 4: 2D PCA projection of the 8D space ─────────────────
from sklearn.decomposition import PCA

print('Computing PCA projection for visualisation...')
# PCA on a subsample for speed
np.random.seed(RANDOM_STATE)
vis_n = min(200_000, n_pairs)
vis_idx = np.random.choice(n_pairs, size=vis_n, replace=False)

pca = PCA(n_components=2, random_state=RANDOM_STATE)
coords_2d = pca.fit_transform(feature_matrix_scaled[vis_idx])
vis_labels = labels[vis_idx]

fig, ax = plt.subplots(figsize=(10, 8))
for c in range(primary_k):
    mask = vis_labels == c
    ax.scatter(coords_2d[mask, 0], coords_2d[mask, 1],
               s=1, alpha=0.3, color=cluster_colors[c],
               label=f'Cluster {c} (n={mask.sum():,})')
ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
ax.set_title(f'Joint 8D GMM (k={primary_k}): PCA projection of pair distances')
ax.legend(markerscale=10, fontsize=9)
plt.tight_layout()
plt.savefig(output_dir / 'gmm_joint_8d_pca.png', dpi=150, bbox_inches='tight')
plt.close()
print('Saved PCA projection plot')

# ── Plot 5: PCA loadings ──────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 5))
loadings = pca.components_.T  # (8, 2)
for seg_idx, seg_name in enumerate(SEGMENT_NAMES):
    ax.arrow(0, 0, loadings[seg_idx, 0], loadings[seg_idx, 1],
             head_width=0.02, head_length=0.01, fc='steelblue', ec='steelblue')
    ax.text(loadings[seg_idx, 0]*1.1, loadings[seg_idx, 1]*1.1,
            seg_name, fontsize=10, ha='center')
ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
ax.set_title('PCA loadings: which segments drive each axis')
ax.axhline(0, color='gray', linewidth=0.5)
ax.axvline(0, color='gray', linewidth=0.5)
ax.grid(True, alpha=0.3)
ax.set_aspect('equal')
plt.tight_layout()
plt.savefig(output_dir / 'gmm_joint_8d_pca_loadings.png', dpi=150, bbox_inches='tight')
plt.close()
print('Saved PCA loadings plot')

print(f'\\nJoint 8D GMM analysis complete. Output in: {output_dir}')
"