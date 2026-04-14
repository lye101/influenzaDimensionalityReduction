#!/bin/bash
#SBATCH --job-name=gmm_segments
#SBATCH --array=0-7
#SBATCH --time=06:00:00
#SBATCH --mem=200G
#SBATCH --cpus-per-task=4
#SBATCH --partition=pibu_el8
#SBATCH --output=/data/users/ltucker/influenzaData/H5N1_pipeline/log/gmmSegment_%A_%a.out
#SBATCH --error=/data/users/ltucker/influenzaData/H5N1_pipeline/log/gmmSegment_%A_%a.err

SEGMENT_IDX=$SLURM_ARRAY_TASK_ID
SEGMENT_NAMES=("PB2" "PB1" "PA" "HA" "NP" "NA" "MP" "NS")
SEG_NAME=${SEGMENT_NAMES[$SEGMENT_IDX]}

INPUT_DIR="/data/users/ltucker/influenzaData/H5N1_pipeline/output/famsa_distances"
INPUT_FILE="${INPUT_DIR}/symmetric_distances_$((SEGMENT_IDX + 1)).parquet"
OUTPUT_DIR="/data/users/ltucker/influenzaData/H5N1_pipeline/output/famsa_per_segment_analysis/gmm"
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
from scipy.stats import norm

# ── Configuration ──────────────────────────────────────────────
seg_idx = ${SEGMENT_IDX}
seg_name = '${SEG_NAME}'
input_file = '${INPUT_FILE}'
output_dir = '${OUTPUT_DIR}'

K_MIN, K_MAX = 2, 7
N_INIT = 5
SUBSAMPLE_N = 50_000
RANDOM_STATE = 42
MOD_BIC_PENALTY = 3.0

print(f'Processing segment {seg_idx}: {seg_name}')
print(f'Reading {input_file}')

# ── Load data ──────────────────────────────────────────────────
df = pd.read_parquet(input_file)
df = df.set_index(df.columns[0])
upper_tri = df.values[np.triu_indices_from(df.values, k=1)].reshape(-1, 1)
n_total = len(upper_tri)
print(f'Upper triangle size: {n_total} pairs')

# ── Subsample for model selection ──────────────────────────────
np.random.seed(RANDOM_STATE)
if n_total > SUBSAMPLE_N:
    sub_idx = np.random.choice(n_total, size=SUBSAMPLE_N, replace=False)
    subset = upper_tri[sub_idx]
    print(f'Subsampled to {SUBSAMPLE_N} pairs for model selection')
else:
    subset = upper_tri
    print(f'Using all {n_total} pairs (below subsample threshold)')

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

# ── Fit GMMs for k = K_MIN..K_MAX ─────────────────────────────
all_results = []
all_models = {}

for k in range(K_MIN, K_MAX + 1):
    gmm = GaussianMixture(n_components=k, random_state=RANDOM_STATE, n_init=N_INIT)
    gmm.fit(subset)
    bic_sub = gmm.bic(subset)
    aic_sub = gmm.aic(subset)
    icl_sub = compute_icl(gmm, subset)
    mod_bic_sub = compute_modified_bic(gmm, subset, penalty_factor=MOD_BIC_PENALTY)
    row = {
        'k': int(k),
        'bic': float(bic_sub),
        'aic': float(aic_sub),
        'icl': float(icl_sub),
        'mod_bic': float(mod_bic_sub),
        'weights': gmm.weights_.tolist(),
        'means': gmm.means_.flatten().tolist(),
        'covariances': gmm.covariances_.flatten().tolist(),
        'converged': bool(gmm.converged_),
        'n_iter': int(gmm.n_iter_),
    }
    all_results.append(row)
    all_models[k] = gmm
    print(f'  k={k}  BIC={bic_sub:.0f}  AIC={aic_sub:.0f}  ICL={icl_sub:.0f}  modBIC={mod_bic_sub:.0f}')

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

# ── Refit ALL k on FULL data ───────────────────────────────────
full_data_models = {}
for k in range(K_MIN, K_MAX + 1):
    gmm_full = GaussianMixture(n_components=k, random_state=RANDOM_STATE, n_init=N_INIT)
    gmm_full.fit(upper_tri)
    full_data_models[k] = gmm_full
    print(f'  Refit k={k} on full data ({n_total} pairs)')

best_gmm = full_data_models[primary_k]
labels = best_gmm.predict(upper_tri)

# ── Save everything as JSON ────────────────────────────────────
# Enrich each iteration row with its full-data parameters
for row in all_results:
    k = row['k']
    gf = full_data_models[k]
    row['full_data_weights'] = gf.weights_.tolist()
    row['full_data_means'] = gf.means_.flatten().tolist()
    row['full_data_covariances'] = gf.covariances_.flatten().tolist()
    row['full_data_converged'] = bool(gf.converged_)
    row['full_data_n_iter'] = int(gf.n_iter_)

output_data = {
    'segment': seg_name,
    'segment_idx': seg_idx,
    'n_pairs_total': n_total,
    'n_pairs_subsample': int(len(subset)),
    'subsample_n': SUBSAMPLE_N,
    'mod_bic_penalty': MOD_BIC_PENALTY,
    'best_k_per_criterion': best_k_per_criterion,
    'primary_criterion': 'icl',
    'primary_k': primary_k,
    'iterations': all_results,
}
json_path = f'{output_dir}/gmm_{seg_name}.json'
with open(json_path, 'w') as f:
    json.dump(output_data, f, indent=2)
print(f'Saved JSON to {json_path}')

# ── Save criteria as CSV ──────────────────────────────────────
criteria_df = pd.DataFrame(all_results)[['k', 'bic', 'aic', 'icl', 'mod_bic']]
criteria_df['segment'] = seg_name
csv_path = f'{output_dir}/gmm_criteria_{seg_name}.csv'
criteria_df.to_csv(csv_path, index=False)
print(f'Saved criteria CSV to {csv_path}')

# ── Plot 1: density overlay (best model) ──────────────────────
colors_k = plt.cm.tab10(np.arange(primary_k))
x_range = np.linspace(upper_tri.min(), upper_tri.max(), 1000).reshape(-1, 1)

fig, axes = plt.subplots(2, 1, figsize=(12, 10))
axes[0].hist(upper_tri, bins=150, density=True, alpha=0.3, color='gray', edgecolor='none')
for i in range(primary_k):
    w = best_gmm.weights_[i]
    mu = best_gmm.means_[i, 0]
    sigma = np.sqrt(best_gmm.covariances_[i, 0, 0])
    curve = w * norm.pdf(x_range, mu, sigma)
    axes[0].plot(x_range, curve, color=colors_k[i], linewidth=2,
                 label=f'k={i+1} (mu={mu:.3f}, s={sigma:.3f}, w={w:.2f})')
total = np.exp(best_gmm.score_samples(x_range))
axes[0].plot(x_range, total, 'k--', linewidth=2, label='Total mixture')
axes[0].set_title(f'{seg_name} -- GMM density overlay (k={primary_k}, selected by ICL)')
axes[0].set_xlabel('Normalized distance')
axes[0].set_ylabel('Density')
axes[0].legend(fontsize=8)

for i in range(primary_k):
    sub = upper_tri[labels == i]
    axes[1].hist(sub, bins=150, alpha=0.6, color=colors_k[i],
                 label=f'Component {i+1} (n={len(sub):,})', edgecolor='none')
axes[1].set_title(f'{seg_name} -- Histogram colored by assignment (k={primary_k})')
axes[1].set_xlabel('Normalized distance')
axes[1].set_ylabel('Count')
axes[1].legend(fontsize=8)
plt.tight_layout()
plt.savefig(f'{output_dir}/gmm_{seg_name}.png', dpi=150, bbox_inches='tight')
plt.close()
print('Saved density plot')

# ── Plot 2: all four criteria vs k ────────────────────────────
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
    ax.set_title(f'{seg_name} -- {label}')
    ax.set_xticks(ks)
    ax.legend()
    ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f'{output_dir}/gmm_criteria_{seg_name}.png', dpi=150, bbox_inches='tight')
plt.close()
print('Saved criteria plot')

# ── Plot 3: all k overlays side by side ────────────────────────
n_k = K_MAX - K_MIN + 1
fig, axes_all = plt.subplots(1, n_k, figsize=(5 * n_k, 5), sharey=True)
if n_k == 1:
    axes_all = [axes_all]

for i_ax, k in enumerate(range(K_MIN, K_MAX + 1)):
    ax = axes_all[i_ax]
    gmm_full = full_data_models[k]
    ax.hist(upper_tri, bins=150, density=True, alpha=0.3, color='gray', edgecolor='none')
    ck = plt.cm.tab10(np.arange(k))
    for j in range(k):
        w = gmm_full.weights_[j]
        mu = gmm_full.means_[j, 0]
        sigma = np.sqrt(gmm_full.covariances_[j, 0, 0])
        curve = w * norm.pdf(x_range, mu, sigma)
        ax.plot(x_range, curve, color=ck[j], linewidth=1.5)
    total_k = np.exp(gmm_full.score_samples(x_range))
    ax.plot(x_range, total_k, 'k--', linewidth=1.5)
    selected_by = [name for name, bk in best_k_per_criterion.items() if bk == k]
    joined = ', '.join(selected_by)
    tag = f' [{joined}]' if selected_by else ''
    ax.set_title(f'k={k}{tag}', fontsize=10)
    ax.set_xlabel('Normalized distance')
    if i_ax == 0:
        ax.set_ylabel('Density')

plt.suptitle(f'{seg_name} -- GMM fits for all k values', fontsize=13, weight='bold')
plt.tight_layout()
plt.savefig(f'{output_dir}/gmm_all_k_{seg_name}.png', dpi=150, bbox_inches='tight')
plt.close()
print('Saved all-k overlay plot')

print(f'Segment {seg_name} complete.')
"