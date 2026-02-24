import os
from pathlib import Path
import numpy as np
import pandas as pd

## Configuration (including 1.0 for full data)
subsample_config = {
    0.05: 5,
    0.1: 5,
    0.2: 5,
    0.3: 5,
    0.5: 3,
    0.7: 3,
    1.0: 1
}

## Paths
folder = Path("/data/users/ltucker/influenzaData/pipeline/output/1_distances")
output_folder = Path("/data/users/ltucker/influenzaData/pipeline/output/tensor/subsample_indices")
output_folder.mkdir(parents=True, exist_ok=True)

files = sorted(folder.glob("*.parquet"))

# Get total sample size from first file
df_sample = pd.read_parquet(files[0])
n_total_samples = len(df_sample)
n_segments = len(files)
del df_sample

print(f"Total samples: {n_total_samples}")
print(f"Total segments: {n_segments}\n")

random_states = []

## Generate and save indices
for pct_idx, (subsample_ratio, n_repeats) in enumerate(subsample_config.items()):
    print(f"{'='*60}")
    print(f"Generating indices for ratio: {subsample_ratio} ({n_repeats} repeats)")
    print(f"{'='*60}\n")
    
    for repeat in range(n_repeats):
        seed = pct_idx * 1000 + repeat
        random_states.append(seed)
        
        np.random.seed(seed)
        n_keep = int(n_total_samples * subsample_ratio)
        
        if subsample_ratio == 1.0:
            # Full data - all indices
            selected_indices = np.arange(n_total_samples)
        else:
            # Subsample
            selected_indices = np.sort(np.random.choice(n_total_samples, 
                                                         size=n_keep, 
                                                         replace=False))
        
        print(f"Repeat {repeat + 1}/{n_repeats}, Seed: {seed}")
        print(f"  Selected {len(selected_indices)} samples")
        
        # Save indices as numpy
        indices_filename = f"indices_pct{int(subsample_ratio*100):03d}_iter{repeat:02d}_seed{seed:04d}.npy"
        indices_filepath = output_folder / indices_filename
        np.save(str(indices_filepath), selected_indices)
        
        # Save metadata CSV
        metadata_filename = f"metadata_pct{int(subsample_ratio*100):03d}_iter{repeat:02d}_seed{seed:04d}.csv"
        metadata_filepath = output_folder / metadata_filename
        
        metadata_df = pd.DataFrame({
            'subsample_ratio': [subsample_ratio],
            'percentage': [int(subsample_ratio * 100)],
            'seed': [seed],
            'iteration': [repeat],
            'n_total_samples': [n_total_samples],
            'n_selected_samples': [len(selected_indices)],
            'n_segments': [n_segments],
            'indices_file': [indices_filename]
        })
        metadata_df.to_csv(metadata_filepath, index=False)
        
        print(f"  Saved: {indices_filename}")
        print()

# Save master index file
master_df = pd.DataFrame({
    'subsample_ratio': [ratio for ratio, n in subsample_config.items() for _ in range(n)],
    'percentage': [int(ratio*100) for ratio, n in subsample_config.items() for _ in range(n)],
    'iteration': [i for _, n in subsample_config.items() for i in range(n)],
    'seed': random_states
})
master_df.to_csv(output_folder / 'master_index.csv', index=False)

print(f"{'='*60}")
print(f"Generated {len(random_states)} subsample configurations")
print(f"Storage: ~{len(random_states) * n_total_samples * 8 / (1024**2):.1f} MB")
print(f"Master index saved to: {output_folder / 'master_index.csv'}")
print(f"{'='*60}")