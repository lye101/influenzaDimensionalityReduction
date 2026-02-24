import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import tensorly as tl
import argparse

def load_tensor_with_indices(indices_file, parquet_folder):
    """Load full tensor and subsample using indices, returning tensor and sample names"""
    indices_path = Path(indices_file)
    selected_indices = np.load(indices_path)
    
    files = sorted(Path(parquet_folder).glob("*.parquet"))
    all_segments = []
    all_sample_names = []
    segment_names = []
    
    for file in files:
        df = pd.read_parquet(file)
        
        # Store sample names from first segment only
        if len(all_sample_names) == 0:
            all_sample_names = df.index.tolist()
        
        # Store segment name
        segment_names.append(file.stem)
        
        df = df.set_index("column00000")
        all_segments.append(df.values)
    
    full_tensor = tl.tensor(np.stack(all_segments, axis=0))
    subsampled_tensor = full_tensor[:, selected_indices, :][:, :, selected_indices]
    
    # Get selected sample names
    selected_sample_names = [all_sample_names[i] for i in selected_indices]
    
    return subsampled_tensor, selected_indices, selected_sample_names, segment_names

def parse_metadata(sample_names):
    """Parse metadata from sample names"""
    # Ensure sample_names are strings
    metadata = pd.DataFrame({"sample_id": [str(name) for name in sample_names]})
    
    # Header format: A/H5N0|A/chicken/Fujian/9.24_FZHX0071-O/2018|PB2|1|EPI_ISL_697987|china|asia|2018
    # Positions:     0     |1                                    |2  |3|4            |5     |6   |7
    
    # Split and extract fields
    split_names = metadata["sample_id"].str.split("|")
    
    metadata["segment"] = split_names.str[2]
    metadata["country"] = split_names.str[5]
    metadata["continent"] = split_names.str[6]
    metadata["year"] = split_names.str[7]
    
    return metadata

def aggregate_loadings_by_category(loadings, metadata, category, top_n=None):
    """
    Aggregate loadings by a categorical variable
    
    Returns: DataFrame with mean absolute loading per category per component
    """
    df = pd.DataFrame(loadings)
    df[category] = metadata[category].values
    
    # Group by category and calculate mean absolute loading
    grouped = df.groupby(category).agg(lambda x: np.mean(np.abs(x)))
    
    # Calculate proportion (normalize each column to sum to 1)
    proportions = grouped.div(grouped.sum(axis=0), axis=1)
    
    # If top_n specified, keep only top N categories per component
    if top_n:
        # For each component, keep top N categories
        mask = pd.DataFrame(False, index=proportions.index, columns=proportions.columns)
        for col in proportions.columns:
            top_cats = proportions[col].nlargest(top_n).index
            mask.loc[top_cats, col] = True
        proportions = proportions[mask].fillna(0)
    
    return proportions

def plot_loading_heatmap(loadings, labels, title, output_path, top_n=None):
    """Create heatmap of loadings"""
    n_components = loadings.shape[1]
    
    # If top_n specified, show only top contributing features per component
    if top_n:
        # Get top N by mean absolute loading across components
        mean_abs_loading = np.mean(np.abs(loadings), axis=1)
        top_indices = np.argsort(mean_abs_loading)[-top_n:]
        loadings = loadings[top_indices, :]
        labels = [labels[i] for i in top_indices]
    
    fig, ax = plt.subplots(figsize=(min(12, 2 + n_components), min(16, 2 + len(labels) * 0.3)))
    
    sns.heatmap(loadings, 
                xticklabels=[f'PC{i+1}' for i in range(n_components)],
                yticklabels=labels,
                cmap='RdBu_r',
                center=0,
                cbar_kws={'label': 'Loading'},
                ax=ax)
    
    ax.set_title(title, fontsize=14, pad=10)
    ax.set_xlabel('Principal Component', fontsize=11)
    ax.set_ylabel('Feature', fontsize=11)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved: {output_path.name}")

def plot_stacked_bar_proportions(proportions_dict, component_idx, output_path):
    """Create stacked bar chart showing proportions for one component across categories"""
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    categories = ['segment', 'year', 'continent', 'country']
    
    for idx, category in enumerate(categories):
        ax = axes[idx]
        props = proportions_dict[category]
        
        if component_idx < props.shape[1]:
            # Get data for this component
            data = props.iloc[:, component_idx].sort_values(ascending=False)
            
            # Skip if no data
            if len(data) == 0 or data.sum() == 0:
                ax.text(0.5, 0.5, 'No data available', 
                       ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f'PC{component_idx+1} - {category.capitalize()}', fontsize=12)
                continue
            
            # Plot - don't pass colors to plot(), let pandas handle it
            data.plot(kind='barh', ax=ax)
            
            ax.set_xlabel('Proportion of Component', fontsize=11)
            ax.set_ylabel(category.capitalize(), fontsize=11)
            ax.set_title(f'PC{component_idx+1} - {category.capitalize()} Composition', fontsize=12)
            ax.set_xlim([0, 1])
            ax.grid(axis='x', alpha=0.3)
        else:
            ax.text(0.5, 0.5, 'Component not available', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'PC{component_idx+1} - {category.capitalize()}', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved: {output_path.name}")

def plot_loading_composition_grid(proportions_dict, n_components, output_path):
    """Create grid showing composition of all components across all categories"""
    
    categories = ['segment', 'year', 'continent', 'country']
    
    fig, axes = plt.subplots(len(categories), n_components, 
                            figsize=(4 * n_components, 4 * len(categories)))
    
    # Handle single component case
    if n_components == 1:
        axes = axes.reshape(-1, 1)
    
    for cat_idx, category in enumerate(categories):
        props = proportions_dict[category]
        
        for comp_idx in range(n_components):
            ax = axes[cat_idx, comp_idx]
            
            if comp_idx < props.shape[1]:
                # Get top 10 categories for this component
                data = props.iloc[:, comp_idx].nlargest(10).sort_values(ascending=True)
                
                colors = plt.cm.Set3(np.linspace(0, 1, len(data)))
                data.plot(kind='barh', ax=ax, color=colors, legend=False)
                
                ax.set_xlabel('Proportion', fontsize=9)
                if comp_idx == 0:
                    ax.set_ylabel(category.capitalize(), fontsize=10)
                else:
                    ax.set_ylabel('')
                
                if cat_idx == 0:
                    ax.set_title(f'PC{comp_idx+1}', fontsize=11, fontweight='bold')
                
                ax.set_xlim([0, max(1, data.max() * 1.1)])
                ax.grid(axis='x', alpha=0.3)
                ax.tick_params(axis='both', labelsize=8)
            else:
                ax.axis('off')
    
    plt.suptitle('Component Composition Across Categories (Top 10 per component)', 
                 fontsize=14, y=0.995)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved: {output_path.name}")

def calculate_loading_statistics(loadings_list, metadata_list, category):
    """Calculate mean and standard error of proportions across iterations"""
    
    all_proportions = []
    
    for loadings, metadata in zip(loadings_list, metadata_list):
        props = aggregate_loadings_by_category(loadings, metadata, category)
        all_proportions.append(props)
    
    # Align all dataframes to have same index and columns
    all_indices = set()
    all_columns = set()
    for df in all_proportions:
        all_indices.update(df.index)
        all_columns.update(df.columns)
    
    all_indices = sorted(all_indices)
    all_columns = sorted(all_columns)
    
    # Reindex all dataframes
    aligned_props = [df.reindex(index=all_indices, columns=all_columns, fill_value=0) 
                     for df in all_proportions]
    
    # Stack into 3D array
    props_array = np.stack([df.values for df in aligned_props], axis=0)
    
    # Calculate statistics
    mean_props = np.mean(props_array, axis=0)
    std_props = np.std(props_array, axis=0)
    se_props = std_props / np.sqrt(len(loadings_list))
    
    mean_df = pd.DataFrame(mean_props, index=all_indices, columns=all_columns)
    se_df = pd.DataFrame(se_props, index=all_indices, columns=all_columns)
    
    return mean_df, se_df

def plot_proportions_with_error(mean_props, se_props, component_idx, category, output_path):
    """Plot proportions with error bars for one component"""
    
    if component_idx >= mean_props.shape[1]:
        print(f"Component {component_idx} not available for {category}")
        return
    
    # Get data
    means = mean_props.iloc[:, component_idx].sort_values(ascending=False).head(15)
    errors = se_props.loc[means.index, component_idx]
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    y_pos = np.arange(len(means))
    colors = plt.cm.Set3(np.linspace(0, 1, len(means)))
    
    ax.barh(y_pos, means.values, xerr=errors.values, 
            color=colors, alpha=0.7, capsize=5)
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(means.index)
    ax.set_xlabel('Proportion of Component (Mean ± SE)', fontsize=11)
    ax.set_ylabel(category.capitalize(), fontsize=11)
    ax.set_title(f'PC{component_idx+1} - {category.capitalize()} Composition (with variability)', 
                 fontsize=12)
    ax.set_xlim([0, 1])
    ax.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved: {output_path.name}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--iteration', type=int, default=None, 
                       help='Process specific iteration only (default: all)')
    args = parser.parse_args()
    
    # Paths
    indices_folder = Path("/data/users/ltucker/influenzaData/pipeline/output/tensor/subsample_indices")
    parquet_folder = Path("/data/users/ltucker/influenzaData/pipeline/output/1_distances")
    output_folder = Path("/data/users/ltucker/influenzaData/pipeline/output/tensor/svd_loadings_analysis")
    output_folder.mkdir(parents=True, exist_ok=True)
    
    # Fixed to 50% subsamples only
    percentage = 50
    
    # Get all 50% iterations from master index
    master_index = pd.read_csv(indices_folder / 'master_index.csv')
    iterations_50 = master_index[master_index['percentage'] == percentage]
    
    if len(iterations_50) == 0:
        print(f"ERROR: No {percentage}% subsamples found in master index")
        return
    
    print(f"Found {len(iterations_50)} iterations for {percentage}% subsamples")
    print(iterations_50)
    
    # Filter to specific iteration if requested
    if args.iteration is not None:
        iterations_50 = iterations_50[iterations_50['iteration'] == args.iteration]
        if len(iterations_50) == 0:
            print(f"ERROR: Iteration {args.iteration} not found for {percentage}%")
            return
    
    print(f"\nProcessing {len(iterations_50)} iteration(s)")
    print("="*60)
    
    # Storage for cross-iteration analysis
    all_loadings_mode0 = []
    all_loadings_mode1 = []
    all_loadings_mode2 = []
    all_metadata = []
    
    # Process each iteration
    for idx, row in iterations_50.iterrows():
        iteration = int(row['iteration'])
        seed = int(row['seed'])
        
        print(f"\n{'='*60}")
        print(f"Processing iteration {iteration}, seed {seed}")
        print(f"{'='*60}")
        
        # Load tensor
        indices_file = indices_folder / f"indices_pct{percentage:03d}_iter{iteration:02d}_seed{seed:04d}.npy"
        print(f"Loading tensor from indices: {indices_file.name}")
        
        tensor, selected_indices, sample_names, segment_names = load_tensor_with_indices(
            indices_file, parquet_folder
        )
        
        print(f"Tensor shape: {tensor.shape}")
        print(f"Segments: {segment_names}")
        print(f"Number of samples: {len(sample_names)}")
        
        # Parse metadata
        metadata = parse_metadata(sample_names)
        print(f"\nMetadata summary:")
        print(f"  Segments: {metadata['segment'].nunique()} unique")
        print(f"  Countries: {metadata['country'].nunique()} unique")
        print(f"  Continents: {metadata['continent'].nunique()} unique")
        print(f"  Years: {metadata['year'].nunique()} unique")
        
        # Unfold and perform SVD
        print("\nUnfolding tensor and performing SVD...")
        
        # Mode 0: Segments
        tensor_mode0 = tl.unfold(tensor, mode=0)
        print(f"Mode 0 shape: {tensor_mode0.shape}")
        U0, W0, V0 = np.linalg.svd(tensor_mode0, full_matrices=False)
        print(f"Mode 0 SVD complete - {len(W0)} components")
        
        # Mode 1: Samples (rows)
        tensor_mode1 = tl.unfold(tensor, mode=1)
        print(f"Mode 1 shape: {tensor_mode1.shape}")
        U1, W1, V1 = np.linalg.svd(tensor_mode1, full_matrices=False)
        print(f"Mode 1 SVD complete - {len(W1)} components")
        
        # Mode 2: Samples (cols)
        tensor_mode2 = tl.unfold(tensor, mode=2)
        print(f"Mode 2 shape: {tensor_mode2.shape}")
        U2, W2, V2 = np.linalg.svd(tensor_mode2, full_matrices=False)
        print(f"Mode 2 SVD complete - {len(W2)} components")
        
        # Save SVD matrices
        base_name = f"svd_pct{percentage:03d}_iter{iteration:02d}_seed{seed:04d}"
        
        np.save(output_folder / f"{base_name}_mode0_U.npy", U0)
        np.save(output_folder / f"{base_name}_mode0_W.npy", W0)
        np.save(output_folder / f"{base_name}_mode0_V.npy", V0)
        
        np.save(output_folder / f"{base_name}_mode1_U.npy", U1)
        np.save(output_folder / f"{base_name}_mode1_W.npy", W1)
        np.save(output_folder / f"{base_name}_mode1_V.npy", V1)
        
        np.save(output_folder / f"{base_name}_mode2_U.npy", U2)
        np.save(output_folder / f"{base_name}_mode2_W.npy", W2)
        np.save(output_folder / f"{base_name}_mode2_V.npy", V2)
        
        print(f"\nSaved SVD matrices: {base_name}_mode*")
        
        # Store for cross-iteration analysis
        all_loadings_mode0.append(U0)
        all_loadings_mode1.append(U1)
        all_loadings_mode2.append(U2)
        all_metadata.append(metadata)
        
        # Generate plots for this iteration
        print("\nGenerating loading plots...")
        
        # Mode 0: All segments (only 8, show all)
        plot_loading_heatmap(
            U0, 
            segment_names,
            f'Mode 0: Segment Loadings (Iter {iteration})',
            output_folder / f"loadings_mode0_segments_{base_name}.png"
        )
        
        # Mode 1: Top 5 samples
        plot_loading_heatmap(
            U1,
            sample_names,
            f'Mode 1: Sample Loadings - Top 5 (Iter {iteration})',
            output_folder / f"loadings_mode1_samples_top5_{base_name}.png",
            top_n=5
        )
        
        # Mode 2: Top 5 samples
        plot_loading_heatmap(
            U2,
            sample_names,
            f'Mode 2: Sample Loadings - Top 5 (Iter {iteration})',
            output_folder / f"loadings_mode2_samples_top5_{base_name}.png",
            top_n=5
        )
        
        # Aggregate loadings by metadata categories (Mode 1 only, since Mode 2 is same samples)
        print("\nAggregating loadings by metadata categories...")
        
        proportions = {}
        for category in ['segment', 'year', 'continent', 'country']:
            proportions[category] = aggregate_loadings_by_category(U1, metadata, category)
            
            # Save proportions
            csv_filename = f"proportions_{category}_{base_name}.csv"
            proportions[category].to_csv(output_folder / csv_filename)
            print(f"Saved: {csv_filename}")
        
        # Plot composition for first 5 components
        n_components_to_plot = min(5, U1.shape[1])
        
        for comp_idx in range(n_components_to_plot):
            plot_stacked_bar_proportions(
                proportions,
                comp_idx,
                output_folder / f"composition_PC{comp_idx+1}_{base_name}.png"
            )
        
        # Plot composition grid
        plot_loading_composition_grid(
            proportions,
            n_components_to_plot,
            output_folder / f"composition_grid_{base_name}.png"
        )
    
    # Cross-iteration analysis (only if multiple iterations)
    if len(iterations_50) > 1:
        print("\n" + "="*60)
        print("Performing cross-iteration analysis...")
        print("="*60)
        
        for category in ['segment', 'year', 'continent', 'country']:
            print(f"\nAnalyzing {category}...")
            
            # Calculate mean and SE (using Mode 1 loadings)
            mean_props, se_props = calculate_loading_statistics(
                all_loadings_mode1, 
                all_metadata, 
                category
            )
            
            # Save statistics
            mean_props.to_csv(output_folder / f"mean_proportions_{category}_pct{percentage:03d}.csv")
            se_props.to_csv(output_folder / f"se_proportions_{category}_pct{percentage:03d}.csv")
            print(f"Saved mean and SE for {category}")
            
            # Plot first 5 components with error bars
            n_components_to_plot = min(5, mean_props.shape[1])
            for comp_idx in range(n_components_to_plot):
                plot_proportions_with_error(
                    mean_props,
                    se_props,
                    comp_idx,
                    category,
                    output_folder / f"composition_with_error_{category}_PC{comp_idx+1}_pct{percentage:03d}.png"
                )
    
    print("\n" + "="*60)
    print("Complete!")
    print(f"Output saved to: {output_folder}")
    print("="*60)

if __name__ == "__main__":
    main()