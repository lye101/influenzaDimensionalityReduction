import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import defaultdict
import argparse

def load_svd_results(output_folder, percentage=None):
    """
    Load all SVD results, optionally filtered by percentage
    
    Returns:
        dict: {percentage: [list of dataframes]}
    """
    results = defaultdict(list)
    
    # Find all SVD CSV files
    csv_files = sorted(output_folder.glob("svd_values_pct*.csv"))
    
    if len(csv_files) == 0:
        print(f"WARNING: No SVD result files found in {output_folder}")
        return results
    
    print(f"Found {len(csv_files)} SVD result files")
    
    for csv_file in csv_files:
        # Parse filename: svd_values_pct{XXX}_iter{YY}_seed{ZZZZ}.csv
        filename = csv_file.stem
        parts = filename.split('_')
        
        # Extract percentage
        pct_str = parts[2]  # pct005, pct010, etc.
        pct = int(pct_str.replace('pct', ''))
        
        # Extract iteration and seed
        iter_str = parts[3]  # iter00, iter01, etc.
        iteration = int(iter_str.replace('iter', ''))
        
        seed_str = parts[4]  # seed0000, seed1000, etc.
        seed = int(seed_str.replace('seed', ''))
        
        # Filter by percentage if specified
        if percentage is not None and pct != percentage:
            continue
        
        # Load data
        df = pd.read_csv(csv_file)
        df['iteration'] = iteration
        df['seed'] = seed
        df['percentage'] = pct
        
        results[pct].append(df)
    
    print(f"Loaded data for percentages: {sorted(results.keys())}")
    for pct in sorted(results.keys()):
        print(f"  {pct}%: {len(results[pct])} iterations")
    
    return results

def plot_combined_scree_single_percentage(results_dict, percentage, output_folder):
    """
    Create combined scree plot for a single percentage across all iterations
    """
    if percentage not in results_dict:
        print(f"WARNING: No data found for {percentage}%")
        return
    
    dfs = results_dict[percentage]
    n_iterations = len(dfs)
    
    print(f"\nCreating combined plot for {percentage}% ({n_iterations} iterations)")
    
    # Create figure
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    
    # Color map for iterations
    colors = plt.cm.tab10(np.linspace(0, 1, n_iterations))
    
    # Plot each mode
    modes = ['mode0', 'mode1', 'mode2']
    mode_names = ['Mode 0: Segments', 'Mode 1: Samples (Rows)', 'Mode 2: Samples (Cols)']
    
    for mode_idx, (mode, mode_name) in enumerate(zip(modes, mode_names)):
        # Singular values (top row)
        ax_sv = axes[0, mode_idx]
        # Cumulative variance (bottom row)
        ax_cum = axes[1, mode_idx]
        
        # Collect data for mean/std calculation
        all_sv = []
        all_cum = []
        
        for iter_idx, df in enumerate(dfs):
            sv_col = f'{mode}_singular_value'
            cum_col = f'{mode}_cumulative_variance'
            
            # Get data (drop NaN values)
            sv_data = df[sv_col].dropna()
            cum_data = df[cum_col].dropna()
            
            all_sv.append(sv_data.values)
            all_cum.append(cum_data.values)
            
            # Plot individual lines with transparency
            components = range(1, len(sv_data) + 1)
            
            # Limit display for modes 1 and 2
            if mode_idx > 0:
                max_show = min(50, len(sv_data))
                components = range(1, max_show + 1)
                sv_data = sv_data[:max_show]
                cum_data = cum_data[:max_show]
            
            label = f"Seed {df['seed'].iloc[0]}" if n_iterations <= 10 else None
            
            ax_sv.plot(components, sv_data, 'o-', 
                      color=colors[iter_idx], 
                      alpha=0.4, 
                      markersize=3,
                      linewidth=1,
                      label=label)
            
            ax_cum.plot(components, cum_data, 'o-', 
                       color=colors[iter_idx], 
                       alpha=0.4,
                       markersize=3,
                       linewidth=1,
                       label=label)
        
        # Calculate and plot mean line
        min_len = min(len(x) for x in all_sv)
        sv_array = np.array([x[:min_len] for x in all_sv])
        cum_array = np.array([x[:min_len] for x in all_cum])
        
        sv_mean = np.mean(sv_array, axis=0)
        cum_mean = np.mean(cum_array, axis=0)
        
        # Limit display for modes 1 and 2
        if mode_idx > 0:
            max_show = min(50, len(sv_mean))
            sv_mean = sv_mean[:max_show]
            cum_mean = cum_mean[:max_show]
            min_len = max_show
        
        components_mean = range(1, min_len + 1)
        
        ax_sv.plot(components_mean, sv_mean, 'k-', 
                  linewidth=0.5, 
                  label='Mean',
                  zorder=100)
        
        ax_cum.plot(components_mean, cum_mean, 'k-', 
                   linewidth=0.5,
                   label='Mean',
                   zorder=100)
        
        # Format singular value plot
        ax_sv.set_xlabel('Component', fontsize=11)
        ax_sv.set_ylabel('Singular Value', fontsize=11)
        title_suffix = f' (first {max_show})' if mode_idx > 0 else ''
        ax_sv.set_title(f'{mode_name} - Singular Values{title_suffix}', fontsize=12)
        ax_sv.grid(True, alpha=0.3)
        if n_iterations <= 10:
            ax_sv.legend(fontsize=8, loc='best')
        
        # Format cumulative variance plot
        ax_cum.set_xlabel('Number of Components', fontsize=11)
        ax_cum.set_ylabel('Cumulative Variance Explained', fontsize=11)
        ax_cum.set_title(f'{mode_name} - Cumulative Variance{title_suffix}', fontsize=12)
        ax_cum.axhline(y=0.90, color='r', linestyle='--', linewidth=2, label='90%', zorder=101)
        ax_cum.axhline(y=0.95, color='orange', linestyle='--', linewidth=2, label='95%', zorder=101)
        ax_cum.grid(True, alpha=0.3)
        ax_cum.legend(fontsize=8, loc='best')
        ax_cum.set_ylim([0.85, 1.05])
    
    plt.suptitle(f'Combined Scree Plots - {percentage}% Subsample ({n_iterations} iterations)', 
                 fontsize=16, y=0.995)
    plt.tight_layout()
    
    # Save plot
    plot_filename = f"combined_scree_plot_pct{percentage:03d}.png"
    plt.savefig(output_folder / plot_filename, dpi=300, bbox_inches='tight')
    print(f"Saved: {plot_filename}")
    plt.close()

def plot_combined_across_percentages(results_dict, output_folder):
    """
    Create mega-plot showing mean cumulative variance across all percentages
    """
    if len(results_dict) == 0:
        print("WARNING: No data to plot across percentages")
        return
    
    print(f"\nCreating combined plot across {len(results_dict)} percentages")
    
    # Create figure - show only cumulative variance for clarity
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    
    modes = ['mode0', 'mode1', 'mode2']
    mode_names = ['Mode 0: Segments', 'Mode 1: Samples (Rows)', 'Mode 2: Samples (Cols)']
    
    # Color map for percentages
    percentages = sorted(results_dict.keys())
    colors = plt.cm.viridis(np.linspace(0, 1, len(percentages)))
    
    for mode_idx, (mode, mode_name) in enumerate(zip(modes, mode_names)):
        ax = axes[mode_idx]
        cum_col = f'{mode}_cumulative_variance'
        
        for pct_idx, pct in enumerate(percentages):
            dfs = results_dict[pct]
            
            # Collect cumulative variance across iterations
            all_cum = []
            for df in dfs:
                cum_data = df[cum_col].dropna()
                all_cum.append(cum_data.values)
            
            # Calculate mean
            min_len = min(len(x) for x in all_cum)
            cum_array = np.array([x[:min_len] for x in all_cum])
            cum_mean = np.mean(cum_array, axis=0)
            cum_std = np.std(cum_array, axis=0)
            
            # Limit display for modes 1 and 2
            if mode_idx > 0:
                max_show = min(50, len(cum_mean))
                cum_mean = cum_mean[:max_show]
                cum_std = cum_std[:max_show]
                min_len = max_show
            
            components = range(1, min_len + 1)
            
            # Plot mean line
            ax.plot(components, cum_mean, 
                   color=colors[pct_idx],
                   linewidth=0.5,
                   label=f'{pct}%',
                   marker='o', # if len(components) < 20 else None,
                   markersize=4)
            
            # Optional: add shaded std region (commented out to avoid clutter)
            # ax.fill_between(components, 
            #                  cum_mean - cum_std, 
            #                  cum_mean + cum_std,
            #                  color=colors[pct_idx],
            #                  alpha=0.1)
        
        # Format plot
        ax.set_xlabel('Number of Components', fontsize=12)
        ax.set_ylabel('Cumulative Variance Explained', fontsize=12)
        title_suffix = ' (first 50)' if mode_idx > 0 else ''
        ax.set_title(f'{mode_name}{title_suffix}', fontsize=13)
        ax.axhline(y=0.90, color='r', linestyle='--', linewidth=2, alpha=0.7, label='90%')
        ax.axhline(y=0.95, color='orange', linestyle='--', linewidth=2, alpha=0.7, label='95%')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9, loc='best', ncol=2)
        ax.set_ylim([0.85, 1.05])
    
    plt.suptitle('Cumulative Variance Across All Percentages (Mean per percentage)', 
                 fontsize=16)
    plt.tight_layout()
    
    # Save plot
    plot_filename = "combined_scree_plot_all_percentages.png"
    plt.savefig(output_folder / plot_filename, dpi=300, bbox_inches='tight')
    print(f"Saved: {plot_filename}")
    plt.close()

def plot_mega_grid_all(results_dict, output_folder):
    """
    Create mega-grid showing all percentages × all modes
    """
    if len(results_dict) == 0:
        print("WARNING: No data for mega-grid")
        return
    
    percentages = sorted(results_dict.keys())
    n_pct = len(percentages)
    
    print(f"\nCreating mega-grid for {n_pct} percentages")
    
    # Create figure - rows = percentages, cols = modes
    fig, axes = plt.subplots(n_pct, 3, figsize=(18, 4 * n_pct))
    
    # Handle single percentage case
    if n_pct == 1:
        axes = axes.reshape(1, -1)
    
    modes = ['mode0', 'mode1', 'mode2']
    mode_names = ['Segments', 'Rows', 'Cols']
    
    for pct_idx, pct in enumerate(percentages):
        dfs = results_dict[pct]
        
        for mode_idx, (mode, mode_name) in enumerate(zip(modes, mode_names)):
            ax = axes[pct_idx, mode_idx]
            cum_col = f'{mode}_cumulative_variance'
            
            # Collect and plot all iterations
            all_cum = []
            for df in dfs:
                cum_data = df[cum_col].dropna()
                all_cum.append(cum_data.values)
                
                # Limit display
                if mode_idx > 0:
                    max_show = min(50, len(cum_data))
                    cum_data = cum_data[:max_show]
                
                components = range(1, len(cum_data) + 1)
                ax.plot(components, cum_data, alpha=0.3, linewidth=1)
            
            # Calculate and plot mean
            min_len = min(len(x) for x in all_cum)
            cum_array = np.array([x[:min_len] for x in all_cum])
            cum_mean = np.mean(cum_array, axis=0)
            
            if mode_idx > 0:
                max_show = min(50, len(cum_mean))
                cum_mean = cum_mean[:max_show]
                min_len = max_show
            
            components_mean = range(1, min_len + 1)
            ax.plot(components_mean, cum_mean, 'k-', linewidth=0.5, label='Mean')
            
            # Format
            ax.axhline(y=0.90, color='r', linestyle='--', alpha=0.7)
            ax.axhline(y=0.95, color='orange', linestyle='--', alpha=0.7)
            ax.grid(True, alpha=0.3)
            ax.set_ylim([0.85, 1.05])
            
            # Labels
            if pct_idx == n_pct - 1:
                ax.set_xlabel('Components', fontsize=10)
            if mode_idx == 0:
                ax.set_ylabel(f'{pct}%\nCum. Var.', fontsize=10)
            if pct_idx == 0:
                ax.set_title(f'Mode: {mode_name}', fontsize=11)
    
    plt.suptitle('Cumulative Variance: All Percentages × All Modes', fontsize=16)
    plt.tight_layout()
    
    plot_filename = "mega_grid_all_percentages_modes.png"
    plt.savefig(output_folder / plot_filename, dpi=300, bbox_inches='tight')
    print(f"Saved: {plot_filename}")
    plt.close()

def main():
    parser = argparse.ArgumentParser(description='Combine scree plots from SVD results')
    parser.add_argument('--percentage', type=int, default=None, 
                       help='Process only this percentage (default: process all)')
    parser.add_argument('--skip-individual', action='store_true',
                       help='Skip individual percentage plots')
    parser.add_argument('--skip-combined', action='store_true',
                       help='Skip combined across percentages plot')
    parser.add_argument('--skip-megagrid', action='store_true',
                       help='Skip mega-grid plot')
    args = parser.parse_args()
    
    # Paths
    input_folder = Path("/data/users/ltucker/influenzaData/pipeline/output/tensor/svd_scree_plots")
    output_folder = Path("/data/users/ltucker/influenzaData/pipeline/output/tensor/svd_combined_plots")
    output_folder.mkdir(parents=True, exist_ok=True)
    
    print("="*60)
    print("Combined Scree Plot Generator")
    print("="*60)
    
    # Load all results
    results_dict = load_svd_results(input_folder, percentage=args.percentage)
    
    if len(results_dict) == 0:
        print("\nERROR: No SVD results found. Make sure 07b has completed.")
        return
    
    # Generate plots based on flags
    if not args.skip_individual:
        print("\n" + "="*60)
        print("Generating individual percentage plots...")
        print("="*60)
        
        percentages_to_plot = [args.percentage] if args.percentage else sorted(results_dict.keys())
        
        for pct in percentages_to_plot:
            plot_combined_scree_single_percentage(results_dict, pct, output_folder)
    
    if not args.skip_combined and args.percentage is None:
        print("\n" + "="*60)
        print("Generating combined across percentages plot...")
        print("="*60)
        plot_combined_across_percentages(results_dict, output_folder)
    
    if not args.skip_megagrid and args.percentage is None:
        print("\n" + "="*60)
        print("Generating mega-grid plot...")
        print("="*60)
        plot_mega_grid_all(results_dict, output_folder)
    
    print("\n" + "="*60)
    print("Complete!")
    print(f"Output saved to: {output_folder}")
    print("="*60)

if __name__ == "__main__":
    main()