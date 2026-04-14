#!/bin/bash

#SBATCH --job-name=consensus_tree
#SBATCH --output=/data/users/ltucker/influenzaData/H5N1_pipeline/log/consensus_%j.out
#SBATCH --error=/data/users/ltucker/influenzaData/H5N1_pipeline/log/consensus_%j.err
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --partition=pibu_el8

set -e

echo "Job ID: $SLURM_JOB_ID"
echo "Start Time: $(date)"

WORKDIR="/data/users/ltucker/influenzaData/H5N1_pipeline"
IQTREE="$WORKDIR/iqtree-2.4.0-Linux-intel/bin/iqtree2"
TREE_DIR="$WORKDIR/output/iqtree"
OUTPUT_DIR="$WORKDIR/output/IQtree_consensus"
CONTAINER="/data/users/ltucker/influenzaData/pipeline/jupyter-tensor_15.sif"

mkdir -p "$OUTPUT_DIR"

export TREE_DIR OUTPUT_DIR

apptainer exec "$CONTAINER" python3 << 'PYEOF'

import os
import sys
import re
import pandas as pd

sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', buffering=1)
sys.stderr = os.fdopen(sys.stderr.fileno(), 'w', buffering=1)

tree_dir = os.environ["TREE_DIR"]
output_dir = os.environ["OUTPUT_DIR"]

# Regex to extract EPI_ISL_XXXXX from anywhere in a tip label
epi_pattern = re.compile(r'EPI_ISL_\d+')

# Tip labels sit between ( or , and : in Newick
tip_pattern = re.compile(r'(?<=[\(,])([^(:,\)]+)(?=:)')

all_mappings = []
epi_sets = {}
newicks = {}
rename_maps = {}

# Pass 1: Load trees, extract EPI IDs
for seg in range(1, 9):
    with open(f"{tree_dir}/segment_{seg}.treefile") as f:
        newick = f.read().strip()

    tips = tip_pattern.findall(newick)
    print(f"Segment {seg}: {len(tips)} tips found", flush=True)

    rename_map = {}
    no_epi_count = 0

    for tip in tips:
        match = epi_pattern.search(tip)
        if match:
            epi_id = match.group()
            rename_map[tip] = epi_id
            all_mappings.append({
                "segment": seg,
                "iqtree_header": tip,
                "epi_id": epi_id,
            })
        else:
            no_epi_count += 1

    if no_epi_count > 0:
        print(f"  WARNING: {no_epi_count} tips without EPI ID in segment {seg}", flush=True)

    rename_maps[seg] = rename_map
    epi_sets[seg] = set(rename_map.values())
    newicks[seg] = newick

# Find shared EPI IDs
shared = epi_sets[1]
for seg in range(2, 9):
    shared = shared & epi_sets[seg]

print(f"\nTaxa per segment: {[len(s) for s in epi_sets.values()]}", flush=True)
print(f"Shared across all 8: {len(shared)}", flush=True)

if len(shared) == 0:
    raise ValueError("No shared EPI IDs found!")

dropped = sum(len(s) for s in epi_sets.values()) - len(shared) * 8
if dropped > 0:
    print(f"Will prune {dropped} total tips across all segments", flush=True)

# Pass 2: Rename tips to EPI IDs, prune non-shared
for seg in range(1, 9):
    newick = newicks[seg]

    # Rename: longest first to avoid partial matches
    for old_name in sorted(rename_maps[seg].keys(), key=len, reverse=True):
        epi_id = rename_maps[seg][old_name]
        if epi_id in shared:
            newick = newick.replace(old_name, epi_id)
        else:
            # Replace with empty to mark for pruning — but Newick pruning via
            # string manipulation is fragile, so instead we use IQ-TREE's
            # built-in pruning: write a taxa list file
            pass

    # For non-shared taxa, simpler to rename everything then let IQ-TREE handle it
    # So rename ALL tips to EPI IDs first
    newick = newicks[seg]
    for old_name in sorted(rename_maps[seg].keys(), key=len, reverse=True):
        epi_id = rename_maps[seg][old_name]
        newick = newick.replace(old_name, epi_id)

    with open(f"{output_dir}/segment_{seg}_renamed.treefile", "w") as f:
        f.write(newick + "\n")

    print(f"Segment {seg}: renamed and saved", flush=True)

# Write shared taxa list (for IQ-TREE pruning if needed)
with open(f"{output_dir}/shared_taxa.txt", "w") as f:
    for epi_id in sorted(shared):
        f.write(epi_id + "\n")

# Combine renamed trees
with open(f"{output_dir}/all_renamed_trees.nwk", "w") as out:
    for seg in range(1, 9):
        with open(f"{output_dir}/segment_{seg}_renamed.treefile") as f:
            out.write(f.read())

# Save mapping with metadata extracted from segment 1 headers
# (use segment 1 as reference for strain info)
mapping_df = pd.DataFrame(all_mappings)
mapping_df.to_csv(f"{output_dir}/tip_label_mapping.csv", index=False)
print(f"\nMapping saved: {len(mapping_df)} rows", flush=True)
print(f"Shared taxa list saved: {len(shared)} EPI IDs", flush=True)
print(f"Combined trees written to {output_dir}/all_renamed_trees.nwk", flush=True)

PYEOF

if [ $? -ne 0 ]; then
    echo "ERROR: Python renaming step failed" >&2
    exit 1
fi

echo "Python renaming complete"

# If pruning is needed, use IQ-TREE to prune each tree first
# Check if all segments have same count
COUNTS=$(grep -c "EPI_ISL_" "$OUTPUT_DIR"/segment_*_renamed.treefile | cut -d: -f2 | sort -u | wc -l)
if [ "$COUNTS" -gt 1 ]; then
    echo "Taxa counts differ — pruning to shared set"
    for SEG in $(seq 1 8); do
        $IQTREE \
            -t "$OUTPUT_DIR/segment_${SEG}_renamed.treefile" \
            --keep-ident \
            -o "$OUTPUT_DIR/segment_${SEG}_pruned.treefile" \
            --taxa "$OUTPUT_DIR/shared_taxa.txt" \
            -T 1 2>/dev/null || true
    done
    # Recombine pruned trees
    cat "$OUTPUT_DIR"/segment_*_pruned.treefile > "$OUTPUT_DIR/all_renamed_trees.nwk"
    echo "Pruned trees recombined"
fi

# Build consensus tree
$IQTREE \
    -t "$OUTPUT_DIR/all_renamed_trees.nwk" \
    --con-tree \
    --prefix "$OUTPUT_DIR/consensus" \
    -T $SLURM_CPUS_PER_TASK

if [ $? -eq 0 ]; then
    echo "Consensus tree written to: $OUTPUT_DIR/consensus.contree"
else
    echo "ERROR: Consensus tree construction failed" >&2
    exit 1
fi

echo "End Time: $(date)"