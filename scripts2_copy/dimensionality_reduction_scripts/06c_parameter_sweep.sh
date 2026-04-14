#!/bin/bash

#SBATCH --job-name=consensus_tree
#SBATCH --output=/data/users/ltucker/influenzaData/H5N1_pipeline/log/consensus/consensus_%j.out
#SBATCH --error=/data/users/ltucker/influenzaData/H5N1_pipeline/log/consensus/consensus_%j.err
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
OUTPUT_DIR="$WORKDIR/output/consensus"
CONTAINER="/data/users/ltucker/influenzaData/pipeline/jupyter-tensor_15.sif"

mkdir -p "$OUTPUT_DIR"

export TREE_DIR OUTPUT_DIR

# Step 1: Rename tips using Python inside container
apptainer exec "$CONTAINER" python3 << 'PYEOF'

import os
import sys
import pandas as pd
from Bio import Phylo

sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', buffering=1)
sys.stderr = os.fdopen(sys.stderr.fileno(), 'w', buffering=1)

tree_dir = os.environ["TREE_DIR"]
output_dir = os.environ["OUTPUT_DIR"]

def make_common_name(label):
    """Remove segment name (pos 2) and segment number (pos 3) from header."""
    parts = label.split("|")
    # [0]subtype, [1]strain, [2]seg_name, [3]seg_num, [4]epi_id, [5]country, [6]continent, [7]year
    common_parts = [parts[0], parts[1]] + parts[4:]
    return "|".join(common_parts)

all_mappings = []

for seg in range(1, 9):
    treefile = f"{tree_dir}/segment_{seg}.treefile"
    print(f"Loading {treefile}", flush=True)
    tree = Phylo.read(treefile, "newick")
    tips = tree.get_terminals()

    for tip in tips:
        original = tip.name
        common = make_common_name(original)
        parts = original.split("|")

        all_mappings.append({
            "segment": seg,
            "segment_name": parts[2] if len(parts) > 2 else "",
            "iqtree_header": original,
            "common_name": common,
            "epi_id": parts[4] if len(parts) > 4 else "",
            "country": parts[5] if len(parts) > 5 else "",
            "continent": parts[6] if len(parts) > 6 else "",
            "year": parts[7] if len(parts) > 7 else ""
        })

        tip.name = common

    print(f"Segment {seg}: {len(tips)} tips renamed", flush=True)
    Phylo.write(tree, f"{output_dir}/segment_{seg}_renamed.treefile", "newick")

# Save mapping
df = pd.DataFrame(all_mappings)
df.to_csv(f"{output_dir}/tip_label_mapping.csv", index=False)
print(f"\nMapping saved: {len(df)} rows", flush=True)
print(f"Unique common names: {df['common_name'].nunique()}", flush=True)

# Combine renamed trees
with open(f"{output_dir}/all_renamed_trees.nwk", "w") as out:
    for seg in range(1, 9):
        with open(f"{output_dir}/segment_{seg}_renamed.treefile") as f:
            out.write(f.read())

print(f"Combined trees written to {output_dir}/all_renamed_trees.nwk", flush=True)

PYEOF

if [ $? -ne 0 ]; then
    echo "ERROR: Python renaming step failed" >&2
    exit 1
fi

echo "Python renaming complete"

# Step 2: Build consensus tree
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