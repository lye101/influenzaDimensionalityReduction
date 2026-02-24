#!/bin/bash

#SBATCH --time=2-00:00:00
#SBATCH --mem=200GB
#SBATCH --cpus-per-task=10
#SBATCH --job-name=DistanceMatrix
#SBATCH --partition=pibu_el8
#SBATCH --array=1-8
#SBATCH --output=/data/users/ltucker/influenzaData/pipeline/log/distance_matrix_%A_%a.out
#SBATCH --error=/data/users/ltucker/influenzaData/pipeline/log/distance_matrix_%A_%a.err

WORKDIR=/data/users/ltucker/influenzaData/pipeline
SEGMENTS_DIR="$WORKDIR/output/1_segments"
DISTANCES_DIR="$WORKDIR/output/1_distances"

# get dependencies
eval "$(conda shell.bash hook)"
conda activate distanceMatrix

mkdir -p "$DISTANCES_DIR"

# Create array of segment files
segments=("$SEGMENTS_DIR"/*.fa)

# Get the segment for this array task (array index starts at 1, bash arrays start at 0)
segment="${segments[$SLURM_ARRAY_TASK_ID-1]}"

# Extract base name
base=$(basename "$segment" .fa)

echo "Processing segment: $base (Array task $SLURM_ARRAY_TASK_ID)"
echo "Input file: $segment"

# Run famsa on this segment
famsa \
    -dist_export \
    -square_matrix \
    -v \
    -t 10 \
    -gz \
    "$segment" \
    "$DISTANCES_DIR/distance_${base}.csv"

# Check if successful
if [ $? -eq 0 ]; then
    echo "Successfully completed: $base"
else
    echo "ERROR: Failed to process $base" >&2
    exit 1
fi

conda deactivate