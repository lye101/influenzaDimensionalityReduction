#!/bin/bash

#SBATCH --job-name=iqtree
#SBATCH --output=/data/users/ltucker/influenzaData/H5N1_pipeline/log/iqtree/iqtree_%A_%a.out
#SBATCH --error=/data/users/ltucker/influenzaData/H5N1_pipeline/log/iqtree/iqtree_%A_%a.err
#SBATCH --array=1-8
#SBATCH --time=5-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=200G
#SBATCH --partition=pibu_el8

echo "Job ID: $SLURM_JOB_ID"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Start Time: $(date)"

# Paths
WORKDIR="/data/users/ltucker/influenzaData/H5N1_pipeline"
IQTREE="$WORKDIR/iqtree-2.4.0-Linux-intel/bin/iqtree2"
ALIGNED_DIR="$WORKDIR/output/mafft_aligned"
OUTPUT_DIR="$WORKDIR/output/iqtree"
LOG_DIR="$WORKDIR/log/iqtree"

mkdir -p "$OUTPUT_DIR"
mkdir -p "$LOG_DIR"

# Input alignment for this array task
SEGMENT=$SLURM_ARRAY_TASK_ID
INPUT="$ALIGNED_DIR/aligned_${SEGMENT}.fa"

echo "Processing segment: $SEGMENT"
echo "Input file: $INPUT"

# Check input exists
if [ ! -f "$INPUT" ]; then
    echo "ERROR: Input file not found: $INPUT" >&2
    exit 1
fi

# Run IQ-TREE
$IQTREE \
    -s "$INPUT" \
    -st DNA \
    -m MFP \
    -B 1000 -bnni \
    --alrt 1000 \
    --safe \
    -T $SLURM_CPUS_PER_TASK \
    --seed 12345 \
    --prefix "$OUTPUT_DIR/segment_${SEGMENT}" \
#    -redo

# Check if successful
if [ $? -eq 0 ]; then
    echo "Successfully completed segment: $SEGMENT"
    echo "Best model chosen by ModelFinder:"
    grep "Best-fit model" "$OUTPUT_DIR/segment_${SEGMENT}.iqtree" || true
else
    echo "ERROR: IQ-TREE failed for segment $SEGMENT" >&2
    exit 1
fi

echo "End Time: $(date)"