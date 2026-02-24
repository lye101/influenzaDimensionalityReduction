#!/bin/bash
#SBATCH --job-name=tucker
#SBATCH --output=/data/users/ltucker/influenzaData/pipeline/log/tucker/tucker_%A_%a.log
#SBATCH --error=/data/users/ltucker/influenzaData/pipeline/log/tucker/tucker_%A_%a.err
#SBATCH --array=0-28
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=150G
#SBATCH --partition=pibu_el8

echo "Job ID: $SLURM_JOB_ID"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Start Time: $(date)"

mkdir -p /data/users/ltucker/influenzaData/pipeline/log/tucker

# Define rank to use (adjust as needed)
RANK="2,3,3"

# Read master index
MASTER_INDEX="/data/users/ltucker/influenzaData/pipeline/output/tensor/subsample_indices/master_index.csv"

# Check if master index exists
if [ ! -f "$MASTER_INDEX" ]; then
    echo "ERROR: Master index not found at $MASTER_INDEX"
    echo "Please run 07a_generate_indices.sh first"
    exit 1
fi

echo "Reading from master index: $MASTER_INDEX"
echo "Total lines in file:"
wc -l "$MASTER_INDEX"

# Skip header (line 1) and get the appropriate data line
# Array index 0 corresponds to line 2 (first data line)
LINE_NUM=$((SLURM_ARRAY_TASK_ID + 2))

echo "Reading line number: $LINE_NUM"

# Read the line
PARAMS=$(sed -n "${LINE_NUM}p" "$MASTER_INDEX")

echo "Raw line content: '$PARAMS'"

# Check if line is empty
if [ -z "$PARAMS" ]; then
    echo "ERROR: No data found at line $LINE_NUM"
    echo "Array index $SLURM_ARRAY_TASK_ID may be out of range"
    exit 1
fi

# Parse CSV - more robust method
IFS=',' read -r RATIO PCT ITER SEED <<< "$PARAMS"

echo "Parsed values:"
echo "  Ratio: '$RATIO'"
echo "  Percentage: '$PCT'"
echo "  Iteration: '$ITER'"
echo "  Seed: '$SEED'"

# Validate that we got values
if [ -z "$PCT" ] || [ -z "$ITER" ] || [ -z "$SEED" ]; then
    echo "ERROR: Failed to parse CSV line"
    echo "Expected format: subsample_ratio,percentage,iteration,seed"
    exit 1
fi

echo ""
echo "Processing: ${PCT}% subsample, iteration ${ITER}, seed ${SEED}"
echo "Rank: ${RANK}"
echo ""

SCRIPT_PATH="/data/users/ltucker/influenzaData/pipeline/scripts/07d_tucker_decomposition.py"

singularity exec --cleanenv \
    --bind /data \
    /data/users/ltucker/influenzaData/pipeline/jupyter-tensor_12.sif \
    python -u "$SCRIPT_PATH" \
        --percentage "$PCT" \
        --iteration "$ITER" \
        --seed "$SEED" \
        --rank "$RANK"

EXIT_CODE=$?

echo ""
echo "Job completed at $(date) with exit code $EXIT_CODE"
exit $EXIT_CODE