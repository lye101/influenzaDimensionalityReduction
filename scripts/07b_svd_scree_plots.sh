#!/bin/bash
#SBATCH --job-name=svd_scree
#SBATCH --output=/data/users/ltucker/influenzaData/pipeline/log/svd/scree_%A_%a.log
#SBATCH --error=/data/users/ltucker/influenzaData/pipeline/log/svd/scree_%A_%a.err
#SBATCH --array=0-28
#SBATCH --time=7-03:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=120G
#SBATCH --partition=pibu_el8

echo "Job ID: $SLURM_JOB_ID"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Start Time: $(date)"

mkdir -p /data/users/ltucker/influenzaData/pipeline/log/svd

# Read master index
MASTER_INDEX="/data/users/ltucker/influenzaData/pipeline/output/tensor/subsample_indices/master_index.csv"

# Get parameters for this array task (skip header, use 1-indexed)
LINE_NUM=$((SLURM_ARRAY_TASK_ID + 2))
PARAMS=$(sed -n "${LINE_NUM}p" "$MASTER_INDEX")

PCT=$(echo "$PARAMS" | cut -d',' -f2)
ITER=$(echo "$PARAMS" | cut -d',' -f3)
SEED=$(echo "$PARAMS" | cut -d',' -f4)

echo "Processing: ${PCT}% subsample, iteration ${ITER}, seed ${SEED}"

SCRIPT_PATH="/data/users/ltucker/influenzaData/pipeline/scripts/07b_svd_scree_plots.py"

singularity exec --cleanenv \
    --bind /data \
    /data/users/ltucker/influenzaData/pipeline/jupyter-tensor_12.sif \
    python -u "$SCRIPT_PATH" \
        --percentage "$PCT" \
        --iteration "$ITER" \
        --seed "$SEED"

echo "Job completed at $(date)"