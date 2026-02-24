#!/bin/bash
#SBATCH --job-name=svd_scree_missing
#SBATCH --output=/data/users/ltucker/influenzaData/pipeline/log/svd/scree_missing_%A_%a.log
#SBATCH --error=/data/users/ltucker/influenzaData/pipeline/log/svd/scree_missing_%A_%a.err
#SBATCH --array=0-3
#SBATCH --time=7-04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=300G
#SBATCH --partition=pibu_el8

echo "Job ID: $SLURM_JOB_ID"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Start Time: $(date)"

mkdir -p /data/users/ltucker/influenzaData/pipeline/log/svd

# Manual mapping for missing runs
# 70%: 3 iterations (seeds 5000, 5001, 5002)
# 100%: 1 iteration (seed 6000)
case $SLURM_ARRAY_TASK_ID in
    0)
        PCT=70
        ITER=0
        SEED=5000
        ;;
    1)
        PCT=70
        ITER=1
        SEED=5001
        ;;
    2)
        PCT=70
        ITER=2
        SEED=5002
        ;;
    3)
        PCT=100
        ITER=0
        SEED=6000
        ;;
esac

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