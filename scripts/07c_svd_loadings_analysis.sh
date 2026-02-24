#!/bin/bash
#SBATCH --job-name=svd_loadings
#SBATCH --output=/data/users/ltucker/influenzaData/pipeline/log/loadings/loadings_%j.log
#SBATCH --error=/data/users/ltucker/influenzaData/pipeline/log/loadings/loadings_%j.err
#SBATCH --time=7-02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=100G
#SBATCH --partition=pibu_el8

echo "Job ID: $SLURM_JOB_ID"
echo "Start Time: $(date)"

mkdir -p /data/users/ltucker/influenzaData/pipeline/log/loadings

SCRIPT_PATH="/data/users/ltucker/influenzaData/pipeline/scripts/07c_svd_loadings_analysis.py"

# Process all 50% iterations
singularity exec --cleanenv \
    --bind /data \
    /data/users/ltucker/influenzaData/pipeline/jupyter-tensor_12.sif \
    python -u "$SCRIPT_PATH"

echo "Job completed at $(date)"