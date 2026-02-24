#!/bin/bash
#SBATCH --job-name=combine_scree
#SBATCH --output=/data/users/ltucker/influenzaData/pipeline/log/svd/combine_scree_%j.log
#SBATCH --error=/data/users/ltucker/influenzaData/pipeline/log/svd/combine_scree_%j.err
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --partition=pshort_el8 #pibu_el8

echo "Job ID: $SLURM_JOB_ID"
echo "Start Time: $(date)"

mkdir -p /data/users/ltucker/influenzaData/pipeline/log/svd

SCRIPT_PATH="/data/users/ltucker/influenzaData/pipeline/scripts/07b2_combined_scree_plots.py"

# Run with all plots
singularity exec --cleanenv \
    --bind /data \
    /data/users/ltucker/influenzaData/pipeline/jupyter-tensor_12.sif \
    python -u "$SCRIPT_PATH"

echo "Job completed at $(date)"
