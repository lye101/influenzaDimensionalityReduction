#!/bin/bash
#SBATCH --job-name=gen_indices
#SBATCH --output=/data/users/ltucker/influenzaData/pipeline/log/indices/generate_indices_%j.log
#SBATCH --error=/data/users/ltucker/influenzaData/pipeline/log/indices/generate_indices_%j.err
#SBATCH --time=00:10:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=50G
#SBATCH --partition=pibu_el8

echo "Job ID: $SLURM_JOB_ID"
echo "Start Time: $(date)"

mkdir -p /data/users/ltucker/influenzaData/pipeline/log/indices

SCRIPT_PATH="/data/users/ltucker/influenzaData/pipeline/scripts/07a_generate_subsample_indices.py"

singularity exec --cleanenv \
    --bind /data \
    /data/users/ltucker/influenzaData/pipeline/jupyter-tensor_12.sif \
    python -u "$SCRIPT_PATH"

echo "Job completed at $(date)"