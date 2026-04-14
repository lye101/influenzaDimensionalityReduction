#!/bin/bash

#SBATCH --job-name=tucker_decomp
#SBATCH --output=/data/users/ltucker/influenzaData/H5N1_pipeline/log/tensor/tucker_%j.out
#SBATCH --error=/data/users/ltucker/influenzaData/H5N1_pipeline/log/tensor/tucker_%j.err
#SBATCH --time=4-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=300G
#SBATCH --partition=pibu_el8

echo "Job ID: $SLURM_JOB_ID"
echo "Start Time: $(date)"

WORKDIR="/data/users/ltucker/influenzaData/H5N1_pipeline"
LOG_DIR="$WORKDIR/log/tensor"
CONTAINER="$WORKDIR/../pipeline/jupyter-tensor_15.sif"

mkdir -p "$LOG_DIR"
mkdir -p "$WORKDIR/output/famsa_tensor_analysis/tensor"

RANK="4,5,5"
SEED=42

echo "Rank: $RANK"
echo "Seed: $SEED"

singularity exec --bind /data "$CONTAINER" python3 "$WORKDIR/scripts/dimensionality_reduction_scripts/08_tucker_decompositon.py" \
    --rank "$RANK" \
    --seed "$SEED"

exit_code=$?

echo "End Time: $(date)"

if [ $exit_code -eq 0 ]; then
    echo "Tucker decomposition completed successfully"
else
    echo "ERROR: Tucker decomposition failed (exit code $exit_code)" >&2
    exit $exit_code
fi