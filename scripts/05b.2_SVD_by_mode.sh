#!/bin/bash

#SBATCH --job-name=Mode_SVD
#SBATCH --output=/data/users/ltucker/influenzaData/pipeline/log/05b.2_SVD_by_mode_%j.log
#SBATCH --error=/data/users/ltucker/influenzaData/pipeline/log/05b.2_SVD_by_mode_%j.err
#SBATCH --time=1-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=300G
#SBATCH --partition=pibu_el8 # pshort_el8

WORKDIR="/data/users/$USER/influenzaData/pipeline"
singularity exec --contain --cleanenv \
    --bind /data/users/$USER:/data/users/$USER \
    $WORKDIR/jupyter-tensor_11.sif \
    python -u $WORKDIR/scripts/05b.2_SVD_by_mode.py
    