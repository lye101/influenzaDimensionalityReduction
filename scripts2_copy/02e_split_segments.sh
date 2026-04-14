#!/bin/bash

#SBATCH --time=2-00:00:00
#SBATCH --mem=200GB
#SBATCH --cpus-per-task=10
#SBATCH --job-name=splitFasta
#SBATCH --partition=pibu_el8
#SBATCH --output=/data/users/ltucker/influenzaData/H5N1_pipeline/log/header_annotation/split_fasta%j.out
#SBATCH --error=/data/users/ltucker/influenzaData/H5N1_pipeline/log/header_annotation/split_fasta_%j.err

# get dependencies
eval "$(conda shell.bash hook)"
conda activate /data/users/ltucker/influenzaData/H5N1_pipeline/conda_envs/fastaTools

WORKDIR="/data/users/ltucker/influenzaData/H5N1_pipeline"
ANNOTATED_DIR="$WORKDIR/output/annotated_files"
SEGMENTS_DIR="$WORKDIR/output/segments"
INPUTFILE="$ANNOTATED_DIR/complete_virus_with_mapped_headers.fa" 

mkdir -p $SEGMENTS_DIR

# Split by segment
for segment in 1 2 3 4 5 6 7 8; do
    seqkit grep -r -p "\|${segment}\|" "$INPUTFILE" | seqkit sort -n > "$SEGMENTS_DIR/${segment}.fa"
done