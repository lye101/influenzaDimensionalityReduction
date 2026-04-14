#!/bin/bash

#SBATCH --time=2-00:00:00
#SBATCH --mem=200GB
#SBATCH --cpus-per-task=10
#SBATCH --job-name=keep_8_segs
#SBATCH --partition=pibu_el8
#SBATCH --output=/data/users/ltucker/influenzaData/H5N1_pipeline/log/header_annotation/02c_keep_8_segs_%j.out
#SBATCH --error=/data/users/ltucker/influenzaData/H5N1_pipeline/log/header_annotation/02c_keep_8_segs_%j.err

# This script filters the fasta by the headers filtered by the r produced csv so that only whole unique virus remain.
# Although this could have been don with complete_virus_headers.txt I chose to do it this way in case there is an issue downstream.

mkdir -p /data/users/ltucker/influenzaData/H5N1_pipeline/log/header_annotation
# get dependencies
eval "$(conda shell.bash hook)"
conda activate distanceMatrix

WORKDIR=/data/users/ltucker/influenzaData/H5N1_pipeline
ANNOTATED_DIR="$WORKDIR/output/annotated_files"
INPUTFILE="$WORKDIR/output/cleaned_files/H5N1_only.fa"

mkdir -p "$ANNOTATED_DIR"

# Extract headers to keep (second column from CSV)
tail -n +2 "$ANNOTATED_DIR/header_map_with_year_and_location.csv" | \
    cut -d ',' -f 2 | \
    sed 's/"//g' | \
    sed 's/^>//' > "$ANNOTATED_DIR/complete_virus_from_02b.txt"

# Only keep sequences with all 8 segments
seqkit grep -n -f "$ANNOTATED_DIR/complete_virus_from_02b.txt" "$INPUTFILE" -o "$ANNOTATED_DIR/complete_virus_fasta.fa"

rm "$ANNOTATED_DIR/complete_virus_from_02b.txt"