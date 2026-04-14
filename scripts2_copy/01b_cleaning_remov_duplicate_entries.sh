#!/bin/bash

#SBATCH --time=2-00:00:00
#SBATCH --mem=8GB
#SBATCH --cpus-per-task=10
#SBATCH --job-name=filterFasta
#SBATCH --partition=pibu_el8
#SBATCH --output=/data/users/ltucker/influenzaData/H5N1_pipeline/log/cleaning/01a_remove_dups_%j.out
#SBATCH --error=/data/users/ltucker/influenzaData/H5N1_pipeline/log/cleaning/01a_remove_dups_fasta_%j.err

# get dependancies
eval "$(conda shell.bash hook)"
conda activate distanceMatrix

WORKDIR="/data/users/ltucker/influenzaData/H5N1_pipeline"
mkdir -p $WORKDIR/output/cleaned_files

INPUT="$WORKDIR/output/cleaned_files/length_filtered_fasta.fa"
OUTPUT="$WORKDIR/output/cleaned_files/no_duplicates_clean.fa"

# remove duplicates 
seqkit rmdup -n "$INPUT" > "$OUTPUT"

# stats
INPUT_COUNT=$(grep -c "^>" "$INPUT")
OUTPUT_COUNT=$(grep -c "^>" "$OUTPUT")
REMOVED=$((INPUT_COUNT - OUTPUT_COUNT))

echo "===== Duplicate Removal Stats ====="
echo "Input sequences:   $INPUT_COUNT"
echo "Output sequences:  $OUTPUT_COUNT"
echo "Duplicates removed: $REMOVED"
echo "Percent removed:   $(awk "BEGIN {printf \"%.2f\", ($REMOVED/$INPUT_COUNT)*100}")%"
echo "==================================="