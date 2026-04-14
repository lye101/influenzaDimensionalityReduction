#!/bin/bash

#SBATCH --time=2-00:00:00
#SBATCH --mem=200GB
#SBATCH --cpus-per-task=10
#SBATCH --job-name=removeHeaders
#SBATCH --partition=pibu_el8
#SBATCH --output=/data/users/ltucker/influenzaData/H5N1_pipeline/log/cleaning/01b_remove_headers_%j.out
#SBATCH --error=/data/users/ltucker/influenzaData/H5N1_pipeline/log/cleaning/01b_remove_headers_%j.err

eval "$(conda shell.bash hook)"
conda activate fastaTools

WORKDIR="/data/users/ltucker/influenzaData/H5N1_pipeline"

INPUT="$WORKDIR/output/cleaned_files/no_duplicates_clean.fa"
OUTPUT="$WORKDIR/output/cleaned_files/no_duplicates_clean_filtered.fa"

# headers to remove
PATTERN="r1678.*EPI_ISL_137342"

seqkit grep -v -r -p "$PATTERN" "$INPUT" > "$OUTPUT"

# stats
INPUT_COUNT=$(grep -c "^>" "$INPUT")
OUTPUT_COUNT=$(grep -c "^>" "$OUTPUT")
REMOVED=$((INPUT_COUNT - OUTPUT_COUNT))

echo "===== Header Removal Stats ====="
echo "Input sequences:    $INPUT_COUNT"
echo "Output sequences:   $OUTPUT_COUNT"
echo "Sequences removed:  $REMOVED"
echo "================================"