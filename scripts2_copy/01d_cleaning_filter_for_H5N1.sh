#!/bin/bash

#SBATCH --time=2-00:00:00
#SBATCH --mem=200GB
#SBATCH --cpus-per-task=10
#SBATCH --job-name=filterH5N1
#SBATCH --partition=pibu_el8
#SBATCH --output=/data/users/ltucker/influenzaData/H5N1_pipeline/log/cleaning/01c_filter_H5N1_%j.out
#SBATCH --error=/data/users/ltucker/influenzaData/H5N1_pipeline/log/cleaning/01c_filter_H5N1_%j.err

eval "$(conda shell.bash hook)"
conda activate /data/users/ltucker/influenzaData/H5N1_pipeline/conda_envs/fastaTools

WORKDIR="/data/users/ltucker/influenzaData/H5N1_pipeline"

INPUT="$WORKDIR/output/cleaned_files/no_duplicates_clean_filtered.fa"
OUTPUT="$WORKDIR/output/cleaned_files/H5N1_only.fa"

seqkit grep -r -p "^A/H5N1\|" "$INPUT" > "$OUTPUT"

# stats
INPUT_COUNT=$(grep -c "^>" "$INPUT")
OUTPUT_COUNT=$(grep -c "^>" "$OUTPUT")
REMOVED=$((INPUT_COUNT - OUTPUT_COUNT))

echo "===== H5N1 Filter Stats ====="
echo "Input sequences:    $INPUT_COUNT"
echo "H5N1 sequences:     $OUTPUT_COUNT"
echo "Non-H5N1 removed:   $REMOVED"
echo "Percent kept:       $(awk "BEGIN {printf \"%.2f\", ($OUTPUT_COUNT/$INPUT_COUNT)*100}")%"
echo "=============================="