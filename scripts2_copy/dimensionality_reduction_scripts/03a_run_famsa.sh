#!/bin/bash

#SBATCH --time=2-00:00:00
#SBATCH --mem=200GB
#SBATCH --cpus-per-task=10
#SBATCH --job-name=famsa_duckdb
#SBATCH --partition=pibu_el8
#SBATCH --array=1-8
#SBATCH --output=/data/users/ltucker/influenzaData/H5N1_pipeline/log/03_famsa_duckdb_%A_%a.out
#SBATCH --error=/data/users/ltucker/influenzaData/H5N1_pipeline/log/03_famsa_duckdb_%A_%a.err

WORKDIR="/data/users/ltucker/influenzaData/H5N1_pipeline"
SEGMENTS_DIR="$WORKDIR/output/segments"
DISTANCES_DIR="$WORKDIR/output/distances"

mkdir -p "$DISTANCES_DIR"

SEGMENT=$SLURM_ARRAY_TASK_ID

echo "===== Processing segment: $SEGMENT (Array task $SLURM_ARRAY_TASK_ID) ====="

# ---- Step 1: FAMSA distance matrix ----
eval "$(conda shell.bash hook)"
conda activate distanceMatrix

INFILE="$SEGMENTS_DIR/${SEGMENT}.fa"
DISTANCE_CSV="$DISTANCES_DIR/distance_${SEGMENT}.csv"

echo "Input file: $INFILE"
echo "Input sequences: $(grep -c "^>" "$INFILE")"

famsa \
    -dist_export \
    -square_matrix \
    -v \
    -t 10 \
    -gz \
    "$INFILE" \
    "$DISTANCE_CSV"

if [ $? -ne 0 ]; then
    echo "ERROR: FAMSA failed for segment $SEGMENT" >&2
    exit 1
fi
echo "FAMSA completed for segment $SEGMENT"

# ---- Step 2: Convert to parquet with duckdb ----
conda activate /data/users/ltucker/influenzaData/H5N1_pipeline/conda_envs/duckdb

PARQUET_FILE="$DISTANCES_DIR/distance_${SEGMENT}.parquet"

duckdb -c "COPY (SELECT * FROM read_csv('$DISTANCE_CSV', max_line_size=3000000)) TO '$PARQUET_FILE' (FORMAT PARQUET);"

if [ $? -ne 0 ]; then
    echo "ERROR: DuckDB parquet conversion failed for segment $SEGMENT" >&2
    exit 1
fi

echo "Parquet conversion completed for segment $SEGMENT"
echo "===== Segment $SEGMENT done ====="