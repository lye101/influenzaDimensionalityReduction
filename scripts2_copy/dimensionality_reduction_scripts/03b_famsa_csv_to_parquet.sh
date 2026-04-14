#!/bin/bash

#SBATCH --time=3-00:00:00
#SBATCH --mem=200GB
#SBATCH --cpus-per-task=1
#SBATCH --job-name=duckdb
#SBATCH --partition=pibu_el8
#SBATCH --array=1-8
#SBATCH --output=/data/users/ltucker/influenzaData/H5N1_pipeline/log/04_duckdb_parquet_%A_%a.out
#SBATCH --error=/data/users/ltucker/influenzaData/H5N1_pipeline/log/04_duckdb_parquet_%A_%a.err

eval "$(conda shell.bash hook)"
conda activate /data/users/ltucker/influenzaData/H5N1_pipeline/conda_envs/duckdb

WORKDIR="/data/users/ltucker/influenzaData/H5N1_pipeline"
DISTANCES_DIR="$WORKDIR/output/distances"

SEGMENT=$SLURM_ARRAY_TASK_ID

INFILE="$DISTANCES_DIR/distance_${SEGMENT}.csv"
OUTFILE="$DISTANCES_DIR/distance_${SEGMENT}.parquet"

echo "Converting segment $SEGMENT: $INFILE -> $OUTFILE"

duckdb -c "COPY (SELECT * FROM read_csv('$INFILE', max_line_size=3000000)) TO '$OUTFILE' (FORMAT PARQUET);"

if [ $? -ne 0 ]; then
    echo "ERROR: Parquet conversion failed for segment $SEGMENT" >&2
    exit 1
fi

echo "Completed segment $SEGMENT"