#!/bin/bash

#SBATCH --time=1-00:00:00
#SBATCH --mem=100GB
#SBATCH --cpus-per-task=1
#SBATCH --job-name=duckdb
#SBATCH --partition=pibu_el8
#SBATCH --output=/data/users/ltucker/influenzaData/pipeline/log/2b_duckdb_parquet_%A_%a.out
#SBATCH --error=/data/users/ltucker/influenzaData/pipeline/log/2b_duckdb_parquet_%A_%a.err
#SBATCH --array=0-7

# conda setup
eval "$(conda shell.bash hook)" 
conda activate /data/users/ltucker/influenzaData/pipeline/envs/duckdb

# Array of segments gotta catch em all
SEGMENTS=($(seq 1 8))

# Get the segment for this array task
SEGMENT=${SEGMENTS[$SLURM_ARRAY_TASK_ID]}

WORKDIR="/data/users/ltucker/influenzaData/pipeline"
OUTDIR="$WORKDIR/output/1_distances"

mkdir -p $OUTDIR

INFILE="$OUTDIR/distance_$SEGMENT.csv"
OUTFILE="$OUTDIR/distance_$SEGMENT.parquet"

duckdb -c "COPY (SELECT * FROM read_csv('$INFILE', max_line_size=3000000)) TO '$OUTFILE' (FORMAT PARQUET);"

echo "Completed $SEGMENT as parquet"
