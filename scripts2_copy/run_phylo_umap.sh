#!/bin/bash
#SBATCH --job-name=phylo_umap
#SBATCH --output=/data/users/ltucker/influenzaData/H5N1_pipeline/log/%j.out
#SBATCH --error=/data/users/ltucker/influenzaData/H5N1_pipeline/log/%j.err
#SBATCH --time=02:00:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=4
#SBATCH --partition=pibu_el8      # ← change to your partition name

# ── Paths ──────────────────────────────────────────────────────────────────────
PIPELINE=/data/users/ltucker/influenzaData/H5N1_pipeline

SIF=/data/users/ltucker/influenzaData/pipeline/jupyter-tensor_16.sif
IQTREE_DIR=$PIPELINE/output/iqtree
EMBED_DIR=$PIPELINE/output/famsa_per_segment_analysis/embedding_param_sweep/embeddings
SCRIPTDIR=$PIPELINE/scripts
OUTDIR=$PIPELINE/output/phylo_umap_comparison

# Influenza segment → name mapping
# 1=PB2, 2=PB1, 3=PA, 4=HA, 5=NP, 6=NA, 7=MP, 8=NS
SEGMENT=${1:-4}        # pass segment number as argument, default = 4 (HA)

# Map segment number to name for the embedding folder
declare -A SEG_NAMES=([1]=PB2 [2]=PB1 [3]=PA [4]=HA [5]=NP [6]=NA [7]=MP [8]=NS)
SEG_NAME=${SEG_NAMES[$SEGMENT]}

mkdir -p "$OUTDIR"

# ── Run ────────────────────────────────────────────────────────────────────────
# Use 'exec' (not 'run') to bypass the Jupyter entrypoint in the base image
singularity exec \
    --bind "$PIPELINE":/pipeline \
    "$SIF" \
    python /pipeline/scripts/phylo_umap_compare.py \
        --tree     /pipeline/output/iqtree/segment_${SEGMENT}.treefile \
        --umap     /pipeline/output/famsa_per_segment_analysis/embedding_param_sweep/embeddings/${SEG_NAME}/umap_nn15_md0.1.parquet \
        --n-clades 8 \
        --out      /pipeline/output/phylo_umap_comparison/segment_${SEGMENT}_${SEG_NAME}.pdf