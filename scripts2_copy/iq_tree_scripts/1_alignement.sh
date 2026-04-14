#!/bin/bash

#SBATCH --time=1-00:00:00
#SBATCH --mem=16GB
#SBATCH --cpus-per-task=8
#SBATCH --job-name=MAFFT_align
#SBATCH --partition=pibu_el8
#SBATCH --array=1-8
#SBATCH --output=/data/users/ltucker/influenzaData/H5N1_pipeline/log/mafft/mafft_alignment_%A_%a.out
#SBATCH --error=/data/users/ltucker/influenzaData/H5N1_pipeline/log/mafft/mafft_alignment_%A_%a.err

# ==============================================================================
# MAFFT Multiple Sequence Alignment - Influenza Segments 1-8
# ==============================================================================
# Segment lengths range from ~890 bp (NS) to ~2,341 bp (PB2), so sequences
# are short. FFT-NS-2 (--retree 2) gives a good balance of speed and accuracy.
# ==============================================================================

WORKDIR=/data/users/ltucker/influenzaData/H5N1_pipeline
SEGMENTS_DIR="$WORKDIR/output/segments"
ALIGNED_DIR="$WORKDIR/output/mafft_aligned"
LOG_DIR="$WORKDIR/log/mafft"
#ENV="$WORKDIR/conda_envs/fastaTools"
ENV="/data/users/ltucker/influenzaData/pipeline/envs/mafft"

# Activate conda environment
eval "$(conda shell.bash hook)"
conda activate $ENV

# Create output directories
mkdir -p "$ALIGNED_DIR"
mkdir -p "$LOG_DIR"

# Build array of segment files and pick this task's file
segments=("$SEGMENTS_DIR"/*.fa)
segment="${segments[$SLURM_ARRAY_TASK_ID-1]}"

# Extract base name
base=$(basename "$segment" .fa)

echo "========================================"
echo "MAFFT Alignment - Segment $base"
echo "========================================"
echo "Job ID:       $SLURM_JOB_ID (array task $SLURM_ARRAY_TASK_ID)"
echo "Input file:   $segment"
echo "Output file:  $ALIGNED_DIR/aligned_${base}.fa"
echo "Threads:      $SLURM_CPUS_ON_NODE"
echo "Start time:   $(date)"
echo "========================================"

# Verify input exists
if [ ! -f "$segment" ]; then
    echo "ERROR: Input file not found: $segment" >&2
    exit 1
fi

# Report input stats
n_seqs=$(grep -c "^>" "$segment")
echo "Number of sequences: $n_seqs"

# ---- Run MAFFT ----
mafft \
    --retree 2 \
    --maxiterate 0 \
    --thread "$SLURM_CPUS_ON_NODE" \
    --reorder \
    --adjustdirection \
    "$segment" \
    > "$ALIGNED_DIR/aligned_${base}.fa" \
    2> "$LOG_DIR/mafft_${base}_stderr.log"

exit_code=$?

echo "========================================"
echo "End time: $(date)"

if [ $exit_code -eq 0 ]; then
    aligned_len=$(awk '/^>/{if(seq) print length(seq); seq=""} !/^>/{seq=seq$0} END{print length(seq)}' \
                  "$ALIGNED_DIR/aligned_${base}.fa" | head -1)
    echo "Alignment length: $aligned_len bp"
    echo "Successfully completed alignment for segment: $base"
else
    echo "ERROR: MAFFT failed for segment $base (exit code $exit_code)" >&2
    echo "Check stderr log: $LOG_DIR/mafft_${base}_stderr.log" >&2
    exit $exit_code
fi

conda deactivate