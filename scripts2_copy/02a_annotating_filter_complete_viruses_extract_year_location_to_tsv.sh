#!/bin/bash

WORKDIR="/data/users/ltucker/influenzaData/H5N1_pipeline"
FASTA="$WORKDIR/output/cleaned_files/H5N1_only.fa"
OUTDIR="$WORKDIR/output/annotated_files"
HEADERFILE="$OUTDIR/H5N1_only_headers.txt"
mkdir -p $OUTDIR

#1 Get fasta Headers and Extract to a file
grep '^>' $FASTA > $HEADERFILE
echo "Step 1 - Extracted headers: $(wc -l < $HEADERFILE)"

#2 make sure there are only segments with 8x multiple by considering only virus identifying fields and remove segment identifying fields
awk -F'|' '
{virus_id = $1 "|" $2 "|" $5
    lines[virus_id] = lines[virus_id] $0 "\n" 
    count[virus_id]++} 
END {for (virus_id in count) {
        if (count[virus_id] == 8) {
            printf "%s", lines[virus_id]}}}' $HEADERFILE > $OUTDIR/complete_virus_headers.txt
echo "Step 2 - Headers with complete 8 segments: $(wc -l < $OUTDIR/complete_virus_headers.txt)"
echo "Step 2 - Complete viruses: $(( $(wc -l < $OUTDIR/complete_virus_headers.txt) / 8 ))"
echo "Step 2 - Headers removed: $(( $(wc -l < $HEADERFILE) - $(wc -l < $OUTDIR/complete_virus_headers.txt) ))"

#3 Make a TSV with different the orgininal header, the year, and the location
cat $OUTDIR/complete_virus_headers.txt | awk -F'|' '{ n = split($2, arr, "/"); 
                        print $0, arr[NF], arr[NF-2] }' > $OUTDIR/complete_virus_headers_with_extracted_metadata.tsv
echo "Step 3 - Headers in metadata TSV: $(wc -l < $OUTDIR/complete_virus_headers_with_extracted_metadata.tsv)"