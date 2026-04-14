#!/bin/bash

WORKDIR="/data/users/ltucker/influenzaData/H5N1_pipeline"
MAP_CSV="$WORKDIR/output/annotated_files/header_map_with_year_and_location.csv"
IN_FASTA="$WORKDIR/output/annotated_files/complete_virus_fasta.fa"
OUT_FASTA="$WORKDIR/output/annotated_files/complete_virus_with_mapped_headers.fa"



awk -F',' 'NR==FNR && NR>1 {gsub(/"/, "", $2); gsub(/"/, "", $3); map[$2]=$3; next} 
           /^>/ {header=$0; if (header in map) {print map[header]; keep=1} else {print header > "unmapped.txt"; keep=0}; next} 
           keep' $MAP_CSV $IN_FASTA > $OUT_FASTA