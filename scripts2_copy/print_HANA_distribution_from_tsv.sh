#!/bin/bash

INPUT="/data/users/ltucker/influenzaData/H5N1_pipeline/output/annotated_files/complete_virus_headers_with_extracted_metadata.tsv"

echo $INPUT

echo "===== HA subtype distribution ====="
grep '^>' "$INPUT" \
| cut -d'|' -f1 \
| grep -o 'H[0-9]\+' \
| sort \
| uniq -c \
| sort -rn

echo ""
echo "===== NA subtype distribution ====="
grep '^>' "$INPUT" \
| cut -d'|' -f1 \
| grep -o 'N[0-9]\+' \
| sort \
| uniq -c \
| sort -rn

echo ""
echo "===== HA/NA combination distribution ====="
grep '^>' "$INPUT" \
| cut -d'|' -f1 \
| grep -o 'H[0-9]\+N[0-9]\+' \
| sort \
| uniq -c \
| sort -rn