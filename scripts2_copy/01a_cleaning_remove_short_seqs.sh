#!/bin/bash

## Here the goal is uniforamize the sequences to have 
    #a length around what is expected +/- 20%

WORKDIR="/data/users/ltucker/influenzaData/H5N1_pipeline"
INPUT="$WORKDIR/data/clean.fa.gz"
OUTPUT="$WORKDIR/output/cleaned_files/length_filtered_fasta.fa"

mkdir -p $WORKDIR/output/cleaned_files

zcat "$INPUT" | awk '
BEGIN {
    REF["PB2"]=2340; 
    REF["PB1"]=2340; 
    REF["PA"]=2230; 
    REF["HA"]=1770;
    REF["NP"]=1560; 
    REF["NA"]=1460;
    REF["MP"]=1030;
    REF["NS"]=890
}
/^>/ {
    if (header && seq) check_and_print()
    header = $0; seq = ""; next
}
{ seq = seq $0 }
END { if (header && seq) check_and_print() }

function check_and_print() {
    len = length(seq)
    for (seg in REF) {
        pattern = "\\|" seg "\\|"
        if (header ~ pattern) {
            if (len >= REF[seg]*0.8 && len <= REF[seg]*1.2) {
                print header
                print seq
            }
            break
        }
    }
}
' > "$OUTPUT"