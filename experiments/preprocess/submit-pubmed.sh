#!/bin/bash

for (( index=1; index<=18; index+=1 )); do
    sbatch --job-name=pubmed-$index ./pubmed.sh $index
done