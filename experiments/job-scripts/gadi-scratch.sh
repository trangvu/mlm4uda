#!/usr/bin/env bash

qsub -I -P dz21 -l walltime=10:00:00 -l mem=16GB -l ngpus=2 -l ncpus=24 -q  gpuvolta