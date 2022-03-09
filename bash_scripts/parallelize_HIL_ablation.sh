#!/bin/bash

for seed in $(seq 0 7);
do
qsub HIL_ablation_study_allocentric_only.qsub $seed
done 
