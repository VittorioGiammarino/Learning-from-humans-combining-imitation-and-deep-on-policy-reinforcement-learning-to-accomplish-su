#!/bin/bash

for seed in $(seq 0 7);
do
qsub SAC.qsub $seed 
qsub TD3.qsub $seed 
done 
