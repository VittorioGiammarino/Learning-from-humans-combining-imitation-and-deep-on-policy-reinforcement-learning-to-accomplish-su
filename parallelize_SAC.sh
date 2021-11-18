#!/bin/bash

for seed in $(seq 0 7);
do
qsub SAC.qsub $seed 
qsub HSAC_nOptions2_supervised_True.qsub $seed 
qsub HSAC_nOptions2_supervised_False.qsub $seed 
qsub HSAC_nOptions3_supervised_True.qsub $seed 
qsub HSAC_nOptions3_supervised_False.qsub $seed 
done 
