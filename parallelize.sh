#!/bin/bash

for seed in $(seq 0 7);
do
for expert_traj in $(seq 2 49);
do 
qsub PPO.qsub $seed $expert_traj
done 
done
