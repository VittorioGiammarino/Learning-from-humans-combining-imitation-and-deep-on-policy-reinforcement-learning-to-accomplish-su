#!/bin/bash

for seed in $(seq 0 7);
do
for expert_traj in $(seq 2 49);
do 
qsub PPO_HIL_ADV_reward.qsub $seed $expert_traj
qsub PPO_HIL_HRL_ADV_reward.qsub $seed $expert_traj
done 
done
