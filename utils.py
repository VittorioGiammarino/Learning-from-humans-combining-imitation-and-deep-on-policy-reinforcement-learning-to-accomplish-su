#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 29 10:19:08 2021

@author: vittorio
"""
import numpy as np
import World

from sklearn.preprocessing import OneHotEncoder

# %% Preprocessing_data with psi based on the coins clusters distribution  

def Show_DataSet(Folders, action_space, coins):
# =============================================================================
#     action_space = 'complete', which considers an heading direction every 45deg
#                    'simplified', which considers an action every 90
#     coins = 'distr_only' coins on the gaussians distributions
#             'full_coins' all the coins in the original experiment
# =============================================================================
    TrainingSet = np.empty((0,4))
    Labels = np.empty((0,1))
    Time = []
    Real_Time = []
    Trajectories = []
    Real_Traj = []
    Rotation = []
    Reward = []
    Real_reward = []
    Coin_location = []
    for folder in Folders:
        for experiment in range(1,11):
            Training_set_single_traj, Labels_single_traj, Time_single_traj, psi_single_traj, coin_direction_single_traj, reward_single_traj, real_reward, coin_location = World.Foraging.ProcessData(folder, experiment, action_space, coins)
            Training_set_single_traj_together = np.concatenate((0.1*Training_set_single_traj[0:-1,:], psi_single_traj[0:-1].reshape(len(psi_single_traj[0:-1]),1), coin_direction_single_traj[0:-1].reshape(len(psi_single_traj[0:-1]),1)),1)
            TrainingSet = np.append(TrainingSet, Training_set_single_traj_together, 0)
            Labels = np.append(Labels, Labels_single_traj.reshape(len(Labels_single_traj),1), 0)
            Time.append(Time_single_traj)
            Trajectories.append(Training_set_single_traj_together)
            Rotation.append(Labels_single_traj.reshape(len(Labels_single_traj),1))
            Reward.append(reward_single_traj)
            Real_reward.append(real_reward)
            True_traj, _, True_time = World.Foraging.TrueData(folder, experiment)
            Real_Traj.append(True_traj)
            Real_Time.append(True_time)
            Coin_location.append(coin_location)
    
    return TrainingSet, Labels, Trajectories, Rotation, Time, Reward, Real_Traj, Real_reward, Real_Time, Coin_location

def Show_Training(Folders, action_space, coins):
# =============================================================================
#     action_space = 'complete', which considers an heading direction every 45deg
#                    'simplified', which considers an action every 
#     coins = 'distr_only' coins on the gaussians distributions
#             'full_coins' all the coins in the original experiment
# =============================================================================
    TrainingSet = np.empty((0,4))
    Labels = np.empty((0,1))
    Time = []
    Real_Time = []
    Trajectories = []
    Real_Traj = []
    Rotation = []
    Reward = []
    Real_reward = []
    for folder in Folders:
        for experiment in range(1,11):
            Training_set_single_traj, Labels_single_traj, Time_single_traj, psi_single_traj, coin_direction_single_traj, reward_single_traj, real_reward = World.Foraging_Training.ProcessData(folder, experiment, action_space, coins)
            Training_set_single_traj_together = np.concatenate((0.1*Training_set_single_traj[0:-1,:], psi_single_traj[0:-1].reshape(len(psi_single_traj[0:-1]),1), coin_direction_single_traj[0:-1].reshape(len(psi_single_traj[0:-1]),1)),1)
            TrainingSet = np.append(TrainingSet, Training_set_single_traj_together, 0)
            Labels = np.append(Labels, Labels_single_traj.reshape(len(Labels_single_traj),1), 0)
            Time.append(Time_single_traj)
            Trajectories.append(Training_set_single_traj_together)
            Rotation.append(Labels_single_traj.reshape(len(Labels_single_traj),1))
            Reward.append(reward_single_traj)
            Real_reward.append(real_reward)
            True_traj, _, True_time = World.Foraging_Training.TrueData(folder, experiment)
            Real_Traj.append(True_traj)
            Real_Time.append(True_time)      
    
    return TrainingSet, Labels, Trajectories, Rotation, Time, Reward, Real_Traj, Real_reward, Real_Time

def Encode_Data(TrainingSet, Labels):
    T_set = TrainingSet
    # encode psi
    psi = T_set[:,2].reshape(len(T_set[:,2]),1)
    onehot_encoder = OneHotEncoder(sparse=False)
    onehot_encoded_psi = onehot_encoder.fit_transform(psi)
    # encode closest coin direction
    closest_coin_direction = T_set[:,3].reshape(len(T_set[:,3]),1)
    onehot_encoded_closest_coin_direction = onehot_encoder.fit_transform(closest_coin_direction)
    coordinates = T_set[:,0:2].reshape(len(T_set[:,0:2]),2)
    T_set = np.concatenate((coordinates,onehot_encoded_psi,onehot_encoded_closest_coin_direction),1)
    Heading_set = Labels
    info = []
    info.append(onehot_encoded_psi.shape[1])
    info.append(onehot_encoded_closest_coin_direction.shape[1])
    
    return T_set, Heading_set, info
