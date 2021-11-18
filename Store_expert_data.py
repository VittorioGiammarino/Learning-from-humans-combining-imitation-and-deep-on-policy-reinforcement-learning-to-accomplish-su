#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  9 15:58:37 2021

@author: vittorio
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as ptch
import os

import World
from utils import Show_DataSet
from utils import Show_Training

# %% Preprocessing_data from humans with psi based on the coins clusters distribution  
Folders = [6, 7, 11, 12, 15]

TrainingSet, Labels, Trajectories, Rotation, Time, _, _, _, _, _ = Show_DataSet(Folders, 'complete', 'distr_only')
_,_,_,_,_, Reward_eval_human, Real_Traj_eval_human, Real_Reward_eval_human, Real_Time_eval_human, Coins_location = Show_DataSet(Folders, 'complete', 'full_coins')
_, _, _, _, _, Reward_training_human, Real_Traj_training_human, Real_Reward_training_human, Real_Time_training_human = Show_Training(Folders, 'complete', 'full_coins')

# %% Plot human expert

threshold = np.mean(Real_Reward_eval_human)
# performance_training = np.mean(Real_Reward_training_human)
Rand_traj_array = [2, 10, 13, 16, 17, 29, 30, 35, 40, 41, 42, 43, 44, 46, 49]
# Rand_traj = np.argmax(Real_Reward_eval_human)

n_options = 3

for Rand_traj in Rand_traj_array:
    
    supervised_options = np.copy(Trajectories[Rand_traj][:,2])
    
    if n_options == 3:
        for s in range(len(supervised_options)):
            if Trajectories[Rand_traj][s,1] > 6.5 or Trajectories[Rand_traj][s,1] < -6.5 or Trajectories[Rand_traj][s,0] > 6.5 or Trajectories[Rand_traj][s,0] < -6.5:
                if supervised_options[s] == 0:
                    supervised_options[s] = 2

    size_data = len(Trajectories[Rand_traj])
    
    nTraj = Rand_traj%10
    folder = Folders[int(Rand_traj/10)]
    coins_location = Coins_location[nTraj]
    
    sigma1 = 0.5
    circle1 = ptch.Circle((6, 7.5), 2*sigma1, color='k',  fill=False)
    sigma2 = 1.1
    circle2 = ptch.Circle((-1.5, -5), 2*sigma2, color='k',  fill=False)
    sigma3 = 1.8
    circle3 = ptch.Circle((-5, 3), 2*sigma3, color='k',  fill=False)
    sigma4 = 1.3
    circle4 = ptch.Circle((4.9, -4), 2*sigma4, color='k',  fill=False)
    fig, ax = plt.subplots()
    ax.add_artist(circle1)
    ax.add_artist(circle2)
    ax.add_artist(circle3)
    ax.add_artist(circle4)
    plot_data = plt.scatter(Trajectories[Rand_traj][0:size_data,0], Trajectories[Rand_traj][0:size_data,1], c=Time[Rand_traj][0:size_data], marker='o', cmap='cool') #-1], marker='o', cmap='cool')
    plt.plot(0.1*coins_location[:,0], 0.1*coins_location[:,1], 'xb')
    cbar = fig.colorbar(plot_data, ticks=[10, 100, 200, 300, 400, 500])
    cbar.ax.set_yticklabels(['time = 0', 'time = 100', 'time = 200', 'time = 300', 'time = 400', 'time = 500'])
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Human Agent {}, Reward {}'.format(Rand_traj, Reward_eval_human[Rand_traj]))
    plt.show()  
    
    time = np.linspace(0,480,len(Real_Traj_eval_human[Rand_traj][:,0])) 
    
    sigma1 = 0.5
    circle1 = ptch.Circle((6, 7.5), 2*sigma1, color='k',  fill=False)
    sigma2 = 1.1
    circle2 = ptch.Circle((-1.5, -5), 2*sigma2, color='k',  fill=False)
    sigma3 = 1.8
    circle3 = ptch.Circle((-5, 3), 2*sigma3, color='k',  fill=False)
    sigma4 = 1.3
    circle4 = ptch.Circle((4.9, -4), 2*sigma4, color='k',  fill=False)
    fig, ax = plt.subplots()
    ax.add_artist(circle1)
    ax.add_artist(circle2)
    ax.add_artist(circle3)
    ax.add_artist(circle4)
    plot_data = plt.scatter(0.1*Real_Traj_eval_human[Rand_traj][:,0], 0.1*Real_Traj_eval_human[Rand_traj][:,1], c=time, marker='o', cmap='cool') #-1], marker='o', cmap='cool')
    plt.plot(0.1*coins_location[:,0], 0.1*coins_location[:,1], 'xb')
    cbar = fig.colorbar(plot_data, ticks=[10, 100, 200, 300, 400, 500])
    cbar.ax.set_yticklabels(['time = 0', 'time = 100', 'time = 200', 'time = 300', 'time = 400', 'time = 500'])
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Human Agent, Reward {}'.format(Real_Reward_eval_human[Rand_traj]))
    plt.show() 
    
    sigma1 = 0.5
    circle1 = plt.Circle((6, 7.5), 2*sigma1, color='k',  fill=False)
    sigma2 = 1.1
    circle2 = plt.Circle((-1.5, -5), 2*sigma2, color='k',  fill=False)
    sigma3 = 1.8
    circle3 = plt.Circle((-5, 3), 2*sigma3, color='k',  fill=False)
    sigma4 = 1.3
    circle4 = plt.Circle((4.9, -4), 2*sigma4, color='k',  fill=False)
    fig3, ax3 = plt.subplots()
    plot_data = plt.scatter(Trajectories[Rand_traj][:,0], Trajectories[Rand_traj][:,1], c=supervised_options, marker='o', cmap='brg')
    plt.plot(0.1*coins_location[:,0], 0.1*coins_location[:,1], 'xk')
    cbar = fig3.colorbar(plot_data, ticks=[0, 1, 2])
    cbar.ax.set_yticklabels(['move center', 'collect', 'move wall'])
    ax3.add_artist(circle1)
    ax3.add_artist(circle2)
    ax3.add_artist(circle3)
    ax3.add_artist(circle4)
    plt.xlabel('x')
    plt.ylabel('y')
    # plt.title('Options, action space {}, coins {}'.format(action_space, coins))
    plt.show()   

# %%
if not os.path.exists("./Expert_data"):
    os.makedirs("./Expert_data")
    
np.save("./Expert_data/TrainingSet", TrainingSet)
np.save("./Expert_data/Labels", Labels)
np.save("./Expert_data/Trajectories", Trajectories)
np.save("./Expert_data/Rotation", Rotation)
np.save("./Expert_data/Time", Time)
np.save("./Expert_data/Reward_eval_human", Reward_eval_human)
np.save("./Expert_data/Real_Traj_eval_human", Real_Traj_eval_human)
np.save("./Expert_data/Real_Reward_eval_human", Real_Reward_eval_human)
np.save("./Expert_data/Real_Time_eval_human", Real_Time_eval_human)
np.save("./Expert_data/Coins_location", Coins_location)

