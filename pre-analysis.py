#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 18 16:36:58 2021

@author: vittorio
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import matplotlib.patches as ptch
import pickle

# %% Load Data

TrainingSet_tot = np.load("./Expert_data/TrainingSet.npy")
Labels_tot = np.load("./Expert_data/Labels.npy")
Trajectories = np.load("./Expert_data/Trajectories.npy", allow_pickle=True).tolist()
Rotation = np.load("./Expert_data/Rotation.npy", allow_pickle=True).tolist()
Time = np.load("./Expert_data/Time.npy", allow_pickle=True).tolist()
Reward_eval_human = np.load("./Expert_data/Reward_eval_human.npy")
Real_Traj_eval_human = np.load("./Expert_data/Real_Traj_eval_human.npy", allow_pickle=True).tolist()
Real_Reward_eval_human = np.load("./Expert_data/Real_Reward_eval_human.npy", allow_pickle=True).tolist()
Real_Time_eval_human = np.load("./Expert_data/Real_Time_eval_human.npy", allow_pickle=True).tolist()
Coins_location = np.load("./Expert_data/Coins_location.npy")
    
threshold = np.mean(Real_Reward_eval_human)
Rand_traj = 2
TrainingSet = Trajectories[Rand_traj]
Labels = Rotation[Rand_traj]
size_data = len(Trajectories[Rand_traj])
coins_location = Coins_location[Rand_traj,:,:] 

len_trajs = []
for i in range(len(Trajectories)):
    len_trajs.append(len(Trajectories[i]))
    
mean_len_trajs = int(np.mean(len_trajs))


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
    
    coins_location = Coins_location[Rand_traj]
    
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
    plt.title('Human Agent {}, Reward {}'.format(Rand_traj, Reward_eval_human[Rand_traj]))
    plt.show()   

# %%

PPO_IL = []
TRPO_IL = []
UATRPO_IL = []
HPPO_IL = []
HPPO_IL_delay_30_iter = []
HPPO_IL_delay_20_iter = []
HPPO_IL_delay_10_iter = []
HPPO_IL_delay_5_iter = []

for i in range(8):
    with open(f'results/FlatRL/evaluation_PPO_IL_True_GAIL_False_Mixed_False_Foraging_{i}.npy', 'rb') as f:
        PPO_IL.append(np.load(f, allow_pickle=True))
        
    with open(f'results/FlatRL/evaluation_TRPO_IL_True_GAIL_False_Mixed_False_Foraging_{i}.npy', 'rb') as f:
        TRPO_IL.append(np.load(f, allow_pickle=True))
        
    with open(f'results/FlatRL/evaluation_UATRPO_IL_True_GAIL_False_Mixed_False_Foraging_{i}.npy', 'rb') as f:
        UATRPO_IL.append(np.load(f, allow_pickle=True))
        
    # with open(f'results/HRL/evaluation_HPPO_HIL_True_HGAIL_False_Mixed_False_Foraging_{i}.npy', 'rb') as f:
    #     HPPO_IL.append(np.load(f, allow_pickle=True))
        
    with open(f'results/HRL/evaluation_HPPO_HIL_True_delayed_2_Foraging_{i}.npy', 'rb') as f:
        HPPO_IL_delay_20_iter.append(np.load(f, allow_pickle=True))
        
    with open(f'results/HRL/evaluation_HPPO_HIL_True_delayed_5_Foraging_{i}.npy', 'rb') as f:
        HPPO_IL_delay_10_iter.append(np.load(f, allow_pickle=True))
        
    with open(f'results/HRL/evaluation_HPPO_HIL_True_delayed_3_Foraging_{i}.npy', 'rb') as f:
        HPPO_IL_delay_5_iter.append(np.load(f, allow_pickle=True))
        
    with open(f'results/HRL/evaluation_HPPO_HIL_True_delayed_6_Foraging_{i}.npy', 'rb') as f:
        HPPO_IL_delay_30_iter.append(np.load(f, allow_pickle=True))
            
Real_Reward_eval_human = np.load("./Expert_data/Real_Reward_eval_human.npy", allow_pickle=True).tolist()    
threshold = np.mean(Real_Reward_eval_human)

# notes:
    # first HPPO keeps the hierarchy constant and optimizes low level
    # delayed bad hyperparmater (number of updates) choice
    # delayed 2 updates both b and hi every 20 iterations for 10 times
    # delayed 3 updates both b and hi every 5 iterations for 10 times
    # delayed 4 updates both b and hi adaptively for 10 times
    # delayed 5 updates both b and hi every 10 iterations for 10 times
    # delayed 6 updates both b and hi every 30 iterations for 10 times

    
# %%

steps = np.linspace(0,6e6,len(PPO_IL[0]))
Human_average_performance = threshold*np.ones((len(steps),))

fig, ax = plt.subplots()
clrs = sns.color_palette("husl", 9)
ax.plot(steps, PPO_IL[0], label='seed 0', c=clrs[0])
ax.plot(steps, PPO_IL[1], label='seed 1', c=clrs[1])
ax.plot(steps, PPO_IL[2], label='seed 2', c=clrs[2])
ax.plot(steps, PPO_IL[3], label='seed 3', c=clrs[3])
ax.plot(steps, PPO_IL[4], label='seed 4', c=clrs[4])
ax.plot(steps, PPO_IL[5], label='seed 5', c=clrs[5])
ax.plot(steps, PPO_IL[6], label='seed 6', c=clrs[6])
ax.plot(steps, PPO_IL[7], label='seed 7', c=clrs[7])
ax.plot(steps, Human_average_performance, "--", label='Humans', c=clrs[8])
box = ax.get_position()
ax.set_position([box.x0, box.y0 + box.height * 0.20,
                 box.width, box.height * 0.8])
# Put a legend below current axis
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2),
          fancybox=True, shadow=True, ncol=5)
ax.set_ylim([0,300])
ax.set_xlabel('Steps')
ax.set_ylabel('Reward')
ax.set_title('PPO')
plt.savefig('Figures/PPO.pdf', format='pdf')

HPPO_IL = HPPO_IL_delay_20_iter

fig, ax = plt.subplots()
clrs = sns.color_palette("husl", 9)
ax.plot(steps, HPPO_IL[0], label='seed 0', c=clrs[0])
ax.plot(steps, HPPO_IL[1], label='seed 1', c=clrs[1])
ax.plot(steps, HPPO_IL[2], label='seed 2', c=clrs[2])
ax.plot(steps, HPPO_IL[3], label='seed 3', c=clrs[3])
ax.plot(steps, HPPO_IL[4], label='seed 4', c=clrs[4])
ax.plot(steps, HPPO_IL[5], label='seed 5', c=clrs[5])
ax.plot(steps, HPPO_IL[6], label='seed 6', c=clrs[6])
ax.plot(steps, HPPO_IL[7], label='seed 7', c=clrs[7])
ax.plot(steps, Human_average_performance, "--", label='Humans', c=clrs[8])
box = ax.get_position()
ax.set_position([box.x0, box.y0 + box.height * 0.20,
                 box.width, box.height * 0.8])
# Put a legend below current axis
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2),
          fancybox=True, shadow=True, ncol=5)
ax.set_ylim([0,300])
ax.set_xlabel('Steps')
ax.set_ylabel('Reward')
ax.set_title('HPPO')
plt.savefig('Figures/HPPO.pdf', format='pdf')

# %%

fig, ax = plt.subplots()
clrs = sns.color_palette("husl", 9)
ax.plot(steps, TRPO_IL[0], label='seed 0', c=clrs[0])
ax.plot(steps, TRPO_IL[1], label='seed 1', c=clrs[1])
ax.plot(steps, TRPO_IL[2], label='seed 2', c=clrs[2])
ax.plot(steps, TRPO_IL[3], label='seed 3', c=clrs[3])
ax.plot(steps, TRPO_IL[4], label='seed 4', c=clrs[4])
ax.plot(steps, TRPO_IL[5], label='seed 5', c=clrs[5])
ax.plot(steps, TRPO_IL[6], label='seed 6', c=clrs[6])
ax.plot(steps, TRPO_IL[7], label='seed 7', c=clrs[7])
ax.plot(steps, Human_average_performance, "--", label='Humans', c=clrs[8])
box = ax.get_position()
ax.set_position([box.x0, box.y0 + box.height * 0.20,
                 box.width, box.height * 0.8])
# Put a legend below current axis
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2),
          fancybox=True, shadow=True, ncol=5)
ax.set_ylim([0,300])
ax.set_xlabel('Steps')
ax.set_ylabel('Reward')
ax.set_title('TRPO')
plt.savefig('Figures/TRPO.pdf', format='pdf')

fig, ax = plt.subplots()
clrs = sns.color_palette("husl", 9)
ax.plot(steps, UATRPO_IL[0], label='seed 0', c=clrs[0])
ax.plot(steps, UATRPO_IL[1], label='seed 1', c=clrs[1])
ax.plot(steps, UATRPO_IL[2], label='seed 2', c=clrs[2])
ax.plot(steps, UATRPO_IL[3], label='seed 3', c=clrs[3])
ax.plot(steps, UATRPO_IL[4], label='seed 4', c=clrs[4])
ax.plot(steps, UATRPO_IL[5], label='seed 5', c=clrs[5])
ax.plot(steps, UATRPO_IL[6], label='seed 6', c=clrs[6])
ax.plot(steps, UATRPO_IL[7], label='seed 7', c=clrs[7])
ax.plot(steps, Human_average_performance, "--", label='Humans', c=clrs[8])
box = ax.get_position()
ax.set_position([box.x0, box.y0 + box.height * 0.20,
                 box.width, box.height * 0.8])
# Put a legend below current axis
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2),
          fancybox=True, shadow=True, ncol=5)
ax.set_ylim([0,300])
ax.set_xlabel('Steps')
ax.set_ylabel('Reward')
ax.set_title('UATRPO')
plt.savefig('Figures/UATRPO.pdf', format='pdf')

# %%

PPO_mean = np.mean(np.array(PPO_IL),0)
PPO_std = np.std(np.array(PPO_IL),0)

HPPO_IL_delay_5_iter_mean = np.mean(np.array(HPPO_IL_delay_5_iter),0)
HPPO_IL_delay_5_iter_std = np.std(np.array(HPPO_IL_delay_5_iter),0)

HPPO_IL_delay_10_iter_mean = np.mean(np.array(HPPO_IL_delay_10_iter),0)
HPPO_IL_delay_10_iter_std = np.std(np.array(HPPO_IL_delay_10_iter),0)

HPPO_IL_delay_20_iter_mean = np.mean(np.array(HPPO_IL_delay_20_iter),0)
HPPO_IL_delay_20_iter_std = np.std(np.array(HPPO_IL_delay_20_iter),0)

HPPO_IL_delay_30_iter_mean = np.mean(np.array(HPPO_IL_delay_30_iter),0)
HPPO_IL_delay_30_iter_std = np.std(np.array(HPPO_IL_delay_30_iter),0)

TRPO_mean = np.mean(np.array(TRPO_IL),0)
TRPO_std = np.std(np.array(TRPO_IL),0)

UATRPO_mean = np.mean(np.array(UATRPO_IL),0)
UATRPO_std = np.std(np.array(UATRPO_IL),0)

HPPO_mean = HPPO_IL_delay_10_iter_mean
HPPO_std = HPPO_IL_delay_10_iter_std

fig, ax = plt.subplots()
clrs = sns.color_palette("husl", 9)
ax.plot(steps, PPO_mean, label='PPO', c=clrs[0])
ax.fill_between(steps, PPO_mean-PPO_std, PPO_mean+PPO_std, alpha=0.2, facecolor=clrs[0])
ax.plot(steps, HPPO_mean, label='HPPO', c=clrs[4])
ax.fill_between(steps, HPPO_mean-HPPO_std, HPPO_mean+HPPO_std, alpha=0.2, facecolor=clrs[4])
ax.plot(steps, TRPO_mean, label='TRPO', c=clrs[1])
ax.fill_between(steps, TRPO_mean-TRPO_std, TRPO_mean+TRPO_std, alpha=0.2, facecolor=clrs[1])
ax.plot(steps, UATRPO_mean, label='UATRPO', c=clrs[3])
ax.fill_between(steps, UATRPO_mean-UATRPO_std, UATRPO_mean+UATRPO_std, alpha=0.2, facecolor=clrs[3])
ax.plot(steps, Human_average_performance, "--", label='Humans', c=clrs[2])
box = ax.get_position()
ax.set_position([box.x0, box.y0 + box.height * 0.1,
                 box.width, box.height * 0.9])
# Put a legend below current axis
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15),
          fancybox=True, shadow=True, ncol=5)
ax.set_ylim([0,300])
ax.set_xlabel('Steps')
ax.set_ylabel('Reward')
ax.set_title('Comparison')
plt.savefig('Figures/on_policy_comparison.pdf', format='pdf')

fig, ax = plt.subplots()
clrs = sns.color_palette("husl", 9)
ax.plot(steps, PPO_mean, label='PPO', c=clrs[0])
# ax.fill_between(steps, PPO_mean-PPO_std, PPO_mean+PPO_std, alpha=0.2, facecolor=clrs[0])
ax.plot(steps, HPPO_IL_delay_5_iter_mean, label='HPPO 5 iter delay', c=clrs[4])
# ax.fill_between(steps, HPPO_IL_delay_5_iter_mean-HPPO_IL_delay_5_iter_std, HPPO_IL_delay_5_iter_mean+HPPO_IL_delay_5_iter_std, alpha=0.2, facecolor=clrs[4])
ax.plot(steps, HPPO_IL_delay_10_iter_mean, label='HPPO 10 iter delay', c=clrs[1])
# ax.fill_between(steps, HPPO_IL_delay_10_iter_mean-HPPO_IL_delay_10_iter_std, HPPO_IL_delay_10_iter_mean+HPPO_IL_delay_10_iter_std, alpha=0.2, facecolor=clrs[1])
ax.plot(steps, HPPO_IL_delay_20_iter_mean, label='HPPO 20 iter delay', c=clrs[3])
# ax.fill_between(steps, HPPO_IL_delay_20_iter_mean-HPPO_IL_delay_20_iter_std, HPPO_IL_delay_20_iter_mean+HPPO_IL_delay_20_iter_std, alpha=0.2, facecolor=clrs[3])
ax.plot(steps, HPPO_IL_delay_30_iter_mean, label='HPPO 30 iter delay', c=clrs[6])
# ax.fill_between(steps, HPPO_IL_delay_30_iter_mean-HPPO_IL_delay_30_iter_std, HPPO_IL_delay_30_iter_mean+HPPO_IL_delay_30_iter_std, alpha=0.2, facecolor=clrs[6])
ax.plot(steps, Human_average_performance, "--", label='Humans', c=clrs[2])
box = ax.get_position()
ax.set_position([box.x0, box.y0 + box.height * 0.1,
                 box.width, box.height * 0.9])
# Put a legend below current axis
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15),
          fancybox=True, shadow=True, ncol=5)
ax.set_ylim([0,300])
ax.set_xlabel('Steps')
ax.set_ylabel('Reward')
ax.set_title('Comparison')
plt.savefig('Figures/PPO_vs_HPPO.pdf', format='pdf')

# %% Plot HIL

if not os.path.exists("./Figures/HIL_ablation_study"):
    os.makedirs("./Figures/HIL_ablation_study")


coins_array = range(50)

Results_dictionary = {}

HIL_nOptions_1_supervised_False = []
HIL_nOptions_2_supervised_False = []
HIL_nOptions_3_supervised_False = []
HIL_nOptions_2_supervised_True = []
HIL_nOptions_3_supervised_True = []

best_reward_nOptions_1_supervised_False = 0
best_reward_nOptions_2_supervised_False = 0
best_reward_nOptions_3_supervised_False = 0
best_reward_nOptions_2_supervised_True = 0
best_reward_nOptions_3_supervised_True = 0

HIL_nOptions_1_supervised_False_dict = {}
HIL_nOptions_2_supervised_False_dict = {}
HIL_nOptions_3_supervised_False_dict = {}
HIL_nOptions_2_supervised_True_dict = {}
HIL_nOptions_3_supervised_True_dict = {}

Best_nOptions_1_supervised_False_dict = {}
Best_nOptions_2_supervised_False_dict = {}
Best_nOptions_3_supervised_False_dict = {}
Best_nOptions_2_supervised_True_dict = {}
Best_nOptions_3_supervised_True_dict = {}


for trj in coins_array:
    HIL_traj_2_nOptions_1_supervised_False = []
    HIL_traj_2_nOptions_2_supervised_False = []
    HIL_traj_2_nOptions_3_supervised_False = []
    HIL_traj_2_nOptions_2_supervised_True = []
    HIL_traj_2_nOptions_3_supervised_True = []
    
    Expert_Traj_dict = {}
    
    supervised = ["True", "False"]
    
    for supervision in supervised:
        if supervision == "True":
            minNOptions = 2
        else:
            minNOptions = 1
            
        for nOptions in range(minNOptions, 4):
    
            for i in range(8):        
                with open(f'results/HRL/HIL_traj_{trj}_nOptions_{nOptions}_supervised_{supervision}_{i}.npy', 'rb') as f:
                    
                    if nOptions == 1:
                        HIL_traj_2_nOptions_1_supervised_False.append(np.load(f, allow_pickle=True))
                        HIL_nOptions_1_supervised_False.append(HIL_traj_2_nOptions_1_supervised_False[i])
                        
                        reward = HIL_traj_2_nOptions_1_supervised_False[i][-1]
                        
                        if reward > best_reward_nOptions_1_supervised_False:
                            Best_nOptions_1_supervised_False_dict['reward'] = reward
                            Best_nOptions_1_supervised_False_dict['seed'] = i
                            Best_nOptions_1_supervised_False_dict['expert traj'] = trj
                            best_reward_nOptions_1_supervised_False = reward
                        
                        HIL_nOptions_1_supervised_False_dict[f'HIL_traj_{trj}_nOptions_{nOptions}_supervised_{supervision}_{i}'] = HIL_traj_2_nOptions_1_supervised_False[i]                    
                        Expert_Traj_dict[f'HIL_traj_{trj}_nOptions_{nOptions}_supervised_{supervision}_{i}'] = HIL_traj_2_nOptions_1_supervised_False[i]
                        
                    elif nOptions == 2 and supervision == "False":
                        HIL_traj_2_nOptions_2_supervised_False.append(np.load(f, allow_pickle=True))
                        HIL_nOptions_2_supervised_False.append(HIL_traj_2_nOptions_2_supervised_False[i])
                        
                        reward = HIL_traj_2_nOptions_2_supervised_False[i][-1]
                        
                        if reward > best_reward_nOptions_2_supervised_False:
                            Best_nOptions_2_supervised_False_dict['reward'] = reward
                            Best_nOptions_2_supervised_False_dict['seed'] = i
                            Best_nOptions_2_supervised_False_dict['expert traj'] = trj
                            best_reward_nOptions_2_supervised_False = reward
                        
                        HIL_nOptions_2_supervised_False_dict[f'HIL_traj_{trj}_nOptions_{nOptions}_supervised_{supervision}_{i}'] = HIL_traj_2_nOptions_2_supervised_False[i]
                        Expert_Traj_dict[f'HIL_traj_{trj}_nOptions_{nOptions}_supervised_{supervision}_{i}'] = HIL_traj_2_nOptions_2_supervised_False[i]
        
                    elif nOptions == 3 and supervision == "False":
                        HIL_traj_2_nOptions_3_supervised_False.append(np.load(f, allow_pickle=True))
                        HIL_nOptions_3_supervised_False.append(HIL_traj_2_nOptions_3_supervised_False[i])
                        
                        reward = HIL_traj_2_nOptions_3_supervised_False[i][-1]
                        
                        if reward > best_reward_nOptions_3_supervised_False:
                            Best_nOptions_3_supervised_False_dict['reward'] = reward
                            Best_nOptions_3_supervised_False_dict['seed'] = i
                            Best_nOptions_3_supervised_False_dict['expert traj'] = trj
                            best_reward_nOptions_3_supervised_False = reward
                        
                        HIL_nOptions_3_supervised_False_dict[f'HIL_traj_{trj}_nOptions_{nOptions}_supervised_{supervision}_{i}'] = HIL_traj_2_nOptions_3_supervised_False[i]                      
                        Expert_Traj_dict[f'HIL_traj_{trj}_nOptions_{nOptions}_supervised_{supervision}_{i}'] = HIL_traj_2_nOptions_3_supervised_False[i]
                        
                    elif nOptions == 2 and supervision == "True":
                        HIL_traj_2_nOptions_2_supervised_True.append(np.load(f, allow_pickle=True))
                        HIL_nOptions_2_supervised_True.append(HIL_traj_2_nOptions_2_supervised_True[i])
                        
                        reward = HIL_traj_2_nOptions_2_supervised_True[i][-1]
                        
                        if reward > best_reward_nOptions_2_supervised_True:
                            Best_nOptions_2_supervised_True_dict['reward'] = reward
                            Best_nOptions_2_supervised_True_dict['seed'] = i
                            Best_nOptions_2_supervised_True_dict['expert traj'] = trj
                            best_reward_nOptions_2_supervised_True = reward
                        
                        HIL_nOptions_2_supervised_True_dict[f'HIL_traj_{trj}_nOptions_{nOptions}_supervised_{supervision}_{i}'] = HIL_traj_2_nOptions_2_supervised_True[i]                       
                        Expert_Traj_dict[f'HIL_traj_{trj}_nOptions_{nOptions}_supervised_{supervision}_{i}'] = HIL_traj_2_nOptions_2_supervised_True[i]
        
                    elif nOptions == 3 and supervision == "True":
                        HIL_traj_2_nOptions_3_supervised_True.append(np.load(f, allow_pickle=True))
                        HIL_nOptions_3_supervised_True.append(HIL_traj_2_nOptions_3_supervised_True[i])
                        
                        reward = HIL_traj_2_nOptions_3_supervised_True[i][-1]
                        
                        if reward > best_reward_nOptions_3_supervised_True:
                            Best_nOptions_3_supervised_True_dict['reward'] = reward
                            Best_nOptions_3_supervised_True_dict['seed'] = i
                            Best_nOptions_3_supervised_True_dict['expert traj'] = trj
                            best_reward_nOptions_3_supervised_True = reward
                        
                        HIL_nOptions_3_supervised_True_dict[f'HIL_traj_{trj}_nOptions_{nOptions}_supervised_{supervision}_{i}'] = HIL_traj_2_nOptions_3_supervised_True[i]                   
                        Expert_Traj_dict[f'HIL_traj_{trj}_nOptions_{nOptions}_supervised_{supervision}_{i}'] = HIL_traj_2_nOptions_3_supervised_True[i]
                        
                        
    Results_dictionary[f'HIL_Expert_traj_{trj}'] = Expert_Traj_dict
                                
    Real_Reward_eval_human = np.load("./Expert_data/Real_Reward_eval_human.npy", allow_pickle=True).tolist()    
    threshold = np.mean(Real_Reward_eval_human)
    
    HIL_traj_2_nOptions_1_supervised_False_mean = np.mean(np.array(HIL_traj_2_nOptions_1_supervised_False),0)
    HIL_traj_2_nOptions_1_supervised_False_std= np.std(np.array(HIL_traj_2_nOptions_1_supervised_False),0)
    
    HIL_traj_2_nOptions_2_supervised_False_mean = np.mean(np.array(HIL_traj_2_nOptions_2_supervised_False),0)
    HIL_traj_2_nOptions_2_supervised_False_std= np.std(np.array(HIL_traj_2_nOptions_2_supervised_False),0)
    
    HIL_traj_2_nOptions_3_supervised_False_mean = np.mean(np.array(HIL_traj_2_nOptions_3_supervised_False),0)
    HIL_traj_2_nOptions_3_supervised_False_std= np.std(np.array(HIL_traj_2_nOptions_3_supervised_False),0)
    
    HIL_traj_2_nOptions_2_supervised_True_mean = np.mean(np.array(HIL_traj_2_nOptions_2_supervised_True),0)
    HIL_traj_2_nOptions_2_supervised_True_std= np.std(np.array(HIL_traj_2_nOptions_2_supervised_True),0)
    
    HIL_traj_2_nOptions_3_supervised_True_mean = np.mean(np.array(HIL_traj_2_nOptions_3_supervised_True),0)
    HIL_traj_2_nOptions_3_supervised_True_std= np.std(np.array(HIL_traj_2_nOptions_3_supervised_True),0)
    
    BW_iters = np.linspace(0,10,len(HIL_traj_2_nOptions_1_supervised_False[0]))
    Human_average_performance = threshold*np.ones((len(BW_iters),))
    
    # fig, ax = plt.subplots()
    # clrs = sns.color_palette("husl", 9)
    # ax.plot(BW_iters, HIL_traj_2_nOptions_1_supervised_False[0], label='seed 0', c=clrs[0])
    # ax.plot(BW_iters, HIL_traj_2_nOptions_1_supervised_False[1], label='seed 1', c=clrs[1])
    # ax.plot(BW_iters, HIL_traj_2_nOptions_1_supervised_False[2], label='seed 2', c=clrs[2])
    # ax.plot(BW_iters, HIL_traj_2_nOptions_1_supervised_False[3], label='seed 3', c=clrs[3])
    # ax.plot(BW_iters, HIL_traj_2_nOptions_1_supervised_False[4], label='seed 4', c=clrs[4])
    # ax.plot(BW_iters, HIL_traj_2_nOptions_1_supervised_False[5], label='seed 5', c=clrs[5])
    # ax.plot(BW_iters, HIL_traj_2_nOptions_1_supervised_False[6], label='seed 6', c=clrs[6])
    # ax.plot(BW_iters, HIL_traj_2_nOptions_1_supervised_False[7], label='seed 7', c=clrs[7])
    # ax.plot(BW_iters, Human_average_performance, "--", label='Humans', c=clrs[8])
    # box = ax.get_position()
    # ax.set_position([box.x0, box.y0 + box.height * 0.20,
    #                  box.width, box.height * 0.8])
    # # Put a legend below current axis
    # ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2),
    #           fancybox=True, shadow=True, ncol=5)
    # ax.set_ylim([0,300])
    # ax.set_xlabel('Iterations')
    # ax.set_ylabel('Reward')
    # ax.set_title('HIL 2 options')
    # plt.savefig('Figures/HIL_ablation_study/HIL_traj_2_nOptions_1_supervised_False.pdf', format='pdf')
    
    # fig, ax = plt.subplots()
    # clrs = sns.color_palette("husl", 9)
    # ax.plot(BW_iters, HIL_traj_2_nOptions_1_supervised_False_mean, label='HIL, traj: 2, nOptions: 1, supervised: False', c=clrs[0])
    # ax.fill_between(BW_iters, HIL_traj_2_nOptions_1_supervised_False_mean-HIL_traj_2_nOptions_1_supervised_False_std, HIL_traj_2_nOptions_1_supervised_False_mean+HIL_traj_2_nOptions_1_supervised_False_std, alpha=0.2, facecolor=clrs[0])
    # ax.plot(BW_iters, HIL_traj_2_nOptions_2_supervised_False_mean, label='HIL, traj: 2, nOptions: 2, supervised: False', c=clrs[1])
    # ax.fill_between(BW_iters, HIL_traj_2_nOptions_2_supervised_False_mean-HIL_traj_2_nOptions_2_supervised_False_std, HIL_traj_2_nOptions_2_supervised_False_mean+HIL_traj_2_nOptions_2_supervised_False_std, alpha=0.2, facecolor=clrs[1])
    # ax.plot(BW_iters, HIL_traj_2_nOptions_3_supervised_False_mean, label='HIL, traj: 2, nOptions: 3, supervised: False', c=clrs[2])
    # ax.fill_between(BW_iters, HIL_traj_2_nOptions_3_supervised_False_mean-HIL_traj_2_nOptions_3_supervised_False_std, HIL_traj_2_nOptions_3_supervised_False_mean+HIL_traj_2_nOptions_3_supervised_False_std, alpha=0.2, facecolor=clrs[2])
    # ax.plot(BW_iters, HIL_traj_2_nOptions_2_supervised_True_mean, label='HIL, traj: 2, nOptions: 2, supervised: True', c=clrs[3])
    # ax.fill_between(BW_iters, HIL_traj_2_nOptions_2_supervised_True_mean-HIL_traj_2_nOptions_2_supervised_True_std, HIL_traj_2_nOptions_2_supervised_True_mean+HIL_traj_2_nOptions_2_supervised_True_std, alpha=0.2, facecolor=clrs[3])
    # ax.plot(BW_iters, HIL_traj_2_nOptions_3_supervised_True_mean, label='HIL, traj: 2, nOptions: 3, supervised: True', c=clrs[4])
    # ax.fill_between(BW_iters, HIL_traj_2_nOptions_3_supervised_True_mean-HIL_traj_2_nOptions_3_supervised_True_std, HIL_traj_2_nOptions_3_supervised_True_mean+HIL_traj_2_nOptions_3_supervised_True_std, alpha=0.2, facecolor=clrs[4])
    # ax.plot(BW_iters, Human_average_performance, "--", label='Humans', c=clrs[8])
    # box = ax.get_position()
    # ax.set_position([box.x0, box.y0 + box.height * 0.1,
    #                  box.width, box.height * 0.9])
    # # Put a legend below current axis
    # ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15),
    #           fancybox=True, shadow=True, ncol=3)
    # ax.set_ylim([0,300])
    # ax.set_xlabel('Steps')
    # ax.set_ylabel('Reward')
    # ax.set_title('Comparison')
    # plt.savefig('Figures/HIL_ablation_study/HIL_comparison_std_on.pdf', format='pdf', bbox_inches='tight')
    
    fig, ax = plt.subplots(figsize=(8,5))
    clrs = sns.color_palette("husl", 9)
    ax.plot(BW_iters, HIL_traj_2_nOptions_1_supervised_False_mean, label='HIL, traj: 2, nOptions: 1, supervised: False', c=clrs[0])
    # ax.fill_between(BW_iters, HIL_traj_2_nOptions_1_supervised_False_mean-HIL_traj_2_nOptions_1_supervised_False_std, HIL_traj_2_nOptions_1_supervised_False_mean+HIL_traj_2_nOptions_1_supervised_False_std, alpha=0.2, facecolor=clrs[0])
    ax.plot(BW_iters, HIL_traj_2_nOptions_2_supervised_False_mean, label='HIL, traj: 2, nOptions: 2, supervised: False', c=clrs[1])
    # ax.fill_between(BW_iters, HIL_traj_2_nOptions_2_supervised_False_mean-HIL_traj_2_nOptions_2_supervised_False_std, HIL_traj_2_nOptions_2_supervised_False_mean+HIL_traj_2_nOptions_2_supervised_False_std, alpha=0.2, facecolor=clrs[1])
    ax.plot(BW_iters, HIL_traj_2_nOptions_3_supervised_False_mean, label='HIL, traj: 2, nOptions: 3, supervised: False', c=clrs[2])
    # ax.fill_between(BW_iters, HIL_traj_2_nOptions_3_supervised_False_mean-HIL_traj_2_nOptions_3_supervised_False_std, HIL_traj_2_nOptions_3_supervised_False_mean+HIL_traj_2_nOptions_3_supervised_False_std, alpha=0.2, facecolor=clrs[2])
    ax.plot(BW_iters, HIL_traj_2_nOptions_2_supervised_True_mean, label='HIL, traj: 2, nOptions: 2, supervised: True', c=clrs[3])
    # ax.fill_between(BW_iters, HIL_traj_2_nOptions_2_supervised_True_mean-HIL_traj_2_nOptions_2_supervised_True_std, HIL_traj_2_nOptions_2_supervised_True_mean+HIL_traj_2_nOptions_2_supervised_True_std, alpha=0.2, facecolor=clrs[3])
    ax.plot(BW_iters, HIL_traj_2_nOptions_3_supervised_True_mean, label='HIL, traj: 2, nOptions: 3, supervised: True', c=clrs[4])
    # ax.fill_between(BW_iters, HIL_traj_2_nOptions_3_supervised_True_mean-HIL_traj_2_nOptions_3_supervised_True_std, HIL_traj_2_nOptions_3_supervised_True_mean+HIL_traj_2_nOptions_3_supervised_True_std, alpha=0.2, facecolor=clrs[4])
    ax.plot(BW_iters, Human_average_performance, "--", label='Humans', c=clrs[8])
    box = ax.get_position()
    ax.set_position([box.x0, box.y0 + box.height * 0.1,
                     box.width, box.height * 0.9])
    # Put a legend below current axis
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15),
              fancybox=True, shadow=True, ncol=1)
    ax.set_ylim([0,300])
    ax.set_xlabel('Steps')
    ax.set_ylabel('Reward')
    ax.set_title('Comparison')
    plt.savefig(f'Figures/HIL_ablation_study/HIL_traj_{trj}_comparison_std_off.pdf', format='pdf', bbox_inches='tight')
    
 
Results_dictionary['HIL_nOptions_1_supervised_False'] = HIL_nOptions_1_supervised_False_dict
Results_dictionary['HIL_nOptions_2_supervised_False'] = HIL_nOptions_2_supervised_False_dict
Results_dictionary['HIL_nOptions_3_supervised_False'] = HIL_nOptions_3_supervised_False_dict
Results_dictionary['HIL_nOptions_2_supervised_True'] = HIL_nOptions_2_supervised_True_dict
Results_dictionary['HIL_nOptions_3_supervised_True'] = HIL_nOptions_3_supervised_True_dict

Results_dictionary['Best_nOptions_1_supervised_False'] = Best_nOptions_1_supervised_False_dict
Results_dictionary['Best_nOptions_2_supervised_False'] = Best_nOptions_2_supervised_False_dict
Results_dictionary['Best_nOptions_3_supervised_False'] = Best_nOptions_3_supervised_False_dict
Results_dictionary['Best_nOptions_2_supervised_True'] = Best_nOptions_2_supervised_True_dict
Results_dictionary['Best_nOptions_3_supervised_True'] = Best_nOptions_3_supervised_True_dict

    
HIL_nOptions_1_supervised_False_mean = np.mean(np.array(HIL_nOptions_1_supervised_False),0)
HIL_nOptions_1_supervised_False_std= np.std(np.array(HIL_nOptions_1_supervised_False),0)

HIL_nOptions_2_supervised_False_mean = np.mean(np.array(HIL_nOptions_2_supervised_False),0)
HIL_nOptions_2_supervised_False_std= np.std(np.array(HIL_nOptions_2_supervised_False),0)

HIL_nOptions_3_supervised_False_mean = np.mean(np.array(HIL_nOptions_3_supervised_False),0)
HIL_nOptions_3_supervised_False_std= np.std(np.array(HIL_nOptions_3_supervised_False),0)

HIL_nOptions_2_supervised_True_mean = np.mean(np.array(HIL_nOptions_2_supervised_True),0)
HIL_nOptions_2_supervised_True_std= np.std(np.array(HIL_nOptions_2_supervised_True),0)

HIL_nOptions_3_supervised_True_mean = np.mean(np.array(HIL_nOptions_3_supervised_True),0)
HIL_nOptions_3_supervised_True_std= np.std(np.array(HIL_nOptions_3_supervised_True),0)
    
fig, ax = plt.subplots(figsize=(8,5))
clrs = sns.color_palette("husl", 9)
ax.plot(BW_iters, HIL_nOptions_1_supervised_False_mean, label='HIL, nOptions: 1, supervised: False', c=clrs[0])
# ax.fill_between(BW_iters, HIL_traj_2_nOptions_1_supervised_False_mean-HIL_traj_2_nOptions_1_supervised_False_std, HIL_traj_2_nOptions_1_supervised_False_mean+HIL_traj_2_nOptions_1_supervised_False_std, alpha=0.2, facecolor=clrs[0])
ax.plot(BW_iters, HIL_nOptions_2_supervised_False_mean, label='HIL, nOptions: 2, supervised: False', c=clrs[1])
# ax.fill_between(BW_iters, HIL_traj_2_nOptions_2_supervised_False_mean-HIL_traj_2_nOptions_2_supervised_False_std, HIL_traj_2_nOptions_2_supervised_False_mean+HIL_traj_2_nOptions_2_supervised_False_std, alpha=0.2, facecolor=clrs[1])
ax.plot(BW_iters, HIL_nOptions_3_supervised_False_mean, label='HIL, nOptions: 3, supervised: False', c=clrs[2])
# ax.fill_between(BW_iters, HIL_traj_2_nOptions_3_supervised_False_mean-HIL_traj_2_nOptions_3_supervised_False_std, HIL_traj_2_nOptions_3_supervised_False_mean+HIL_traj_2_nOptions_3_supervised_False_std, alpha=0.2, facecolor=clrs[2])
ax.plot(BW_iters, HIL_nOptions_2_supervised_True_mean, label='HIL, nOptions: 2, supervised: True', c=clrs[3])
# ax.fill_between(BW_iters, HIL_traj_2_nOptions_2_supervised_True_mean-HIL_traj_2_nOptions_2_supervised_True_std, HIL_traj_2_nOptions_2_supervised_True_mean+HIL_traj_2_nOptions_2_supervised_True_std, alpha=0.2, facecolor=clrs[3])
ax.plot(BW_iters, HIL_nOptions_3_supervised_True_mean, label='HIL, nOptions: 3, supervised: True', c=clrs[4])
# ax.fill_between(BW_iters, HIL_traj_2_nOptions_3_supervised_True_mean-HIL_traj_2_nOptions_3_supervised_True_std, HIL_traj_2_nOptions_3_supervised_True_mean+HIL_traj_2_nOptions_3_supervised_True_std, alpha=0.2, facecolor=clrs[4])
ax.plot(BW_iters, Human_average_performance, "--", label='Humans', c=clrs[8])
box = ax.get_position()
ax.set_position([box.x0, box.y0 + box.height * 0.1,
                 box.width, box.height * 0.9])
# Put a legend below current axis
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15),
          fancybox=True, shadow=True, ncol=1)
ax.set_ylim([0,300])
ax.set_xlabel('Steps')
ax.set_ylabel('Reward')
ax.set_title('Comparison')
plt.savefig('Figures/HIL_ablation_study/HIL_over_trajs_comparison_std_off.pdf', format='pdf', bbox_inches='tight')



# %%

def save_obj(obj, name):
    with open('results/'+ name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name):
    with open('results/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)

if not os.path.exists("./results/HIL_ablation_study"):
    os.makedirs("./results/HIL_ablation_study")

save_obj(Results_dictionary, 'HIL_ablation_study/Sorted_results')


# %%

HIL_best_nOptions_1 = {}

for trj in coins_array:
    HIL_nOptions_1_supervised_False = []
    HIL_best_nOptions_1_traj = {}
    
    HIL_best_nOptions_1[f'HIL_traj_{trj}_nOptions_1_supervised_False'] = HIL_best_nOptions_1_traj
    
    for i in range(8):        
        with open(f'results/HRL/HIL_traj_{trj}_nOptions_1_supervised_False_{i}.npy', 'rb') as f:
            HIL_nOptions_1_supervised_False.append(np.load(f, allow_pickle=True))
    
    reward_array = np.array(HIL_nOptions_1_supervised_False)
    
    best_seed = np.argmax(reward_array[:,-1])
    
    HIL_best_nOptions_1[f'HIL_traj_{trj}_nOptions_1_supervised_False']['best_seed'] = best_seed
    HIL_best_nOptions_1[f'HIL_traj_{trj}_nOptions_1_supervised_False']['traj'] = trj
    

save_obj(HIL_best_nOptions_1, 'HIL_ablation_study/Best_results_nOptions_1')
# %%
IL_RL_best_nOptions_1 = {}

coins_array = range(50)

PPO_RL_AllHumans = [[[] for i in range(8)] for coin in coins_array]

j=0
for human in coins_array:
    IL_RL_best_nOptions_1_traj = {}
    IL_RL_best_nOptions_1[f'IL_RL_traj_{human}'] = IL_RL_best_nOptions_1_traj
    
    for i in range(8):
        with open(f'results/HRL/evaluation_PPO_HIL_True_traj_{human}_nOptions_1_supervised_False_{i}.npy', 'rb') as f:
            PPO_RL_AllHumans[j][i] = np.load(f, allow_pickle=True)
            
    reward_array = np.array(PPO_RL_AllHumans[j])
    best_seed = np.argmax(reward_array[:,-1])
    
    IL_RL_best_nOptions_1[f'IL_RL_traj_{human}']['best_seed'] = best_seed
    IL_RL_best_nOptions_1[f'IL_RL_traj_{human}']['traj'] = human
            
    j+=1

save_obj(IL_RL_best_nOptions_1, 'IL_RL_best_nOptions_1')

# %% Allocentric only

HIL_allocentric_only_best_nOptions_1 = {}

for trj in coins_array:
    HIL_allocentric_only_nOptions_1_supervised_False = []
    HIL_allocentric_only_best_nOptions_1_traj = {}
    
    HIL_allocentric_only_best_nOptions_1[f'HIL_allocentric_only_traj_{trj}_nOptions_1_supervised_False'] = HIL_allocentric_only_best_nOptions_1_traj
    
    for i in range(8):        
        with open(f'results/HRL/HIL_allocentric_only_traj_{trj}_nOptions_1_supervised_False_{i}.npy', 'rb') as f:
            HIL_allocentric_only_nOptions_1_supervised_False.append(np.load(f, allow_pickle=True))
    
    reward_array = np.array(HIL_allocentric_only_nOptions_1_supervised_False)
    
    best_seed = np.argmax(reward_array[:,-1])
    
    HIL_allocentric_only_best_nOptions_1[f'HIL_allocentric_only_traj_{trj}_nOptions_1_supervised_False']['best_seed'] = best_seed
    HIL_allocentric_only_best_nOptions_1[f'HIL_allocentric_only_traj_{trj}_nOptions_1_supervised_False']['traj'] = trj
    

save_obj(HIL_allocentric_only_best_nOptions_1, 'HIL_ablation_study/Best_results_allocentric_only_nOptions_1')

# %% Egocentric only

HIL_egocentric_only_best_nOptions_1 = {}

coins_array = range(50)

for trj in coins_array:
    HIL_egocentric_only_nOptions_1_supervised_False = []
    HIL_egocentric_only_best_nOptions_1_traj = {}
    
    HIL_egocentric_only_best_nOptions_1[f'HIL_egocentric_only_traj_{trj}_nOptions_1_supervised_False'] = HIL_egocentric_only_best_nOptions_1_traj
    
    for i in range(8):        
        with open(f'results/HRL/HIL_egocentric_only_traj_{trj}_nOptions_1_supervised_False_{i}.npy', 'rb') as f:
            HIL_egocentric_only_nOptions_1_supervised_False.append(np.load(f, allow_pickle=True))
    
    reward_array = np.array(HIL_egocentric_only_nOptions_1_supervised_False)
    
    best_seed = np.argmax(reward_array[:,-1])
    
    HIL_egocentric_only_best_nOptions_1[f'HIL_egocentric_only_traj_{trj}_nOptions_1_supervised_False']['best_seed'] = best_seed
    HIL_egocentric_only_best_nOptions_1[f'HIL_egocentric_only_traj_{trj}_nOptions_1_supervised_False']['traj'] = trj
    

save_obj(HIL_egocentric_only_best_nOptions_1, 'HIL_ablation_study/Best_results_egocentric_only_nOptions_1')
# %% Allocentric only

IL_RL_allocentric_only_best_nOptions_1 = {}

coins_array = range(50)

PPO_RL_allocentric_only_AllHumans = [[[] for i in range(8)] for coin in coins_array]

j=0
for human in coins_array:
    IL_RL_allocentric_only_best_nOptions_1_traj = {}
    IL_RL_allocentric_only_best_nOptions_1[f'IL_RL_allocentric_only_traj_{human}'] = IL_RL_allocentric_only_best_nOptions_1_traj
    
    for i in range(8):
        with open(f'results/HRL/evaluation_PPO_allocentric_only_HIL_True_traj_{human}_nOptions_1_supervised_False_{i}.npy', 'rb') as f:
            PPO_RL_allocentric_only_AllHumans[j][i] = np.load(f, allow_pickle=True)
            
    reward_array = np.array(PPO_RL_allocentric_only_AllHumans[j])
    best_seed = np.argmax(reward_array[:,-1])
    
    IL_RL_allocentric_only_best_nOptions_1[f'IL_RL_allocentric_only_traj_{human}']['best_seed'] = best_seed
    IL_RL_allocentric_only_best_nOptions_1[f'IL_RL_allocentric_only_traj_{human}']['traj'] = human
            
    j+=1

save_obj(IL_RL_best_nOptions_1, 'IL_RL_allocentric_only_best_nOptions_1')
# %%

def save_obj(obj, name):
    with open('results/'+ name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name):
    with open('results/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)

if not os.path.exists("./results/HIL_ablation_study"):
    os.makedirs("./results/HIL_ablation_study")

Results_Best_HIL_and_HRL = {}
Results_Best_HIL_and_HRL["nOptions_1_supervised_False"] = 7
Results_Best_HIL_and_HRL["nOptions_2_supervised_True"] = 6
Results_Best_HIL_and_HRL["nOptions_2_supervised_False"] = 0
Results_Best_HIL_and_HRL["nOptions_3_supervised_True"] = 2
Results_Best_HIL_and_HRL["nOptions_3_supervised_False"] = 7

save_obj(Results_Best_HIL_and_HRL, 'Results_Best_HIL_and_HRL')