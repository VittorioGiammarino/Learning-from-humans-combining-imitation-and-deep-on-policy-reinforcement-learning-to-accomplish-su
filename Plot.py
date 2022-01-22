#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 15 16:37:18 2021

@author: vittorio
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import matplotlib.patches as ptch
import pickle

def load_obj(name):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)

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

# %% Plot top view of the environment

coins_location = Coins_location[0]

wall_1 = [[-11,-10+i*0.1] for i in range(201)]
wall_2 = [[-10+i*0.1,11] for i in range(201)]
wall_3 = [[11,10-i*0.1] for i in range(201)]
wall_4 = [[-10+i*0.1,-11] for i in range(201)]


sigma1 = 0.5
circle1 = ptch.Circle((6, 7.5), 2*sigma1, color='k',  fill=False)
sigma2 = 1.1
circle2 = ptch.Circle((-1.5, -5), 2*sigma2, color='k',  fill=False)
sigma3 = 1.8
circle3 = ptch.Circle((-5, 3), 2*sigma3, color='k',  fill=False)
sigma4 = 1.3
circle4 = ptch.Circle((4.9, -4), 2*sigma4, color='k',  fill=False)
fig, ax = plt.subplots(squeeze = True, figsize=(5,5))
ax.add_artist(circle1)
ax.add_artist(circle2)
ax.add_artist(circle3)
ax.add_artist(circle4)

for i in range(len(wall_1)):
    ax.plot([wall_1[i][0], wall_1[i][0]+1, wall_1[i][0]+1, wall_1[i][0], wall_1[i][0]],
             [wall_1[i][1], wall_1[i][1], wall_1[i][1]+0.1, wall_1[i][1]+0.1, wall_1[i][1]],'y-')

for i in range(len(wall_1)):
    ax.fill([wall_1[i][0], wall_1[i][0]+1, wall_1[i][0]+1, wall_1[i][0], wall_1[i][0]],
             [wall_1[i][1], wall_1[i][1], wall_1[i][1]+0.1, wall_1[i][1]+0.1, wall_1[i][1]],'k')
    
for i in range(len(wall_2)):
    ax.plot([wall_2[i][0], wall_2[i][0]+0.1, wall_2[i][0]+0.1, wall_2[i][0], wall_2[i][0]],
             [wall_2[i][1], wall_2[i][1], wall_2[i][1]-1, wall_2[i][1]-1, wall_2[i][1]],'r-')

for i in range(len(wall_2)):
    ax.fill([wall_2[i][0], wall_2[i][0]+0.1, wall_2[i][0]+0.1, wall_2[i][0], wall_2[i][0]],
             [wall_2[i][1], wall_2[i][1], wall_2[i][1]-1, wall_2[i][1]-1, wall_2[i][1]],'k')
    
for i in range(len(wall_3)):
    ax.plot([wall_3[i][0], wall_3[i][0]-1, wall_3[i][0]-1, wall_3[i][0], wall_3[i][0]],
             [wall_3[i][1], wall_3[i][1], wall_3[i][1]-0.1, wall_3[i][1]-0.1, wall_3[i][1]],'-', c= 'tab:gray')

for i in range(len(wall_3)):
    ax.fill([wall_3[i][0], wall_3[i][0]-1, wall_3[i][0]-1, wall_3[i][0], wall_3[i][0]],
             [wall_3[i][1], wall_3[i][1], wall_3[i][1]-0.1, wall_3[i][1]-0.1, wall_3[i][1]],'k')
    
for i in range(len(wall_4)):
    ax.plot([wall_4[i][0], wall_4[i][0]+0.1, wall_4[i][0]+0.1, wall_4[i][0], wall_4[i][0]],
             [wall_4[i][1], wall_4[i][1], wall_4[i][1]+1, wall_4[i][1]+1, wall_4[i][1]],'-', c = 'tab:brown')

for i in range(len(wall_4)):
    ax.fill([wall_4[i][0], wall_4[i][0]+0.1, wall_4[i][0]+0.1, wall_4[i][0], wall_4[i][0]],
             [wall_4[i][1], wall_4[i][1], wall_4[i][1]+1, wall_4[i][1]+1, wall_4[i][1]],'k')
    

plt.plot(0.1*coins_location[:,0], 0.1*coins_location[:,1], 'xb')
plt.axis('off')
plt.savefig('Figures/Environment_top_view.pdf', bbox_inches='tight', format='pdf')  
plt.show()  

# %% Plot all humans trajectories Top View

coins_array = range(50)

wall_1 = [[-11,-10+i*0.1] for i in range(201)]
wall_2 = [[-10+i*0.1,11] for i in range(201)]
wall_3 = [[11,10-i*0.1] for i in range(201)]
wall_4 = [[-10+i*0.1,-11] for i in range(201)]

# selected by inspection considering: (i) human performance, (ii) at least a trajectory from each subject, (iii) not too many lost info in the preprocessing step

columns = 5
rows = 10
fig, ax_array = plt.subplots(rows, columns, squeeze=False, figsize=(25,45))
x=0
for k,ax_row in enumerate(ax_array):
    for j,axes in enumerate(ax_row):
        
        
        coins_location = Coins_location[coins_array[x]]
        Rand_traj = coins_array[x]
        time = np.linspace(0,480,len(Real_Traj_eval_human[Rand_traj][:,0])) 
        sigma1 = 0.5
        circle1 = ptch.Circle((6, 7.5), 2*sigma1, color='k',  fill=False)
        sigma2 = 1.1
        circle2 = ptch.Circle((-1.5, -5), 2*sigma2, color='k',  fill=False)
        sigma3 = 1.8
        circle3 = ptch.Circle((-5, 3), 2*sigma3, color='k',  fill=False)
        sigma4 = 1.3
        circle4 = ptch.Circle((4.9, -4), 2*sigma4, color='k',  fill=False)
        axes.add_artist(circle1)
        axes.add_artist(circle2)
        axes.add_artist(circle3)
        axes.add_artist(circle4)
        
        for i in range(len(wall_1)):
            axes.plot([wall_1[i][0], wall_1[i][0]+1, wall_1[i][0]+1, wall_1[i][0], wall_1[i][0]],
                      [wall_1[i][1], wall_1[i][1], wall_1[i][1]+0.1, wall_1[i][1]+0.1, wall_1[i][1]],'y-')
        
        for i in range(len(wall_1)):
            axes.fill([wall_1[i][0], wall_1[i][0]+1, wall_1[i][0]+1, wall_1[i][0], wall_1[i][0]],
                      [wall_1[i][1], wall_1[i][1], wall_1[i][1]+0.1, wall_1[i][1]+0.1, wall_1[i][1]],'k')
            
        for i in range(len(wall_2)):
            axes.plot([wall_2[i][0], wall_2[i][0]+0.1, wall_2[i][0]+0.1, wall_2[i][0], wall_2[i][0]],
                      [wall_2[i][1], wall_2[i][1], wall_2[i][1]-1, wall_2[i][1]-1, wall_2[i][1]],'r-')
        
        for i in range(len(wall_2)):
            axes.fill([wall_2[i][0], wall_2[i][0]+0.1, wall_2[i][0]+0.1, wall_2[i][0], wall_2[i][0]],
                      [wall_2[i][1], wall_2[i][1], wall_2[i][1]-1, wall_2[i][1]-1, wall_2[i][1]],'k')
            
        for i in range(len(wall_3)):
            axes.plot([wall_3[i][0], wall_3[i][0]-1, wall_3[i][0]-1, wall_3[i][0], wall_3[i][0]],
                      [wall_3[i][1], wall_3[i][1], wall_3[i][1]-0.1, wall_3[i][1]-0.1, wall_3[i][1]],'-', c= 'tab:gray')
        
        for i in range(len(wall_3)):
            axes.fill([wall_3[i][0], wall_3[i][0]-1, wall_3[i][0]-1, wall_3[i][0], wall_3[i][0]],
                      [wall_3[i][1], wall_3[i][1], wall_3[i][1]-0.1, wall_3[i][1]-0.1, wall_3[i][1]],'k')
            
        for i in range(len(wall_4)):
            axes.plot([wall_4[i][0], wall_4[i][0]+0.1, wall_4[i][0]+0.1, wall_4[i][0], wall_4[i][0]],
                      [wall_4[i][1], wall_4[i][1], wall_4[i][1]+1, wall_4[i][1]+1, wall_4[i][1]],'-', c = 'tab:brown')
        
        for i in range(len(wall_4)):
            axes.fill([wall_4[i][0], wall_4[i][0]+0.1, wall_4[i][0]+0.1, wall_4[i][0], wall_4[i][0]],
                      [wall_4[i][1], wall_4[i][1], wall_4[i][1]+1, wall_4[i][1]+1, wall_4[i][1]],'k')
            
        plot_data = axes.scatter(0.1*Real_Traj_eval_human[Rand_traj][0:,0], 0.1*Real_Traj_eval_human[Rand_traj][0:,1], c=time, marker='o', cmap='cool') #-1], marker='o', cmap='cool')
        axes.plot(0.1*coins_location[:,0], 0.1*coins_location[:,1], 'xb')
        # cbar = fig.colorbar(plot_data, ticks=[10, 100, 200, 300, 400, 500])
        # cbar.ax.set_yticklabels(['time = 0', 'time = 100', 'time = 200', 'time = 300', 'time = 400', 'time = 500'])
        axes.title.set_text(f'Traj {coins_array[x]+1}')
        axes.axis('off') 
        
        x+=1
  
plt.savefig('Figures/HIL_ablation_real_Trajs.jpg', bbox_inches='tight', format='jpg')    
plt.show()

# %% Plot some human trajectories

coins_array = [13, 40]

# selected by inspection considering: (i) human performance, (ii) at least a trajectory from each subject, (iii) not too many lost info in the preprocessing step

columns = 2
rows = 1
fig, ax_array = plt.subplots(rows, columns, squeeze=False, figsize=(10,5))
x=0
for k,ax_row in enumerate(ax_array):
    for j,axes in enumerate(ax_row):
        
        
        coins_location = Coins_location[coins_array[x]]
        Rand_traj = coins_array[x]
        time = np.linspace(0,480,len(Real_Traj_eval_human[Rand_traj][:,0])) 
        sigma1 = 0.5
        circle1 = ptch.Circle((6, 7.5), 2*sigma1, color='k',  fill=False)
        sigma2 = 1.1
        circle2 = ptch.Circle((-1.5, -5), 2*sigma2, color='k',  fill=False)
        sigma3 = 1.8
        circle3 = ptch.Circle((-5, 3), 2*sigma3, color='k',  fill=False)
        sigma4 = 1.3
        circle4 = ptch.Circle((4.9, -4), 2*sigma4, color='k',  fill=False)
        axes.add_artist(circle1)
        axes.add_artist(circle2)
        axes.add_artist(circle3)
        axes.add_artist(circle4)
        
        for i in range(len(wall_1)):
            axes.plot([wall_1[i][0], wall_1[i][0]+1, wall_1[i][0]+1, wall_1[i][0], wall_1[i][0]],
                      [wall_1[i][1], wall_1[i][1], wall_1[i][1]+0.1, wall_1[i][1]+0.1, wall_1[i][1]],'y-')
        
        for i in range(len(wall_1)):
            axes.fill([wall_1[i][0], wall_1[i][0]+1, wall_1[i][0]+1, wall_1[i][0], wall_1[i][0]],
                      [wall_1[i][1], wall_1[i][1], wall_1[i][1]+0.1, wall_1[i][1]+0.1, wall_1[i][1]],'k')
            
        for i in range(len(wall_2)):
            axes.plot([wall_2[i][0], wall_2[i][0]+0.1, wall_2[i][0]+0.1, wall_2[i][0], wall_2[i][0]],
                      [wall_2[i][1], wall_2[i][1], wall_2[i][1]-1, wall_2[i][1]-1, wall_2[i][1]],'r-')
        
        for i in range(len(wall_2)):
            axes.fill([wall_2[i][0], wall_2[i][0]+0.1, wall_2[i][0]+0.1, wall_2[i][0], wall_2[i][0]],
                      [wall_2[i][1], wall_2[i][1], wall_2[i][1]-1, wall_2[i][1]-1, wall_2[i][1]],'k')
            
        for i in range(len(wall_3)):
            axes.plot([wall_3[i][0], wall_3[i][0]-1, wall_3[i][0]-1, wall_3[i][0], wall_3[i][0]],
                      [wall_3[i][1], wall_3[i][1], wall_3[i][1]-0.1, wall_3[i][1]-0.1, wall_3[i][1]],'-', c= 'tab:gray')
        
        for i in range(len(wall_3)):
            axes.fill([wall_3[i][0], wall_3[i][0]-1, wall_3[i][0]-1, wall_3[i][0], wall_3[i][0]],
                      [wall_3[i][1], wall_3[i][1], wall_3[i][1]-0.1, wall_3[i][1]-0.1, wall_3[i][1]],'k')
            
        for i in range(len(wall_4)):
            axes.plot([wall_4[i][0], wall_4[i][0]+0.1, wall_4[i][0]+0.1, wall_4[i][0], wall_4[i][0]],
                      [wall_4[i][1], wall_4[i][1], wall_4[i][1]+1, wall_4[i][1]+1, wall_4[i][1]],'-', c = 'tab:brown')
        
        for i in range(len(wall_4)):
            axes.fill([wall_4[i][0], wall_4[i][0]+0.1, wall_4[i][0]+0.1, wall_4[i][0], wall_4[i][0]],
                      [wall_4[i][1], wall_4[i][1], wall_4[i][1]+1, wall_4[i][1]+1, wall_4[i][1]],'k')
            
        plot_data = axes.scatter(0.1*Real_Traj_eval_human[Rand_traj][0:,0], 0.1*Real_Traj_eval_human[Rand_traj][0:,1], c=time, marker='o', cmap='cool') #-1], marker='o', cmap='cool')
        axes.plot(0.1*coins_location[:,0], 0.1*coins_location[:,1], 'xb')
        # axes.title.set_text('Human Agent {}, Trial N {}, Reward {}, ({})'.format(int(Rand_traj/10) + 1, Rand_traj%10 + 1, Real_Reward_eval_human[Rand_traj], coins_array[i]))
        axes.title.set_text(f'Traj {coins_array[x]+1}')
        axes.axis('off') 
        
        x+=1

fig.subplots_adjust(right=0.8)
cbar_ax = fig.add_axes([0.85, 0.15, 0.02, 0.7])
cbar = fig.colorbar(plot_data, ticks=[10, 100, 200, 300, 400, 500], cax=cbar_ax)
cbar.ax.set_yticklabels(['0s', '100s', '200s', '300s', '400s', '500s'])
plt.savefig('Figures/Two_sample_Trajs.jpg', bbox_inches='tight', format='jpg')    
plt.show()

# %% HIL ablation study only Options 1 and some selected trajectories
Reward_eval_human = np.load("./Expert_data/Reward_eval_human.npy")
Real_Reward_eval_human = np.load("./Expert_data/Real_Reward_eval_human.npy", allow_pickle=True).tolist()
Processing_difference = Real_Reward_eval_human - Reward_eval_human 

coins_array = [9, 10, 13, 40, 43]

# selected by inspection considering: (i) human performance, (ii) at least a trajectory from each subject, (iii) not too many lost info in the preprocessing step

HIL_ablation_study_results = load_obj('results/HIL_ablation_study/Sorted_results')
clrs = sns.color_palette("husl", 10)


columns = 5
rows = 1
fig, ax_array = plt.subplots(rows, columns, squeeze=False, figsize=(20,5))
i=0
for k,ax_row in enumerate(ax_array):
    for j,axes in enumerate(ax_row):

        HIL_nOptions_1_supervised_False = []
        
        for j in range(8):
            HIL_nOptions_1_supervised_False.append(HIL_ablation_study_results[f'HIL_Expert_traj_{coins_array[i]}'][f'HIL_traj_{coins_array[i]}_nOptions_1_supervised_False_{j}'])
            
        HIL_nOptions_1_supervised_False_mean = np.mean(np.array(HIL_nOptions_1_supervised_False),0)
        HIL_nOptions_1_supervised_False_std = np.std(np.array(HIL_nOptions_1_supervised_False),0)
        
        BW_iters = np.linspace(0,10,len(HIL_nOptions_1_supervised_False_mean))
        threshold = np.mean(Real_Reward_eval_human)
        Human_average_performance = threshold*np.ones((len(BW_iters),))
        
        Expert_value = Reward_eval_human[coins_array[i]]*np.ones((len(BW_iters),))
        Original = Real_Reward_eval_human[coins_array[i]]*np.ones((len(BW_iters),))
        
        axes.plot(BW_iters, HIL_nOptions_1_supervised_False_mean, label='Imitation Learning Agent', c=clrs[0])
        axes.fill_between(BW_iters, HIL_nOptions_1_supervised_False_mean-HIL_nOptions_1_supervised_False_std, HIL_nOptions_1_supervised_False_mean+HIL_nOptions_1_supervised_False_std, alpha=0.2, facecolor=clrs[0])
        axes.plot(BW_iters, Original, "-.", label='Imitated Human', c=clrs[7])
        axes.plot(BW_iters, Human_average_performance, "--", label='Humans average', c=clrs[9])
        
        axes.set_ylim([0, 300])
        axes.set_xlabel('Epochs')
        if i == 0:
            axes.set_ylabel('Reward')
            
        axes.title.set_text(f'Traj {coins_array[i]+1}')
            
        i+=1
  
# box = ax_array.get_position()
# ax_array.set_position([box.x0, box.y0 + box.height * 0.1,
#                  box.width, box.height * 0.9])
# Put a legend below current axis
handles, labels = axes.get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.04), fancybox=True, shadow=True, ncol=8)
plt.savefig('Figures/HIL_over_trajs_selected_comparison_grid_flat.pdf', format='pdf', bbox_inches='tight')
plt.savefig('Figures/HIL_over_trajs_selected_comparison_grid_flat.jpg', format='jpg', bbox_inches='tight')
plt.show()

# %% HIL ablation study only Options 1
Reward_eval_human = np.load("./Expert_data/Reward_eval_human.npy")
Real_Reward_eval_human = np.load("./Expert_data/Real_Reward_eval_human.npy", allow_pickle=True).tolist()
Processing_difference = Real_Reward_eval_human - Reward_eval_human 

coins_array = range(50)

# selected by inspection considering: (i) human performance, (ii) at least a trajectory from each subject, (iii) not too many lost info in the preprocessing step

HIL_ablation_study_results = load_obj('results/HIL_ablation_study/Sorted_results')
clrs = sns.color_palette("husl", 10)

columns = 5
rows = 10
fig, ax_array = plt.subplots(rows, columns, squeeze=False, figsize=(25,45))
i=0
for k,ax_row in enumerate(ax_array):
    for j,axes in enumerate(ax_row):

        HIL_nOptions_1_supervised_False = []
        
        for j in range(8):
            HIL_nOptions_1_supervised_False.append(HIL_ablation_study_results[f'HIL_Expert_traj_{coins_array[i]}'][f'HIL_traj_{coins_array[i]}_nOptions_1_supervised_False_{j}'])
            
        HIL_nOptions_1_supervised_False_mean = np.mean(np.array(HIL_nOptions_1_supervised_False),0)
        HIL_nOptions_1_supervised_False_std = np.std(np.array(HIL_nOptions_1_supervised_False),0)
        
        BW_iters = np.linspace(0,10,len(HIL_nOptions_1_supervised_False_mean))
        threshold = np.mean(Real_Reward_eval_human)
        Human_average_performance = threshold*np.ones((len(BW_iters),))
        
        Expert_value = Reward_eval_human[coins_array[i]]*np.ones((len(BW_iters),))
        Original = Real_Reward_eval_human[coins_array[i]]*np.ones((len(BW_iters),))
        
        axes.plot(BW_iters, HIL_nOptions_1_supervised_False_mean, label='nOptions 1', c=clrs[0])
        axes.fill_between(BW_iters, HIL_nOptions_1_supervised_False_mean-HIL_nOptions_1_supervised_False_std, HIL_nOptions_1_supervised_False_mean+HIL_nOptions_1_supervised_False_std, alpha=0.2, facecolor=clrs[0])
        axes.plot(BW_iters, Original, "-.", label='Imitated Human', c=clrs[7])
        axes.plot(BW_iters, Human_average_performance, "--", label='Humans average', c=clrs[9])
        
        axes.set_ylim([0, 300])
        axes.title.set_text(f'Traj {coins_array[i]+1}')
        
        i+=1
  
# box = ax_array.get_position()
# ax_array.set_position([box.x0, box.y0 + box.height * 0.1,
#                  box.width, box.height * 0.9])
# Put a legend below current axis
handles, labels = axes.get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.12), fancybox=True, shadow=True, ncol=8)
plt.savefig('Figures/HIL_over_trajs_comparison_grid_flat.pdf', format='pdf', bbox_inches='tight')
plt.savefig('Figures/HIL_over_trajs_comparison_grid_flat.jpg', format='jpg', bbox_inches='tight')
plt.show()

# %%
HIL_ablation_study_results = load_obj('results/HIL_ablation_study/Sorted_results')
success = 0

for i in range(50):
    
    HIL_nOptions_1_supervised_False = []
    
    for j in range(8):
        HIL_nOptions_1_supervised_False.append(HIL_ablation_study_results[f'HIL_Expert_traj_{coins_array[i]}'][f'HIL_traj_{coins_array[i]}_nOptions_1_supervised_False_{j}'])
            
    HIL_nOptions_1_supervised_False_mean = np.mean(np.array(HIL_nOptions_1_supervised_False),0)
    HIL_nOptions_1_supervised_False_std = np.std(np.array(HIL_nOptions_1_supervised_False),0)
    
    if HIL_nOptions_1_supervised_False_mean[-1]/325 > 0.4:
        success+=1
        
print(success/50)



# %% analysis on which RL algorithm works better

PPO_IL = []
TRPO_IL = []
TRPO_no_entropy_IL = []
UATRPO_IL = []
UATRPO_no_entropy_IL = []
TD3_IL = []
SAC_IL = []

PPO_no_IL = []

for i in range(8):
    with open(f'results/HRL//evaluation_PPO_HIL_True_nOptions_1_supervised_False_{i}.npy', 'rb') as f:
        PPO_IL.append(np.load(f, allow_pickle=True))
        
    with open(f'results/HRL//evaluation_TRPO_HIL_True_nOptions_1_supervised_False_{i}.npy', 'rb') as f:
        TRPO_IL.append(np.load(f, allow_pickle=True))
        
    with open(f'results/HRL//evaluation_TRPO_entropy_false_HIL_True_nOptions_1_supervised_False_{i}.npy', 'rb') as f:
        TRPO_no_entropy_IL.append(np.load(f, allow_pickle=True))
        
    with open(f'results/HRL//evaluation_UATRPO_HIL_True_nOptions_1_supervised_False_{i}.npy', 'rb') as f:
        UATRPO_IL.append(np.load(f, allow_pickle=True))
        
    with open(f'results/HRL//evaluation_UATRPO_entropy_false_HIL_True_nOptions_1_supervised_False_{i}.npy', 'rb') as f:
        UATRPO_no_entropy_IL.append(np.load(f, allow_pickle=True))     
        
    with open(f'results/HRL//evaluation_TD3_HIL_True_traj_13_{i}.npy', 'rb') as f:
        TD3_IL.append(np.load(f, allow_pickle=True))
        
    with open(f'results/HRL//evaluation_SAC_HIL_True_traj_13_{i}.npy', 'rb') as f:
        SAC_IL.append(np.load(f, allow_pickle=True))

    with open(f'results/HRL//evaluation_PPO_HIL_False_nOptions_1_supervised_False_{i}.npy', 'rb') as f:
        PPO_no_IL.append(np.load(f, allow_pickle=True))
        

steps = np.linspace(0,10.02e6,len(PPO_IL[0]))

Real_Reward_eval_human = np.load("./Expert_data/Real_Reward_eval_human.npy", allow_pickle=True).tolist()    
threshold = np.mean(Real_Reward_eval_human)
Human_average_performance = threshold*np.ones((len(steps),))

PPO_HIL_mean = np.mean(np.array(PPO_IL),0)
PPO_HIL_std = np.std(np.array(PPO_IL),0)

TRPO_HIL_mean = np.mean(np.array(TRPO_IL),0)
TRPO_HIL_std = np.std(np.array(TRPO_IL),0)

TRPO_no_entropy_HIL_mean = np.mean(np.array(TRPO_no_entropy_IL),0)
TRPO_no_entropy_HIL_std = np.std(np.array(TRPO_no_entropy_IL),0)

UATRPO_HIL_mean = np.mean(np.array(UATRPO_IL),0)
UATRPO_HIL_std = np.std(np.array(UATRPO_IL),0)

UATRPO_no_entropy_HIL_mean = np.mean(np.array(UATRPO_no_entropy_IL),0)
UATRPO_no_entropy_HIL_std = np.std(np.array(UATRPO_no_entropy_IL),0)

TD3_HIL_mean = np.mean(np.array(TD3_IL),0)
TD3_HIL_std = np.std(np.array(TD3_IL),0)

SAC_HIL_mean = np.mean(np.array(SAC_IL),0)
SAC_HIL_std = np.std(np.array(SAC_IL),0)

clrs = sns.color_palette("husl", 10)

columns = 2
rows = 1
fig, ax_array = plt.subplots(rows, columns, squeeze=False, figsize=(12,4.5))
x=0
for k,ax_row in enumerate(ax_array):
    for j,axes in enumerate(ax_row):
               
        axes.plot(steps, PPO_HIL_mean, label='PPO', c=clrs[0])
        axes.fill_between(steps, PPO_HIL_mean-PPO_HIL_std, PPO_HIL_mean+PPO_HIL_std, alpha=0.2, facecolor=clrs[0])
        axes.plot(steps, TRPO_no_entropy_HIL_mean, label='TRPO', c=clrs[2])
        axes.fill_between(steps, TRPO_no_entropy_HIL_mean-TRPO_no_entropy_HIL_std, TRPO_no_entropy_HIL_mean+TRPO_no_entropy_HIL_std, alpha=0.2, facecolor=clrs[2])
        axes.plot(steps, UATRPO_no_entropy_HIL_mean, label='UATRPO', c=clrs[4])
        axes.fill_between(steps, UATRPO_no_entropy_HIL_mean-UATRPO_no_entropy_HIL_std, UATRPO_no_entropy_HIL_mean+UATRPO_no_entropy_HIL_std, alpha=0.2, facecolor=clrs[4])
        axes.plot(steps, TD3_HIL_mean, label='TD3', c=clrs[1])
        axes.fill_between(steps, TD3_HIL_mean-TD3_HIL_std, TD3_HIL_mean+TD3_HIL_std, alpha=0.2, facecolor=clrs[1])
        axes.plot(steps, SAC_HIL_mean, label='SAC', c=clrs[3])
        axes.fill_between(steps, SAC_HIL_mean-SAC_HIL_std, SAC_HIL_mean+SAC_HIL_std, alpha=0.2, facecolor=clrs[3])
        axes.plot(steps, Human_average_performance, "--", label='Humans average', c=clrs[9])
        axes.set_xlabel('Steps')
        
        if x==0:
            axes.set_ylim([0, 300])
            axes.set_ylabel('Reward')
        elif x==1:
            axes.set_ylim([0, 300])
            axes.set_xlim([0, 2e6])
            
        x+=1
            

handles, labels = axes.get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.02), fancybox=True, shadow=True, ncol=8)
plt.savefig('Figures/Comparison_RL_methods.pdf', format='pdf', bbox_inches='tight')
plt.savefig('Figures/Comparison_RL_methods.jpg', format='jpg', bbox_inches='tight')


# %% HRL PPO study preWork

PPO_no_IL = []

for i in range(8):
    with open(f'results/HRL/evaluation_PPO_HIL_False_nOptions_1_supervised_False_{i}.npy', 'rb') as f:
        PPO_no_IL.append(np.load(f, allow_pickle=True))
        
PPO_mean = np.mean(np.array(PPO_no_IL),0)
PPO_std = np.std(np.array(PPO_no_IL),0)

coins_array = range(50)

PPO_RL_AllHumans = [[[] for i in range(8)] for coin in coins_array]

j=0
success=0
for human in coins_array:
    for i in range(8):
        with open(f'results/HRL/evaluation_PPO_HIL_True_traj_{human}_nOptions_1_supervised_False_{i}.npy', 'rb') as f:
            PPO_RL_AllHumans[j][i] = np.load(f, allow_pickle=True)
            
            
    j+=1

PPO_RL_mean = []
PPO_RL_std = []
success=0
for j in range(len(coins_array)):
    PPO_RL_mean.append(np.mean(np.array(PPO_RL_AllHumans[j]),0))
    PPO_RL_std.append(np.std(np.array(PPO_RL_AllHumans[j]),0))
    
    if PPO_RL_mean[j][-1]/325>0.85:
        success+=1
        
print(success/50)

success = 0

for i in range(len(Real_Reward_eval_human)):
    
    if Real_Reward_eval_human[i]/325>0.85:
        success+=1
        
print(success/50)
    
# %% HRL PPO study only Options 1 and some selected trajectories
    
steps = np.linspace(0,10.02e6,len(PPO_RL_mean[0]))
Real_Reward_eval_human = np.load("./Expert_data/Real_Reward_eval_human.npy", allow_pickle=True).tolist()    
threshold = np.mean(Real_Reward_eval_human)
Human_average_performance = threshold*np.ones((len(steps),))

coins_array = [9, 10, 13, 40, 43]

clrs = sns.color_palette("husl", 10)

columns = 5
rows = 1
fig, ax_array = plt.subplots(rows, columns, squeeze=False, figsize=(20,5))
i=0
for k,ax_row in enumerate(ax_array):
    for j,axes in enumerate(ax_row):
        
        axes.plot(steps, PPO_RL_mean[coins_array[i]], label=f'PPO + IL', c=clrs[1])
        axes.fill_between(steps, PPO_RL_mean[coins_array[i]]-PPO_RL_std[coins_array[i]], PPO_RL_mean[coins_array[i]]+PPO_RL_std[coins_array[i]], alpha=0.1, facecolor=clrs[1])
        axes.plot(steps, PPO_mean, label='PPO', c=clrs[8])
        axes.fill_between(steps, PPO_mean-PPO_std, PPO_mean+PPO_std, alpha=0.2, facecolor=clrs[8])
        
        Original = Real_Reward_eval_human[coins_array[i]]*np.ones((len(steps),))

        axes.plot(steps, Original, "-.", label='Imitated Human', c=clrs[7])
        axes.plot(steps, Human_average_performance, "--", label='Humans average', c=clrs[9])
        
        axes.set_ylim([0, 300])
        axes.set_xlabel('Steps')
        if i == 0:
            axes.set_ylabel('Reward')
            
        axes.title.set_text(f'Traj {coins_array[i]+1}')
        
        i+=1

handles, labels = axes.get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.04), fancybox=True, shadow=True, ncol=8)
plt.savefig('Figures/Comparison_PPO_some_experts.pdf', format='pdf', bbox_inches='tight')
plt.savefig('Figures/Comparison_PPO_some_experts.jpg', format='jpg', bbox_inches='tight')

# %% HRL PPO study only Options 1 and all the trajectories

Reward_eval_human = np.load("./Expert_data/Reward_eval_human.npy")
Real_Reward_eval_human = np.load("./Expert_data/Real_Reward_eval_human.npy", allow_pickle=True).tolist()
Processing_difference = Real_Reward_eval_human - Reward_eval_human 

coins_array = range(50)

# selected by inspection considering: (i) human performance, (ii) at least a trajectory from each subject, (iii) not too many lost info in the preprocessing step

columns = 5
rows = 10
fig, ax_array = plt.subplots(rows, columns, squeeze=False, figsize=(25,45))
i=0
for k,ax_row in enumerate(ax_array):
    for j,axes in enumerate(ax_row):

        axes.plot(steps, PPO_RL_mean[coins_array[i]], label=f'PPO + IL', c=clrs[1])
        axes.fill_between(steps, PPO_RL_mean[coins_array[i]]-PPO_RL_std[coins_array[i]], PPO_RL_mean[coins_array[i]]+PPO_RL_std[coins_array[i]], alpha=0.1, facecolor=clrs[1])
        axes.plot(steps, PPO_mean, label='PPO', c=clrs[8])
        axes.fill_between(steps, PPO_mean-PPO_std, PPO_mean+PPO_std, alpha=0.2, facecolor=clrs[8])
        
        Original = Real_Reward_eval_human[coins_array[i]]*np.ones((len(steps),))

        axes.plot(steps, Original, "-.", label='Imitated Human', c=clrs[7])
        axes.plot(steps, Human_average_performance, "--", label='Humans average', c=clrs[9])
        
        axes.set_ylim([0, 300])
        
        axes.title.set_text(f'Traj {coins_array[i]+1}')
        
        i+=1

handles, labels = axes.get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.12), fancybox=True, shadow=True, ncol=8)
plt.savefig('Figures/Comparison_PPO_all_experts.pdf', format='pdf', bbox_inches='tight')
plt.savefig('Figures/Comparison_PPO_all_experts.jpg', format='jpg', bbox_inches='tight')
plt.show()

# %% New reward distribution

np.random.seed(0)
coin_cluster_1 = np.round(np.random.multivariate_normal([-70,+30], [[(5)**2, 0], [0, (5)**2]], size=50),0)
coin_cluster_2 = np.round(np.random.multivariate_normal([60,-20], [[(11)**2, 0], [0, (11)**2]], size=75),0)
coin_cluster_3 = np.round(np.random.multivariate_normal([-40,-45], [[(15)**2, 0], [0, (15)**2]], size=100),0)
coin_cluster_4 = np.round(np.random.multivariate_normal([0,60], [[(13)**2, 0], [0, (13)**2]], size=100),0)

coins_location_adv = np.concatenate((coin_cluster_1,coin_cluster_2,coin_cluster_3,coin_cluster_4),0)

wall_1 = [[-11,-10+i*0.1] for i in range(201)]
wall_2 = [[-10+i*0.1,11] for i in range(201)]
wall_3 = [[11,10-i*0.1] for i in range(201)]
wall_4 = [[-10+i*0.1,-11] for i in range(201)]


sigma1 = 0.5
circle1 = ptch.Circle((-7, 3), 2*sigma1, color='k',  fill=False)
sigma2 = 1.1
circle2 = ptch.Circle((6, -2), 2*sigma2, color='k',  fill=False)
sigma3 = 1.5
circle3 = ptch.Circle((-4, -4.5), 2*sigma3, color='k',  fill=False)
sigma4 = 1.3
circle4 = ptch.Circle((0, 6), 2*sigma4, color='k',  fill=False)
fig, ax = plt.subplots(squeeze = True, figsize=(5,5))
ax.add_artist(circle1)
ax.add_artist(circle2)
ax.add_artist(circle3)
ax.add_artist(circle4)

for i in range(len(wall_1)):
    ax.plot([wall_1[i][0], wall_1[i][0]+1, wall_1[i][0]+1, wall_1[i][0], wall_1[i][0]],
             [wall_1[i][1], wall_1[i][1], wall_1[i][1]+0.1, wall_1[i][1]+0.1, wall_1[i][1]],'y-')

for i in range(len(wall_1)):
    ax.fill([wall_1[i][0], wall_1[i][0]+1, wall_1[i][0]+1, wall_1[i][0], wall_1[i][0]],
             [wall_1[i][1], wall_1[i][1], wall_1[i][1]+0.1, wall_1[i][1]+0.1, wall_1[i][1]],'k')
    
for i in range(len(wall_2)):
    ax.plot([wall_2[i][0], wall_2[i][0]+0.1, wall_2[i][0]+0.1, wall_2[i][0], wall_2[i][0]],
             [wall_2[i][1], wall_2[i][1], wall_2[i][1]-1, wall_2[i][1]-1, wall_2[i][1]],'r-')

for i in range(len(wall_2)):
    ax.fill([wall_2[i][0], wall_2[i][0]+0.1, wall_2[i][0]+0.1, wall_2[i][0], wall_2[i][0]],
             [wall_2[i][1], wall_2[i][1], wall_2[i][1]-1, wall_2[i][1]-1, wall_2[i][1]],'k')
    
for i in range(len(wall_3)):
    ax.plot([wall_3[i][0], wall_3[i][0]-1, wall_3[i][0]-1, wall_3[i][0], wall_3[i][0]],
             [wall_3[i][1], wall_3[i][1], wall_3[i][1]-0.1, wall_3[i][1]-0.1, wall_3[i][1]],'-', c= 'tab:gray')

for i in range(len(wall_3)):
    ax.fill([wall_3[i][0], wall_3[i][0]-1, wall_3[i][0]-1, wall_3[i][0], wall_3[i][0]],
             [wall_3[i][1], wall_3[i][1], wall_3[i][1]-0.1, wall_3[i][1]-0.1, wall_3[i][1]],'k')
    
for i in range(len(wall_4)):
    ax.plot([wall_4[i][0], wall_4[i][0]+0.1, wall_4[i][0]+0.1, wall_4[i][0], wall_4[i][0]],
             [wall_4[i][1], wall_4[i][1], wall_4[i][1]+1, wall_4[i][1]+1, wall_4[i][1]],'-', c = 'tab:brown')

for i in range(len(wall_4)):
    ax.fill([wall_4[i][0], wall_4[i][0]+0.1, wall_4[i][0]+0.1, wall_4[i][0], wall_4[i][0]],
             [wall_4[i][1], wall_4[i][1], wall_4[i][1]+1, wall_4[i][1]+1, wall_4[i][1]],'k')
    

plt.plot(0.1*coins_location_adv[:,0], 0.1*coins_location_adv[:,1], 'xb')
plt.axis('off')
plt.savefig('Figures/Environment_top_view_ADV.pdf', bbox_inches='tight', format='pdf')  
plt.show()  

# %% HIL only PPO Adversarial-Reward

coins_array = range(50)
max_number_coins = 325

PPO_IL_only_Adversarial = [[[] for i in range(8)] for coin in coins_array]

j=0
for human in coins_array:
    for i in range(8):
        with open(f'results/HRL/evaluation_PPO_HIL_True_ONLY_HIL_model_traj_{human}_ADV_Reward_True_{i}.npy', 'rb') as f:
            PPO_IL_only_Adversarial[j][i] = np.load(f, allow_pickle=True)/max_number_coins
            
    j+=1

PPO_IL_only_Adversarial_mean = []
PPO_IL_only_Adversarial_std = []
success=0
for j in range(len(coins_array)):
    PPO_IL_only_Adversarial_mean.append(np.mean(np.array(PPO_IL_only_Adversarial[j]),0))
    PPO_IL_only_Adversarial_std.append(np.std(np.array(PPO_IL_only_Adversarial[j]),0))
    
    if PPO_IL_only_Adversarial_mean[j][-1]>0.9:
        success+=1
        
print((success/50)*100)
    
# %% HRL PPO study only Options 1 and some selected trajectories
    
steps = np.linspace(0,2.01e6,len(PPO_IL_only_Adversarial_mean[0]))
Real_Reward_eval_human = np.load("./Expert_data/Real_Reward_eval_human.npy", allow_pickle=True).tolist()    
threshold = np.mean(Real_Reward_eval_human)
Human_average_performance = threshold*np.ones((len(steps),))

coins_array = [9, 10, 13, 40, 43]

clrs = sns.color_palette("husl", 10)

columns = 5
rows = 1
fig, ax_array = plt.subplots(rows, columns, squeeze=False, figsize=(20,5))
i=0
for k,ax_row in enumerate(ax_array):
    for j,axes in enumerate(ax_row):
        
        axes.plot(steps, PPO_IL_only_Adversarial_mean[coins_array[i]], label=f'PPO + IL-only', c=clrs[1])
        axes.fill_between(steps, PPO_IL_only_Adversarial_mean[coins_array[i]]-PPO_IL_only_Adversarial_std[coins_array[i]], PPO_IL_only_Adversarial_mean[coins_array[i]]+PPO_IL_only_Adversarial_std[coins_array[i]], alpha=0.1, facecolor=clrs[1])
        
        Original = np.ones((len(steps),))

        axes.plot(steps, Original, "-.", label='MAX Reward', c=clrs[7])
        
        axes.set_ylim([0, 1.1])
        axes.set_xlabel('Steps')
        if i == 0:
            axes.set_ylabel('Reward')
            
        axes.title.set_text(f'Traj {coins_array[i]+1}')
        
        i+=1

handles, labels = axes.get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.04), fancybox=True, shadow=True, ncol=8)
plt.savefig('Figures/Comparison_PPO_IL_ONLY_ADV_Reward_some_experts.pdf', format='pdf', bbox_inches='tight')
plt.savefig('Figures/Comparison_PPO_IL_ONLY_ADV_Reward_some_experts.jpg', format='jpg', bbox_inches='tight')

# %% HRL PPO study only Options 1 and all the trajectories

Reward_eval_human = np.load("./Expert_data/Reward_eval_human.npy")
Real_Reward_eval_human = np.load("./Expert_data/Real_Reward_eval_human.npy", allow_pickle=True).tolist()
Processing_difference = Real_Reward_eval_human - Reward_eval_human 

coins_array = range(50)

# selected by inspection considering: (i) human performance, (ii) at least a trajectory from each subject, (iii) not too many lost info in the preprocessing step

columns = 5
rows = 10
fig, ax_array = plt.subplots(rows, columns, squeeze=False, figsize=(25,45))
i=0
for k,ax_row in enumerate(ax_array):
    for j,axes in enumerate(ax_row):

        axes.plot(steps, PPO_IL_only_Adversarial_mean[coins_array[i]], label='PPO + IL-only', c=clrs[1])
        axes.fill_between(steps, PPO_IL_only_Adversarial_mean[coins_array[i]]-PPO_IL_only_Adversarial_std[coins_array[i]], PPO_IL_only_Adversarial_mean[coins_array[i]]+PPO_IL_only_Adversarial_std[coins_array[i]], alpha=0.1, facecolor=clrs[1])
        
        Original = np.ones((len(steps),))

        axes.plot(steps, Original, "-.", label='MAX Reward', c=clrs[7])
        
        axes.set_ylim([0, 1.1])
        
        axes.title.set_text(f'Traj {coins_array[i]+1}')
        
        i+=1

handles, labels = axes.get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.12), fancybox=True, shadow=True, ncol=8)
plt.savefig('Figures/Comparison_PPO_IL_ONLY_ADV_Reward_all_experts.pdf', format='pdf', bbox_inches='tight')
plt.savefig('Figures/Comparison_PPO_IL_ONLY_ADV_Reward_all_experts.jpg', format='jpg', bbox_inches='tight')
plt.show()

# %% HIL+HRL PPO Adversarial-Reward

coins_array = range(50)
max_number_coins = 325

PPO_IL_RL_Adversarial = [[[] for i in range(8)] for coin in coins_array]

j=0
for human in coins_array:
    for i in range(8):
        with open(f'results/HRL/evaluation_PPO_HIL_False_BOTH_HIL_HRL_traj_{human}_ADV_Reward_True_{i}.npy', 'rb') as f:
            PPO_IL_RL_Adversarial[j][i] = np.load(f, allow_pickle=True)/max_number_coins
            
    j+=1

PPO_IL_RL_Adversarial_mean = []
PPO_IL_RL_Adversarial_std = []
success=0
for j in range(len(coins_array)):
    PPO_IL_RL_Adversarial_mean.append(np.mean(np.array(PPO_IL_RL_Adversarial[j]),0))
    PPO_IL_RL_Adversarial_std.append(np.std(np.array(PPO_IL_RL_Adversarial[j]),0))
    
    if PPO_IL_RL_Adversarial_mean[j][-1]>0.8:
        success+=1
        
print((success/50)*100)

# %% HRL PPO study only Options 1 and some selected trajectories
    
steps = np.linspace(0,2.01e6,len(PPO_IL_RL_Adversarial_mean[0]))
Real_Reward_eval_human = np.load("./Expert_data/Real_Reward_eval_human.npy", allow_pickle=True).tolist()    
threshold = np.mean(Real_Reward_eval_human)
Human_average_performance = threshold*np.ones((len(steps),))

coins_array = [9, 10, 13, 40, 43]

clrs = sns.color_palette("husl", 10)

columns = 5
rows = 1
fig, ax_array = plt.subplots(rows, columns, squeeze=False, figsize=(20,5))
i=0
for k,ax_row in enumerate(ax_array):
    for j,axes in enumerate(ax_row):
        
        axes.plot(steps, PPO_IL_RL_Adversarial_mean[coins_array[i]], label='PPO + RL-IL', c=clrs[1])
        axes.fill_between(steps, PPO_IL_RL_Adversarial_mean[coins_array[i]]-PPO_IL_RL_Adversarial_std[coins_array[i]], PPO_IL_RL_Adversarial_mean[coins_array[i]]+PPO_IL_RL_Adversarial_std[coins_array[i]], alpha=0.1, facecolor=clrs[1])
        
        Original = np.ones((len(steps),))

        axes.plot(steps, Original, "-.", label='MAX Reward', c=clrs[7])
        
        axes.set_ylim([0, 1.1])
        axes.set_xlabel('Steps')
        if i == 0:
            axes.set_ylabel('Reward')
            
        axes.title.set_text(f'Traj {coins_array[i]+1}')
        
        i+=1

handles, labels = axes.get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.04), fancybox=True, shadow=True, ncol=8)
plt.savefig('Figures/Comparison_PPO_RL_IL_ADV_Reward_some_experts.pdf', format='pdf', bbox_inches='tight')
plt.savefig('Figures/Comparison_PPO_RL_IL_ADV_Reward_some_experts.jpg', format='jpg', bbox_inches='tight')

# %% HRL PPO study only Options 1 and all the trajectories

Reward_eval_human = np.load("./Expert_data/Reward_eval_human.npy")
Real_Reward_eval_human = np.load("./Expert_data/Real_Reward_eval_human.npy", allow_pickle=True).tolist()
Processing_difference = Real_Reward_eval_human - Reward_eval_human 

coins_array = range(50)

# selected by inspection considering: (i) human performance, (ii) at least a trajectory from each subject, (iii) not too many lost info in the preprocessing step

columns = 5
rows = 10
fig, ax_array = plt.subplots(rows, columns, squeeze=False, figsize=(25,45))
i=0
for k,ax_row in enumerate(ax_array):
    for j,axes in enumerate(ax_row):

        axes.plot(steps, PPO_IL_RL_Adversarial_mean[coins_array[i]], label='PPO + RL-IL', c=clrs[1])
        axes.fill_between(steps, PPO_IL_RL_Adversarial_mean[coins_array[i]]-PPO_IL_RL_Adversarial_std[coins_array[i]], PPO_IL_RL_Adversarial_mean[coins_array[i]]+PPO_IL_RL_Adversarial_std[coins_array[i]], alpha=0.1, facecolor=clrs[1])
        
        Original = np.ones((len(steps),))

        axes.plot(steps, Original, "-.", label='MAX Reward', c=clrs[7])
        
        axes.set_ylim([0, 1.1])
        
        axes.title.set_text(f'Traj {coins_array[i]+1}')
        
        i+=1

handles, labels = axes.get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.12), fancybox=True, shadow=True, ncol=8)
plt.savefig('Figures/Comparison_PPO_RL_IL_ADV_Reward_all_experts.pdf', format='pdf', bbox_inches='tight')
plt.savefig('Figures/Comparison_PPO_RL_IL_ADV_Reward_all_experts.jpg', format='jpg', bbox_inches='tight')
plt.show()

# %% Humans

success = 0

for i in range(len(Real_Reward_eval_human)):
    
    if Real_Reward_eval_human[i]/325>0.9:
        success+=1
        
print(success/50)

# %% HIL ablation study only Options 1 and some selected trajectories ALLOCENTRIC ONLY
Reward_eval_human = np.load("./Expert_data/Reward_eval_human.npy")
Real_Reward_eval_human = np.load("./Expert_data/Real_Reward_eval_human.npy", allow_pickle=True).tolist()
Processing_difference = Real_Reward_eval_human - Reward_eval_human 

coins_array = [9, 10, 13, 40, 43]

# selected by inspection considering: (i) human performance, (ii) at least a trajectory from each subject, (iii) not too many lost info in the preprocessing step

coins_array = range(50)

HIL_allocentric_AllHumans = [[[] for i in range(8)] for coin in coins_array]

for human in coins_array:
    for i in range(8):
        with open(f'results/HRL/HIL_allocentric_only_traj_{human}_nOptions_1_supervised_False_{i}.npy', 'rb') as f:
            HIL_allocentric_AllHumans[human][i] = np.load(f, allow_pickle=True)
            
clrs = sns.color_palette("husl", 10)

HIL_ablation_study_results = load_obj('results/HIL_ablation_study/Sorted_results')

#%%

coins_array = [9, 10, 13, 40, 43]
columns = 5
rows = 1
fig, ax_array = plt.subplots(rows, columns, squeeze=False, figsize=(20,5))
i=0
for k,ax_row in enumerate(ax_array):
    for j,axes in enumerate(ax_row):

        HIL_nOptions_1_supervised_False = []
        
        for j in range(8):
            HIL_nOptions_1_supervised_False.append(HIL_ablation_study_results[f'HIL_Expert_traj_{coins_array[i]}'][f'HIL_traj_{coins_array[i]}_nOptions_1_supervised_False_{j}'])
            
        HIL_nOptions_1_supervised_False_mean = np.mean(np.array(HIL_nOptions_1_supervised_False),0)
        HIL_nOptions_1_supervised_False_std = np.std(np.array(HIL_nOptions_1_supervised_False),0)
        
        HIL_allocentric_nOptions_1_supervised_False_mean = np.mean(np.array(HIL_allocentric_AllHumans[coins_array[i]]),0)
        HIL_allocentric_nOptions_1_supervised_False_std = np.std(np.array(HIL_allocentric_AllHumans[coins_array[i]]),0)
        
        BW_iters = np.linspace(0,10,len(HIL_nOptions_1_supervised_False_mean))
        threshold = np.mean(Real_Reward_eval_human)
        Human_average_performance = threshold*np.ones((len(BW_iters),))
        
        Expert_value = Reward_eval_human[coins_array[i]]*np.ones((len(BW_iters),))
        Original = Real_Reward_eval_human[coins_array[i]]*np.ones((len(BW_iters),))
        
        axes.plot(BW_iters, HIL_nOptions_1_supervised_False_mean, label='IL allocentric and egocentric Agent', c=clrs[0])
        axes.fill_between(BW_iters, HIL_nOptions_1_supervised_False_mean-HIL_nOptions_1_supervised_False_std, HIL_nOptions_1_supervised_False_mean+HIL_nOptions_1_supervised_False_std, alpha=0.2, facecolor=clrs[0])
        axes.plot(BW_iters, HIL_allocentric_nOptions_1_supervised_False_mean, label='IL allocentric Agent', c=clrs[2])
        axes.fill_between(BW_iters, HIL_allocentric_nOptions_1_supervised_False_mean-HIL_allocentric_nOptions_1_supervised_False_std, HIL_allocentric_nOptions_1_supervised_False_mean+HIL_allocentric_nOptions_1_supervised_False_std, alpha=0.2, facecolor=clrs[2])
        axes.plot(BW_iters, Original, "-.", label='Imitated Human', c=clrs[7])
        axes.plot(BW_iters, Human_average_performance, "--", label='Humans average', c=clrs[9])
        
        axes.set_ylim([0, 300])
        axes.set_xlabel('Epochs')
        if i == 0:
            axes.set_ylabel('Reward')
            
        axes.title.set_text(f'Traj {coins_array[i]+1}')
            
        i+=1
  
# box = ax_array.get_position()
# ax_array.set_position([box.x0, box.y0 + box.height * 0.1,
#                  box.width, box.height * 0.9])
# Put a legend below current axis
handles, labels = axes.get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.04), fancybox=True, shadow=True, ncol=8)
plt.savefig('Figures/HIL_over_trajs_selected_comparison_grid_flat_ALLOCENTRIC.pdf', format='pdf', bbox_inches='tight')
plt.savefig('Figures/HIL_over_trajs_selected_comparison_grid_flat_ALLOCENTRIC.jpg', format='jpg', bbox_inches='tight')
plt.show()

# %% HIL ablation study only Options 1
Reward_eval_human = np.load("./Expert_data/Reward_eval_human.npy")
Real_Reward_eval_human = np.load("./Expert_data/Real_Reward_eval_human.npy", allow_pickle=True).tolist()
Processing_difference = Real_Reward_eval_human - Reward_eval_human 

coins_array = range(50)

# selected by inspection considering: (i) human performance, (ii) at least a trajectory from each subject, (iii) not too many lost info in the preprocessing step

columns = 5
rows = 10
fig, ax_array = plt.subplots(rows, columns, squeeze=False, figsize=(25,45))
i=0
for k,ax_row in enumerate(ax_array):
    for j,axes in enumerate(ax_row):

        HIL_nOptions_1_supervised_False = []
        
        for j in range(8):
            HIL_nOptions_1_supervised_False.append(HIL_ablation_study_results[f'HIL_Expert_traj_{coins_array[i]}'][f'HIL_traj_{coins_array[i]}_nOptions_1_supervised_False_{j}'])
            
        HIL_nOptions_1_supervised_False_mean = np.mean(np.array(HIL_nOptions_1_supervised_False),0)
        HIL_nOptions_1_supervised_False_std = np.std(np.array(HIL_nOptions_1_supervised_False),0)
        
        HIL_allocentric_nOptions_1_supervised_False_mean = np.mean(np.array(HIL_allocentric_AllHumans[coins_array[i]]),0)
        HIL_allocentric_nOptions_1_supervised_False_std = np.std(np.array(HIL_allocentric_AllHumans[coins_array[i]]),0)
        
        BW_iters = np.linspace(0,10,len(HIL_nOptions_1_supervised_False_mean))
        threshold = np.mean(Real_Reward_eval_human)
        Human_average_performance = threshold*np.ones((len(BW_iters),))
        
        Expert_value = Reward_eval_human[coins_array[i]]*np.ones((len(BW_iters),))
        Original = Real_Reward_eval_human[coins_array[i]]*np.ones((len(BW_iters),))
        
        axes.plot(BW_iters, HIL_nOptions_1_supervised_False_mean, label='IL allocentric and egocentric Agent', c=clrs[0])
        axes.fill_between(BW_iters, HIL_nOptions_1_supervised_False_mean-HIL_nOptions_1_supervised_False_std, HIL_nOptions_1_supervised_False_mean+HIL_nOptions_1_supervised_False_std, alpha=0.2, facecolor=clrs[0])
        axes.plot(BW_iters, HIL_allocentric_nOptions_1_supervised_False_mean, label='IL allocentric Agent', c=clrs[2])
        axes.fill_between(BW_iters, HIL_allocentric_nOptions_1_supervised_False_mean-HIL_allocentric_nOptions_1_supervised_False_std, HIL_allocentric_nOptions_1_supervised_False_mean+HIL_allocentric_nOptions_1_supervised_False_std, alpha=0.2, facecolor=clrs[2])
        axes.plot(BW_iters, Original, "-.", label='Imitated Human', c=clrs[7])
        axes.plot(BW_iters, Human_average_performance, "--", label='Humans average', c=clrs[9])
        
        axes.set_ylim([0, 300])
        # axes.set_xlabel('Epochs')
        if i == 0:
            axes.set_ylabel('Reward')
            
        axes.title.set_text(f'Traj {coins_array[i]+1}')
            
        i+=1
  
# box = ax_array.get_position()
# ax_array.set_position([box.x0, box.y0 + box.height * 0.1,
#                  box.width, box.height * 0.9])
# Put a legend below current axis
handles, labels = axes.get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.12), fancybox=True, shadow=True, ncol=8)
plt.savefig('Figures/HIL_over_trajs_comparison_grid_flat_ALLOCENTRIC.pdf', format='pdf', bbox_inches='tight')
plt.savefig('Figures/HIL_over_trajs_comparison_grid_flat_ALLOCENTRIC.jpg', format='jpg', bbox_inches='tight')
plt.show()

# %%
HIL_nOptions_1_supervised_False = []
success=0
equal=0
for k in range(len(coins_array)):
    for j in range(8):
            HIL_nOptions_1_supervised_False.append(HIL_ablation_study_results[f'HIL_Expert_traj_{k}'][f'HIL_traj_{k}_nOptions_1_supervised_False_{j}'])
            
    HIL_nOptions_1_supervised_False_mean = np.mean(np.array(HIL_nOptions_1_supervised_False),0)
    HIL_allocentric_nOptions_1_supervised_False_mean = np.mean(np.array(HIL_allocentric_AllHumans[k]),0)

    if HIL_nOptions_1_supervised_False_mean[-1] >= HIL_allocentric_nOptions_1_supervised_False_mean[-1]:
        success+=1
        
print(f"Success Percentage {success/len(coins_array)}")
 
# %% HRL PPO study preWork ALLOCENTRIC ONLY

PPO_no_IL = []

for i in range(8):
    with open(f'results/HRL/evaluation_PPO_HIL_False_nOptions_1_supervised_False_{i}.npy', 'rb') as f:
        PPO_no_IL.append(np.load(f, allow_pickle=True))
        
PPO_mean = np.mean(np.array(PPO_no_IL),0)
PPO_std = np.std(np.array(PPO_no_IL),0)

coins_array = range(50)

PPO_RL_AllHumans = [[[] for i in range(8)] for coin in coins_array]

j=0
success=0
for human in coins_array:
    for i in range(8):
        with open(f'results/HRL/evaluation_PPO_HIL_True_traj_{human}_nOptions_1_supervised_False_{i}.npy', 'rb') as f:
            PPO_RL_AllHumans[j][i] = np.load(f, allow_pickle=True)
            
            
    j+=1
    
PPO_RL_allocentric_AllHumans = [[[] for i in range(8)] for coin in coins_array]

j=0
success=0
for human in coins_array:
    for i in range(8):
        with open(f'results/HRL/evaluation_PPO_allocentric_only_HIL_True_traj_{human}_{i}.npy', 'rb') as f:
            PPO_RL_allocentric_AllHumans[j][i] = np.load(f, allow_pickle=True)
            
            
    j+=1

PPO_RL_mean = []
PPO_RL_std = []
PPO_RL_allocentric_mean = []
PPO_RL_allocentric_std = []
success=0
success_allocentric=0
percentage = 0.7
for j in range(len(coins_array)):
    PPO_RL_mean.append(np.mean(np.array(PPO_RL_AllHumans[j]),0))
    PPO_RL_std.append(np.std(np.array(PPO_RL_AllHumans[j]),0))
    PPO_RL_allocentric_mean.append(np.mean(np.array(PPO_RL_allocentric_AllHumans[j]),0))
    PPO_RL_allocentric_std.append(np.std(np.array(PPO_RL_allocentric_AllHumans[j]),0))
    
    if PPO_RL_mean[j][-1]/325>percentage:
        success+=1
        
    if PPO_RL_allocentric_mean[j][-1]/325>percentage:
        success_allocentric+=1
        
print(success/50)
print(success_allocentric/50)

success = 0

for i in range(len(Real_Reward_eval_human)):
    
    if Real_Reward_eval_human[i]/325>percentage:
        success+=1
        
print(success/50)
    
# %% HRL PPO study only Options 1 and some selected trajectories
    
steps = np.linspace(0,10.02e6,len(PPO_RL_mean[0]))
Real_Reward_eval_human = np.load("./Expert_data/Real_Reward_eval_human.npy", allow_pickle=True).tolist()    
threshold = np.mean(Real_Reward_eval_human)
Human_average_performance = threshold*np.ones((len(steps),))

coins_array = [9, 10, 13, 40, 43]

clrs = sns.color_palette("husl", 10)

columns = 5
rows = 1
fig, ax_array = plt.subplots(rows, columns, squeeze=False, figsize=(20,5))
i=0
for k,ax_row in enumerate(ax_array):
    for j,axes in enumerate(ax_row):
        
        axes.plot(steps, PPO_RL_mean[coins_array[i]], label=f'PPO + IL', c=clrs[1])
        axes.fill_between(steps, PPO_RL_mean[coins_array[i]]-PPO_RL_std[coins_array[i]], PPO_RL_mean[coins_array[i]]+PPO_RL_std[coins_array[i]], alpha=0.1, facecolor=clrs[1])
        axes.plot(steps, PPO_RL_allocentric_mean[coins_array[i]], label=f'PPO + IL allocentric only', c=clrs[3])
        axes.fill_between(steps, PPO_RL_allocentric_mean[coins_array[i]]-PPO_RL_std[coins_array[i]], PPO_RL_allocentric_mean[coins_array[i]]+PPO_RL_std[coins_array[i]], alpha=0.1, facecolor=clrs[3])
        axes.plot(steps, PPO_mean, label='PPO', c=clrs[8])
        axes.fill_between(steps, PPO_mean-PPO_std, PPO_mean+PPO_std, alpha=0.2, facecolor=clrs[8])
        
        Original = Real_Reward_eval_human[coins_array[i]]*np.ones((len(steps),))

        axes.plot(steps, Original, "-.", label='Imitated Human', c=clrs[7])
        axes.plot(steps, Human_average_performance, "--", label='Humans average', c=clrs[9])
        
        axes.set_ylim([0, 300])
        axes.set_xlabel('Steps')
        if i == 0:
            axes.set_ylabel('Reward')
            
        axes.title.set_text(f'Traj {coins_array[i]+1}')
        
        i+=1

handles, labels = axes.get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.04), fancybox=True, shadow=True, ncol=8)
plt.savefig('Figures/Comparison_PPO_some_experts_ALLOCENTRIC.pdf', format='pdf', bbox_inches='tight')
plt.savefig('Figures/Comparison_PPO_some_experts_ALLOCENTRIC.jpg', format='jpg', bbox_inches='tight')

# %% HRL PPO study only Options 1 and all the trajectories

Reward_eval_human = np.load("./Expert_data/Reward_eval_human.npy")
Real_Reward_eval_human = np.load("./Expert_data/Real_Reward_eval_human.npy", allow_pickle=True).tolist()
Processing_difference = Real_Reward_eval_human - Reward_eval_human 

coins_array = range(50)

# selected by inspection considering: (i) human performance, (ii) at least a trajectory from each subject, (iii) not too many lost info in the preprocessing step

columns = 5
rows = 10
fig, ax_array = plt.subplots(rows, columns, squeeze=False, figsize=(25,45))
i=0
for k,ax_row in enumerate(ax_array):
    for j,axes in enumerate(ax_row):

        axes.plot(steps, PPO_RL_mean[coins_array[i]], label=f'PPO + IL', c=clrs[1])
        axes.fill_between(steps, PPO_RL_mean[coins_array[i]]-PPO_RL_std[coins_array[i]], PPO_RL_mean[coins_array[i]]+PPO_RL_std[coins_array[i]], alpha=0.1, facecolor=clrs[1])
        axes.plot(steps, PPO_RL_allocentric_mean[coins_array[i]], label=f'PPO + IL allocentric only', c=clrs[3])
        axes.fill_between(steps, PPO_RL_allocentric_mean[coins_array[i]]-PPO_RL_std[coins_array[i]], PPO_RL_allocentric_mean[coins_array[i]]+PPO_RL_std[coins_array[i]], alpha=0.1, facecolor=clrs[3])
        axes.plot(steps, PPO_mean, label='PPO', c=clrs[8])
        axes.fill_between(steps, PPO_mean-PPO_std, PPO_mean+PPO_std, alpha=0.2, facecolor=clrs[8])
        
        Original = Real_Reward_eval_human[coins_array[i]]*np.ones((len(steps),))

        axes.plot(steps, Original, "-.", label='Imitated Human', c=clrs[7])
        axes.plot(steps, Human_average_performance, "--", label='Humans average', c=clrs[9])
        
        axes.set_ylim([0, 300])
        
        axes.title.set_text(f'Traj {coins_array[i]+1}')
        
        i+=1

handles, labels = axes.get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.12), fancybox=True, shadow=True, ncol=8)
plt.savefig('Figures/Comparison_PPO_all_experts_ALLOCENTRIC.pdf', format='pdf', bbox_inches='tight')
plt.savefig('Figures/Comparison_PPO_all_experts_ALLOCENTRIC.jpg', format='jpg', bbox_inches='tight')
plt.show()