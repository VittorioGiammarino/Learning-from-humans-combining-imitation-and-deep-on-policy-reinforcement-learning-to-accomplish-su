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
        # axes.title.set_text('Human Agent {}, Trial N {}, Reward {}, ({})'.format(int(Rand_traj/10) + 1, Rand_traj%10 + 1, Real_Reward_eval_human[Rand_traj], coins_array[i]))
        axes.axis('off') 
        
        x+=1
  
plt.savefig('Figures/HIL_ablation_real_Trajs.jpg', bbox_inches='tight', format='jpg')    
plt.show()

# %% Plot all humans trajectories Top View

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
        axes.axis('off') 
        
        x+=1

fig.subplots_adjust(right=0.8)
cbar_ax = fig.add_axes([0.85, 0.15, 0.02, 0.7])
cbar = fig.colorbar(plot_data, ticks=[10, 100, 200, 300, 400, 500], cax=cbar_ax)
cbar.ax.set_yticklabels(['0s', '100s', '200s', '300s', '400s', '500s'])
plt.savefig('Figures/Two_sample_Trajs.jpg', bbox_inches='tight', format='jpg')    
plt.show()

# %% HIL ablation study only Options 1
Reward_eval_human = np.load("./Expert_data/Reward_eval_human.npy")
Real_Reward_eval_human = np.load("./Expert_data/Real_Reward_eval_human.npy", allow_pickle=True).tolist()
Processing_difference = Real_Reward_eval_human - Reward_eval_human 

coins_array = range(50)

# selected by inspection considering: (i) human performance, (ii) at least a trajectory from each subject, (iii) not too many lost info in the preprocessing step

HIL_ablation_study_results = load_obj('results/HIL_ablation_study/Sorted_results')
clrs = sns.color_palette("husl", 10)

HIL_nOptions_1_supervised_False_total = []

human = [[[] for s in range(5)] for i in range(5)]


columns = 10
rows = 5
fig, ax_array = plt.subplots(rows, columns, squeeze=False, figsize=(45,14))
i=0
for k,ax_row in enumerate(ax_array):
    for j,axes in enumerate(ax_row):

        HIL_nOptions_1_supervised_False = []
        
        for j in range(8):
            HIL_nOptions_1_supervised_False.append(HIL_ablation_study_results[f'HIL_Expert_traj_{coins_array[i]}'][f'HIL_traj_{coins_array[i]}_nOptions_1_supervised_False_{j}'])
            
        HIL_nOptions_1_supervised_False_mean = np.mean(np.array(HIL_nOptions_1_supervised_False),0)

        
        human[k][0].append(HIL_nOptions_1_supervised_False_mean)
        
        BW_iters = np.linspace(0,10,len(HIL_nOptions_1_supervised_False_mean))
        threshold = np.mean(Real_Reward_eval_human)
        Human_average_performance = threshold*np.ones((len(BW_iters),))
        
        Expert_value = Reward_eval_human[coins_array[i]]*np.ones((len(BW_iters),))
        Original = Real_Reward_eval_human[coins_array[i]]*np.ones((len(BW_iters),))
        i+=1
        
        axes.plot(BW_iters, HIL_nOptions_1_supervised_False_mean, label='nOptions 1', c=clrs[0])
        axes.plot(BW_iters, Original, "--", label='Original Human', c=clrs[7])
        axes.plot(BW_iters, Human_average_performance, "--", label='Humans average', c=clrs[9])
  
# box = ax_array.get_position()
# ax_array.set_position([box.x0, box.y0 + box.height * 0.1,
#                  box.width, box.height * 0.9])
# Put a legend below current axis
handles, labels = axes.get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.1), fancybox=True, shadow=True, ncol=8)
plt.savefig('HIL_over_trajs_comparison_grid_flat.pdf', format='pdf', bbox_inches='tight')
plt.savefig('HIL_over_trajs_comparison_grid_flat.jpg', format='jpg', bbox_inches='tight')
plt.show()
