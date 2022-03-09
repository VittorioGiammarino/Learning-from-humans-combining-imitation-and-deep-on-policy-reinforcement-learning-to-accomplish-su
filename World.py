#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 24 17:13:05 2020

@author: vittorio
"""
import numpy as np
from sklearn.preprocessing import OneHotEncoder
import csv

Coins_location = np.load("./Expert_data/Coins_location.npy")
Rand_traj = 2
coins_location_standard = Coins_location[Rand_traj,:,:] 

class Foraging_Training:
    def TransitionCheck4Labels_simplified(state,state_next):
        Transition = np.zeros((4,2))
        Transition[0,0] = state[0] + 1
        Transition[0,1] = state[1] + 0
        Transition[1,0] = state[0] + 0
        Transition[1,1] = state[1] + 1
        Transition[2,0] = state[0] - 1
        Transition[2,1] = state[1] + 0
        Transition[3,0] = state[0] + 0
        Transition[3,1] = state[1] - 1
        index_x = np.where(state_next[0] == Transition[:,0])[0]
        index_y = np.where(state_next[1] == Transition[:,1])[0]
        index = np.intersect1d(index_y,index_x)
        if index.size == 0:
            index = 4
        
        return index
    
    def StateTransition_simplified(original_actions, original_data):
        sim = np.empty((0,2))
        init = original_data[0,:]
        sim = np.append(sim, init.reshape(1,2), 0)
        for i in range(len(original_actions)):
            index = original_actions[i]
            Transition = np.zeros((5,2))
            Transition[0,0] = sim[i,0] + 1
            Transition[0,1] = sim[i,1] + 0
            Transition[1,0] = sim[i,0] + 0
            Transition[1,1] = sim[i,1] + 1
            Transition[2,0] = sim[i,0] - 1
            Transition[2,1] = sim[i,1] + 0
            Transition[3,0] = sim[i,0] + 0
            Transition[3,1] = sim[i,1] - 1
            Transition[4,:] = original_data[i,:]
            next_state = Transition[int(index),:]
            sim = np.append(sim,next_state.reshape(1,2),0)
            
        return sim    
    
    def TransitionCheck4Labels(state,state_next):
        Transition = np.zeros((8,2))
        Transition[0,0] = state[0] + 1
        Transition[0,1] = state[1] + 0
        Transition[1,0] = state[0] + 1
        Transition[1,1] = state[1] + 1
        Transition[2,0] = state[0] + 0
        Transition[2,1] = state[1] + 1
        Transition[3,0] = state[0] - 1
        Transition[3,1] = state[1] + 1
        Transition[4,0] = state[0] - 1
        Transition[4,1] = state[1] + 0
        Transition[5,0] = state[0] - 1
        Transition[5,1] = state[1] - 1
        Transition[6,0] = state[0] + 0
        Transition[6,1] = state[1] - 1
        Transition[7,0] = state[0] + 1
        Transition[7,1] = state[1] - 1
        index_x = np.where(state_next[0] == Transition[:,0])[0]
        index_y = np.where(state_next[1] == Transition[:,1])[0]
        index = np.intersect1d(index_y,index_x)
        if index.size == 0:
            index = 8
        
        return index
    
    def StateTransition(original_actions, original_data):
        sim = np.empty((0,2))
        init = original_data[0,:]
        sim = np.append(sim, init.reshape(1,2), 0)
        for i in range(len(original_actions)):
            index = original_actions[i]
            Transition = np.zeros((9,2))
            Transition[0,0] = sim[i,0] + 1
            Transition[0,1] = sim[i,1] + 0
            Transition[1,0] = sim[i,0] + 1
            Transition[1,1] = sim[i,1] + 1
            Transition[2,0] = sim[i,0] + 0
            Transition[2,1] = sim[i,1] + 1
            Transition[3,0] = sim[i,0] - 1
            Transition[3,1] = sim[i,1] + 1
            Transition[4,0] = sim[i,0] - 1
            Transition[4,1] = sim[i,1] + 0
            Transition[5,0] = sim[i,0] - 1
            Transition[5,1] = sim[i,1] - 1
            Transition[6,0] = sim[i,0] + 0
            Transition[6,1] = sim[i,1] - 1
            Transition[7,0] = sim[i,0] + 1
            Transition[7,1] = sim[i,1] - 1
            Transition[8,:] = original_data[i,:]
            next_state = Transition[int(index),:]
            sim = np.append(sim,next_state.reshape(1,2),0)
            
        return sim
    
    def GetDirectionFromAngle(angle, version):
        
        if version == 'simplified':
            if angle<0:
                angle = angle + 360
            slots = np.arange(45,410,90)
            label_direction = np.min(np.where(angle<=slots)[0])
            if label_direction==4:
                label_direction = 0
            
        elif version == 'complete':
            if angle<0:
                angle = angle + 360
            slots = np.arange(22.5,410,45)
            label_direction = np.min(np.where(angle<=slots)[0])
            if label_direction==8:
                label_direction = 0            
         
        return label_direction
    
    def GeneratePsi(Simulated_states, coin_location, version):      
        reward = 0
        see_coin_array = np.empty((0))
        coin_direction_array = np.empty((0))
        for i in range(len(Simulated_states)):
            see_coin = 0
            dist_from_coins = np.linalg.norm(coin_location-Simulated_states[i,:],2,1)
            l=0
            if np.min(dist_from_coins)<=8:
                index_min = np.argmin(dist_from_coins,0)
                closer_coin_position = coin_location[index_min,:]
                closer_coin_relative_position = np.array([closer_coin_position[0]-Simulated_states[i,0],closer_coin_position[1]-Simulated_states[i,1]])
                angle = np.arctan2(closer_coin_relative_position[1],closer_coin_relative_position[0])*180/np.pi
                coin_direction = Foraging.GetDirectionFromAngle(angle, version)  
            else:
                coin_direction = 8                
                
            for k in range(len(dist_from_coins)):
                if dist_from_coins[k]<=8:
                    see_coin = 1
                if dist_from_coins[k]<=3:
                    coin_location = np.delete(coin_location, l, 0)
                    reward = reward+1
                else:
                    l=l+1
                    
            see_coin_array = np.append(see_coin_array, see_coin)                 
            coin_direction_array = np.append(coin_direction_array, coin_direction)
                       
        return see_coin_array, coin_direction_array, reward
    
    def CoinLocation(Folder, experiment, version = 'distr_only'):
        N_coins = 325
        with open("4_walls_coins_task/FONC_{}_DeID/Behavioral/runNumber{}_coin_location.txt".format(Folder,experiment)) as f:
            coin_location_raw = f.readlines()
            
        if len(coin_location_raw) > N_coins:
            start_counting = len(coin_location_raw) - N_coins
            coin_location_raw = coin_location_raw[start_counting:]

        for i in range(len(coin_location_raw)):
            row = coin_location_raw[i][7:-2]
            row = row.replace('[', ',')
            coin_location_raw[i] = row
            
        coin_location_data = csv.reader(coin_location_raw)
        coin_location = np.empty((0,2))
        for row in coin_location_data:
            if len(row)==3:
                coin_location = np.append(coin_location, np.array([[np.round(float(row[0])), np.round(float(row[2]))]]),0)
            else:
                coin_location = np.append(coin_location, np.array([[np.round(float(row[1])), np.round(float(row[3]))]]),0) 
        
        if version == 'distr_only': 
            bool_distribution = np.empty((4))
            j=0
            for i in range(len(coin_location)):
                bool_distribution[0] = (coin_location[j,0]-60)**2 + (coin_location[j,1]-75)**2 <= (2*5)**2
                bool_distribution[1] = (coin_location[j,0]+15)**2 + (coin_location[j,1]+50)**2 <= (2*11)**2
                bool_distribution[2] = (coin_location[j,0]+50)**2 + (coin_location[j,1]-30)**2 <= (2*18)**2
                bool_distribution[3] = (coin_location[j,0]-49)**2 + (coin_location[j,1]+40)**2 <= (2*13)**2
                
                if np.sum(bool_distribution)==0:
                    coin_location = np.delete(coin_location, j, 0) 
                else:
                    j = j+1                
                
        return coin_location
    
    def TrueData(Folder, experiment):
        
        with open("4_walls_coins_task/FONC_{}_DeID/Behavioral/runNumber{}_position.txt".format(Folder,experiment)) as f:
            data_raw = f.readlines()
    
        for i in range(len(data_raw)):
            row = data_raw[i][1:]
            row = row.replace(']', ',')
            data_raw[i] = row

        agent_data = csv.reader(data_raw)
        Training_set = np.empty((0,2))
        True_set = np.empty((0,2))
        time = np.empty((0,1))

        for row in agent_data:
            True_set = np.append(True_set, np.array([[float(row[0]), float(row[2])]]), 0)
            Training_set = np.append(Training_set, np.array([[np.round(float(row[0])), np.round(float(row[2]))]]), 0)
            time = np.append(time, float(row[3]))    
            
        State_space, Training_set_index = np.unique(Training_set, return_index=True, axis = 0)    
        Training_set_cleaned = Training_set[np.sort(Training_set_index),:]
        time_cleaned = time[np.sort(Training_set_index)]
        True_Training_set = True_set
        
        return True_Training_set, Training_set_cleaned, time_cleaned
            
        
    def ProcessData(Folder, experiment, version, coins = 'full_coins'):
        coin_location = Foraging.CoinLocation(Folder, experiment, coins)
        
        with open("4_walls_coins_task/FONC_{}_DeID/Behavioral/runNumber{}_position.txt".format(Folder,experiment)) as f:
            data_raw = f.readlines()
    
        for i in range(len(data_raw)):
            row = data_raw[i][1:]
            row = row.replace(']', ',')
            data_raw[i] = row

        agent_data = csv.reader(data_raw)
        Training_set = np.empty((0,2))
        time = np.empty((0,1))

        for row in agent_data:
            Training_set = np.append(Training_set, np.array([[np.round(float(row[0])), np.round(float(row[2]))]]), 0)
            time = np.append(time, float(row[3]))
    
        State_space, Training_set_index = np.unique(Training_set, return_index=True, axis = 0)    
        Training_set_cleaned = Training_set[np.sort(Training_set_index),:]
        time_cleaned = time[np.sort(Training_set_index)]

        if version == 'complete':
            Labels = np.empty((0,1))
            for i in range(len(Training_set_cleaned)-1):
                index = Foraging.TransitionCheck4Labels(Training_set_cleaned[i,:], Training_set_cleaned[i+1,:])
                #index = Foraging.TransitionCheck4Labels_simplified(Training_set_cleaned[i,:], Training_set_cleaned[i+1,:])
                if index == 8:
                    dummy = 0
                else:
                    Labels = np.append(Labels, index)
            
            # % Simulate dynamics
            Simulated_states = Foraging.StateTransition(Labels, Training_set_cleaned)
            see_coin_array, coin_direction_array, reward = Foraging.GeneratePsi(Simulated_states, coin_location, version)
            
        if version == 'simplified':
            Labels = np.empty((0,1))
            for i in range(len(Training_set_cleaned)-1):
                index = Foraging.TransitionCheck4Labels_simplified(Training_set_cleaned[i,:], Training_set_cleaned[i+1,:])
                if index == 4:
                    dummy = 0
                else:
                    Labels = np.append(Labels, index)
            
            # % Simulate dynamics
            Simulated_states = Foraging.StateTransition_simplified(Labels, Training_set_cleaned)
            see_coin_array, coin_direction_array, reward = Foraging.GeneratePsi(Simulated_states, coin_location, version)          
         
        with open("4_walls_coins_task/FONC_{}_DeID/Behavioral/runNumber{}_master.txt".format(Folder, experiment)) as f:
            data_raw = f.readlines()
            
        Real_reward = len(data_raw)
                    
        return Simulated_states, Labels, time_cleaned, see_coin_array, coin_direction_array, reward, Real_reward



class Foraging:    
    def TransitionCheck4Labels_simplified(state,state_next):
        Transition = np.zeros((4,2))
        Transition[0,0] = state[0] + 1
        Transition[0,1] = state[1] + 0
        Transition[1,0] = state[0] + 0
        Transition[1,1] = state[1] + 1
        Transition[2,0] = state[0] - 1
        Transition[2,1] = state[1] + 0
        Transition[3,0] = state[0] + 0
        Transition[3,1] = state[1] - 1
        index_x = np.where(state_next[0] == Transition[:,0])[0]
        index_y = np.where(state_next[1] == Transition[:,1])[0]
        index = np.intersect1d(index_y,index_x)
        if index.size == 0:
            index = 4
        
        return index
    
    def StateTransition_simplified(original_actions, original_data):
        sim = np.empty((0,2))
        init = original_data[0,:]
        sim = np.append(sim, init.reshape(1,2), 0)
        for i in range(len(original_actions)):
            index = original_actions[i]
            Transition = np.zeros((5,2))
            Transition[0,0] = sim[i,0] + 1
            Transition[0,1] = sim[i,1] + 0
            Transition[1,0] = sim[i,0] + 0
            Transition[1,1] = sim[i,1] + 1
            Transition[2,0] = sim[i,0] - 1
            Transition[2,1] = sim[i,1] + 0
            Transition[3,0] = sim[i,0] + 0
            Transition[3,1] = sim[i,1] - 1
            Transition[4,:] = original_data[i,:]
            next_state = Transition[int(index),:]
            sim = np.append(sim,next_state.reshape(1,2),0)
            
        return sim    
    
    def TransitionCheck4Labels(state,state_next):
        Transition = np.zeros((8,2))
        Transition[0,0] = state[0] + 1
        Transition[0,1] = state[1] + 0
        Transition[1,0] = state[0] + 1
        Transition[1,1] = state[1] + 1
        Transition[2,0] = state[0] + 0
        Transition[2,1] = state[1] + 1
        Transition[3,0] = state[0] - 1
        Transition[3,1] = state[1] + 1
        Transition[4,0] = state[0] - 1
        Transition[4,1] = state[1] + 0
        Transition[5,0] = state[0] - 1
        Transition[5,1] = state[1] - 1
        Transition[6,0] = state[0] + 0
        Transition[6,1] = state[1] - 1
        Transition[7,0] = state[0] + 1
        Transition[7,1] = state[1] - 1
        index_x = np.where(state_next[0] == Transition[:,0])[0]
        index_y = np.where(state_next[1] == Transition[:,1])[0]
        index = np.intersect1d(index_y,index_x)
        if index.size == 0:
            index = 8
        
        return index
    
    def StateTransition(original_actions, original_data):
        sim = np.empty((0,2))
        init = original_data[0,:]
        sim = np.append(sim, init.reshape(1,2), 0)
        for i in range(len(original_actions)): #range(len(original_actions)):
            index = original_actions[i]
            Transition = np.zeros((9,2))
            Transition[0,0] = sim[i,0] + 1
            Transition[0,1] = sim[i,1] + 0
            Transition[1,0] = sim[i,0] + 1
            Transition[1,1] = sim[i,1] + 1
            Transition[2,0] = sim[i,0] + 0
            Transition[2,1] = sim[i,1] + 1
            Transition[3,0] = sim[i,0] - 1
            Transition[3,1] = sim[i,1] + 1
            Transition[4,0] = sim[i,0] - 1
            Transition[4,1] = sim[i,1] + 0
            Transition[5,0] = sim[i,0] - 1
            Transition[5,1] = sim[i,1] - 1
            Transition[6,0] = sim[i,0] + 0
            Transition[6,1] = sim[i,1] - 1
            Transition[7,0] = sim[i,0] + 1
            Transition[7,1] = sim[i,1] - 1
            Transition[8,:] = original_data[i,:]
            next_state = Transition[int(index),:]
            sim = np.append(sim,next_state.reshape(1,2),0)
            
        return sim
    
    def GetDirectionFromAngle(angle, version):
        
        if version == 'simplified':
            if angle<0:
                angle = angle + 360
            slots = np.arange(45,410,90)
            label_direction = np.min(np.where(angle<=slots)[0])
            if label_direction==4:
                label_direction = 0
            
        elif version == 'complete':
            if angle<0:
                angle = angle + 360
            slots = np.arange(22.5,410,45)
            label_direction = np.min(np.where(angle<=slots)[0])
            if label_direction==8:
                label_direction = 0            
         
        return label_direction
    
    def GeneratePsi(Simulated_states, coin_location, version):      
        reward = 0
        see_coin_array = np.empty((0))
        coin_direction_array = np.empty((0))
        for i in range(len(Simulated_states)):
            see_coin = 0
            dist_from_coins = np.linalg.norm(coin_location-Simulated_states[i,:],2,1)
            l=0
            if np.min(dist_from_coins)<=8:
                index_min = np.argmin(dist_from_coins,0)
                closer_coin_position = coin_location[index_min,:]
                closer_coin_relative_position = np.array([closer_coin_position[0]-Simulated_states[i,0],closer_coin_position[1]-Simulated_states[i,1]])
                angle = np.arctan2(closer_coin_relative_position[1],closer_coin_relative_position[0])*180/np.pi
                coin_direction = Foraging.GetDirectionFromAngle(angle, version)  
            else:
                coin_direction = 8                
                
            for k in range(len(dist_from_coins)):
                if dist_from_coins[k]<=8:
                    see_coin = 1
                if dist_from_coins[k]<=3:
                    coin_location = np.delete(coin_location, l, 0)
                    reward = reward+1
                else:
                    l=l+1
                    
            see_coin_array = np.append(see_coin_array, see_coin)                 
            coin_direction_array = np.append(coin_direction_array, coin_direction)
                       
        return see_coin_array, coin_direction_array, reward
    
    def CoinLocation(Folder, experiment, version = 'distr_only'):
        N_coins = 325
        with open("4_walls_coins_task/FONC_{}_DeID/fMRI/runNumber{}_coin_location.txt".format(Folder,experiment)) as f:
            coin_location_raw = f.readlines()
            
        if len(coin_location_raw) > N_coins:
            start_counting = len(coin_location_raw) - N_coins
            coin_location_raw = coin_location_raw[start_counting:]

        for i in range(len(coin_location_raw)):
            row = coin_location_raw[i][7:-2]
            row = row.replace('[', ',')
            coin_location_raw[i] = row
            
        coin_location_data = csv.reader(coin_location_raw)
        coin_location = np.empty((0,2))
        for row in coin_location_data:
            if len(row)==3:
                coin_location = np.append(coin_location, np.array([[np.round(float(row[0])), np.round(float(row[2]))]]),0)
            else:
                coin_location = np.append(coin_location, np.array([[np.round(float(row[1])), np.round(float(row[3]))]]),0) 
        
        if version == 'distr_only': 
            bool_distribution = np.empty((4))
            j=0
            for i in range(len(coin_location)):
                bool_distribution[0] = (coin_location[j,0]-60)**2 + (coin_location[j,1]-75)**2 <= (2*5)**2
                bool_distribution[1] = (coin_location[j,0]+15)**2 + (coin_location[j,1]+50)**2 <= (2*11)**2
                bool_distribution[2] = (coin_location[j,0]+50)**2 + (coin_location[j,1]-30)**2 <= (2*18)**2
                bool_distribution[3] = (coin_location[j,0]-49)**2 + (coin_location[j,1]+40)**2 <= (2*13)**2
                
                if np.sum(bool_distribution)==0:
                    coin_location = np.delete(coin_location, j, 0) 
                else:
                    j = j+1                
                
        return coin_location
    
    def TrueData(Folder, experiment):
        
        with open("4_walls_coins_task/FONC_{}_DeID/fMRI/runNumber{}_position.txt".format(Folder,experiment)) as f:
            data_raw = f.readlines()
    
        for i in range(len(data_raw)):
            row = data_raw[i][1:]
            row = row.replace(']', ',')
            data_raw[i] = row

        agent_data = csv.reader(data_raw)
        Training_set = np.empty((0,2))
        True_set = np.empty((0,2))
        time = np.empty((0,1))

        for row in agent_data:
            True_set = np.append(True_set, np.array([[float(row[0]), float(row[2])]]), 0)
            Training_set = np.append(Training_set, np.array([[np.round(float(row[0])), np.round(float(row[2]))]]), 0)
            time = np.append(time, float(row[3]))    
            
        State_space, Training_set_index = np.unique(Training_set, return_index=True, axis = 0)    
        Training_set_cleaned = Training_set[np.sort(Training_set_index),:]
        time_cleaned = time[np.sort(Training_set_index)]
        True_Training_set = True_set
        
        return True_Training_set, Training_set_cleaned, time_cleaned
            
        
    def ProcessData(Folder, experiment, version, coins = 'full_coins'):
        coin_location = Foraging.CoinLocation(Folder, experiment, coins)
        
        with open("4_walls_coins_task/FONC_{}_DeID/fMRI/runNumber{}_position.txt".format(Folder,experiment)) as f:
            data_raw = f.readlines()
    
        for i in range(len(data_raw)):
            row = data_raw[i][1:]
            row = row.replace(']', ',')
            data_raw[i] = row

        agent_data = csv.reader(data_raw)
        Training_set = np.empty((0,2))
        time = np.empty((0,1))

        for row in agent_data:
            Training_set = np.append(Training_set, np.array([[np.round(float(row[0])), np.round(float(row[2]))]]), 0)
            time = np.append(time, float(row[3]))
    
        State_space, Training_set_index = np.unique(Training_set, return_index=True, axis=0)    
        Training_set_cleaned = Training_set[np.sort(Training_set_index),:]
        time_cleaned = time[np.sort(Training_set_index)]

        if version == 'complete':
            Labels = np.empty((0,1))
            for i in range(len(Training_set_cleaned)-1):
                index = Foraging.TransitionCheck4Labels(Training_set_cleaned[i,:], Training_set_cleaned[i+1,:])
                #index = Foraging.TransitionCheck4Labels_simplified(Training_set_cleaned[i,:], Training_set_cleaned[i+1,:])
                if index == 8:
                    dummy = 0
                else:
                    Labels = np.append(Labels, index)
            
            # % Simulate dynamics
            Simulated_states = Foraging.StateTransition(Labels, Training_set_cleaned)
            see_coin_array, coin_direction_array, reward = Foraging.GeneratePsi(Simulated_states, coin_location, version)
            
        if version == 'simplified':
            Labels = np.empty((0,1))
            for i in range(len(Training_set_cleaned)-1):
                index = Foraging.TransitionCheck4Labels_simplified(Training_set_cleaned[i,:], Training_set_cleaned[i+1,:])
                if index == 4:
                    dummy = 0
                else:
                    Labels = np.append(Labels, index)
            
            # % Simulate dynamics
            Simulated_states = Foraging.StateTransition_simplified(Labels, Training_set_cleaned)
            see_coin_array, coin_direction_array, reward = Foraging.GeneratePsi(Simulated_states, coin_location, version)   
         
        with open("4_walls_coins_task/FONC_{}_DeID/fMRI/runNumber{}_master.txt".format(Folder, experiment)) as f:
            data_raw = f.readlines()
            
        Real_reward = len(data_raw)
            
        return Simulated_states, Labels, time_cleaned, see_coin_array, coin_direction_array, reward, Real_reward, coin_location
    
    
    class env:
        def __init__(self,  coins_location = coins_location_standard, init_state = np.array([0,0,0,8]), version = 'complete', max_episode_steps = 3000):
            self.state = init_state
            self.version = version
            self.coin_initial = 0.1*coins_location
            self.coin_location = 0.1*coins_location
            self.observation_space = np.array([len(self.state)])
            if version == 'complete':
                self.action_size = 8
            elif version == 'simplified':
                self.action_size = 4
                
            self._max_episode_steps = max_episode_steps
            self.step_counter = 0
                
        def Seed(self, seed):
            self.seed = seed
            np.random.seed(self.seed)
                
        def reset(self, version = 'standard', init_state = np.array([0,0,0,8])):
            if version == 'standard':
                self.state = init_state
                self.coin_location = self.coin_initial
                self.step_counter = 0
            else:
                state = 0.1*np.random.randint(-100,100,2)
                init_state = np.concatenate((state, np.array([0,8])))
                self.state = init_state
                self.coin_location = self.coin_initial
                self.step_counter = 0
                
            return self.state
                    
        def random_sample(self):
            return np.random.randint(0,self.action_size)
                             
        def Transition(state,action):
            Transition = np.zeros((9,2))
            Transition[0,0] = state[0] + 0.1
            Transition[0,1] = state[1] + 0
            Transition[1,0] = state[0] + 0.1
            Transition[1,1] = state[1] + 0.1
            Transition[2,0] = state[0] + 0
            Transition[2,1] = state[1] + 0.1
            Transition[3,0] = state[0] - 0.1
            Transition[3,1] = state[1] + 0.1
            Transition[4,0] = state[0] - 0.1
            Transition[4,1] = state[1] + 0
            Transition[5,0] = state[0] - 0.1
            Transition[5,1] = state[1] - 0.1
            Transition[6,0] = state[0] + 0
            Transition[6,1] = state[1] - 0.1
            Transition[7,0] = state[0] + 0.1
            Transition[7,1] = state[1] - 0.1
            Transition[8,:] = state
            state_plus1 = Transition[int(action),:]
            
            return state_plus1     

        def Transition_simplified(state,action):
            Transition = np.zeros((5,2))
            Transition[0,0] = state[0] + 0.1
            Transition[0,1] = state[1] + 0
            Transition[1,0] = state[0] + 0
            Transition[1,1] = state[1] + 0.1
            Transition[2,0] = state[0] - 0.1
            Transition[2,1] = state[1] + 0
            Transition[3,0] = state[0] + 0
            Transition[3,1] = state[1] - 0.1
            Transition[4,:] = state
            state_plus1 = Transition[int(action),:]
            
            return state_plus1  
    
        def step(self, action):
            self.step_counter += 1
            r=0
            state_partial = self.state[0:2]
            # given action, draw next state
            if self.version == 'simplified':
                state_plus1_partial = Foraging.env.Transition_simplified(state_partial, action)
            elif self.version =='complete':
                state_plus1_partial = Foraging.env.Transition(state_partial, action)
                
            if state_plus1_partial[0]>10 or state_plus1_partial[0]<-10:
                state_plus1_partial[0] = state_partial[0] 

            if state_plus1_partial[1]>10 or state_plus1_partial[1]<-10:
                state_plus1_partial[1] = state_partial[1]                 
                    
            # Update psi and reward and closest coin direction
            dist_from_coins = np.linalg.norm(self.coin_location-state_plus1_partial,2,1)
            l=0
            psi = 0
                
            if np.min(dist_from_coins)<=0.8:
                index_min = np.argmin(dist_from_coins,0)
                closer_coin_position = self.coin_location[index_min,:]
                closer_coin_relative_position = np.array([closer_coin_position[0]-state_plus1_partial[0],closer_coin_position[1]-state_plus1_partial[1]])
                angle = np.arctan2(closer_coin_relative_position[1],closer_coin_relative_position[0])*180/np.pi
                coin_direction = Foraging.GetDirectionFromAngle(angle, self.version)  
            else:
                coin_direction = 8   
            
            for p in range(len(dist_from_coins)):
                if dist_from_coins[p]<=0.8:
                    psi = 1
                if dist_from_coins[p]<=0.3:
                    self.coin_location = np.delete(self.coin_location, l, 0)
                    r = r+1
                else:
                    l=l+1
                    
            state_plus1 = np.concatenate((state_plus1_partial, [psi], [coin_direction]))
            self.state = state_plus1
            if self.step_counter >= self._max_episode_steps:
                done = True
            else:
                done = False
            
            return state_plus1, r, done, False
        
    class env_allocentric_only:
        def __init__(self,  coins_location = coins_location_standard, init_state = np.array([0,0,0,0]), version = 'complete', max_episode_steps = 3000):
            self.state = init_state
            self.version = version
            self.coin_initial = 0.1*coins_location
            self.coin_location = 0.1*coins_location
            self.observation_space = np.array([len(self.state)])
            if version == 'complete':
                self.action_size = 8
            elif version == 'simplified':
                self.action_size = 4
                
            self._max_episode_steps = max_episode_steps
            self.step_counter = 0
                
        def Seed(self, seed):
            self.seed = seed
            np.random.seed(self.seed)
                
        def reset(self, version = 'standard', init_state = np.array([0,0,0,0])):
            if version == 'standard':
                self.state = init_state
                self.coin_location = self.coin_initial
                self.step_counter = 0
            else:
                state = 0.1*np.random.randint(-100,100,2)
                init_state = np.concatenate((state, np.array([0,0])))
                self.state = init_state
                self.coin_location = self.coin_initial
                self.step_counter = 0
                
            return self.state
                    
        def random_sample(self):
            return np.random.randint(0,self.action_size)
                             
        def Transition(state,action):
            Transition = np.zeros((9,2))
            Transition[0,0] = state[0] + 0.1
            Transition[0,1] = state[1] + 0
            Transition[1,0] = state[0] + 0.1
            Transition[1,1] = state[1] + 0.1
            Transition[2,0] = state[0] + 0
            Transition[2,1] = state[1] + 0.1
            Transition[3,0] = state[0] - 0.1
            Transition[3,1] = state[1] + 0.1
            Transition[4,0] = state[0] - 0.1
            Transition[4,1] = state[1] + 0
            Transition[5,0] = state[0] - 0.1
            Transition[5,1] = state[1] - 0.1
            Transition[6,0] = state[0] + 0
            Transition[6,1] = state[1] - 0.1
            Transition[7,0] = state[0] + 0.1
            Transition[7,1] = state[1] - 0.1
            Transition[8,:] = state
            state_plus1 = Transition[int(action),:]
            
            return state_plus1     

        def Transition_simplified(state,action):
            Transition = np.zeros((5,2))
            Transition[0,0] = state[0] + 0.1
            Transition[0,1] = state[1] + 0
            Transition[1,0] = state[0] + 0
            Transition[1,1] = state[1] + 0.1
            Transition[2,0] = state[0] - 0.1
            Transition[2,1] = state[1] + 0
            Transition[3,0] = state[0] + 0
            Transition[3,1] = state[1] - 0.1
            Transition[4,:] = state
            state_plus1 = Transition[int(action),:]
            
            return state_plus1  
    
        def step(self, action):
            self.step_counter += 1
            r=0
            state_partial = self.state[0:2]
            # given action, draw next state
            if self.version == 'simplified':
                state_plus1_partial = Foraging.env.Transition_simplified(state_partial, action)
            elif self.version =='complete':
                state_plus1_partial = Foraging.env.Transition(state_partial, action)
                
            if state_plus1_partial[0]>10 or state_plus1_partial[0]<-10:
                state_plus1_partial[0] = state_partial[0] 

            if state_plus1_partial[1]>10 or state_plus1_partial[1]<-10:
                state_plus1_partial[1] = state_partial[1]                 
                    
            psi = 0
            coin_direction = 0   
            
            dist_from_coins = np.linalg.norm(self.coin_location-state_plus1_partial,2,1)
            l=0
            psi = 0
                 
            for p in range(len(dist_from_coins)):
                if dist_from_coins[p]<=0.3:
                    self.coin_location = np.delete(self.coin_location, l, 0)
                    r = r+1
                else:
                    l=l+1
                                
            state_plus1 = np.concatenate((state_plus1_partial, [psi], [coin_direction]))
            self.state = state_plus1
            if self.step_counter >= self._max_episode_steps:
                done = True
            else:
                done = False
            
            return state_plus1, r, done, False        
            

