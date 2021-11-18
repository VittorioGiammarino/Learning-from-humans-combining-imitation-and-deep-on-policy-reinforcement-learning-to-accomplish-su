#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 12 09:54:01 2021

@author: vittorio
"""
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models import SoftmaxHierarchicalActor
from models import Value_net

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class PPO:
    def __init__(self, state_dim, action_dim, encoding_info = None, num_steps_per_rollout=15000, lr = 3e-4, gae_gamma = 0.99, gae_lambda = 0.99, 
                 epsilon = 0.2, c1 = 1, c2 = 1e-2, minibatch_size=64, num_epochs=10, lambda_gail = 1e-1):
        
        self.actor = SoftmaxHierarchicalActor.NN_PI_LO(state_dim, action_dim).to(device)
        self.value_function = Value_net(state_dim).to(device)
        
        self.optimizer_actor = torch.optim.Adam(self.actor.parameters(), lr)
        self.optimizer_value_function = torch.optim.Adam(self.value_function.parameters(), lr)
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.encoding_info = encoding_info

        self.num_steps_per_rollout = num_steps_per_rollout
        self.gae_gamma = gae_gamma
        self.gae_lambda = gae_lambda
        self.epsilon = epsilon
        self.c1 = c1
        self.c2 = c2
        self.minibatch_size = minibatch_size
        self.num_epochs = num_epochs
        self.lambda_gail = lambda_gail
        
        self.Total_t = 0
        self.Total_iter = 0
        self.states = []
        self.actions = []
        self.returns = []
        self.advantage = []
        self.gammas = []
        
    def reset_counters(self):
        self.Total_t = 0
        self.Total_iter = 0
        
    def encode_state(self, state):
        state = state.flatten()
        coordinates = state[0:2]
        psi = state[2]
        psi_encoded = np.zeros(self.encoding_info[0])
        psi_encoded[int(psi)]=1
        coin_dir_encoded = np.zeros(self.encoding_info[1])
        coin_dir = state[3]
        coin_dir_encoded[int(coin_dir)]=1
        current_state_encoded = np.concatenate((coordinates,psi_encoded,coin_dir_encoded))
        return current_state_encoded
    
    def encode_action(self, action):
        action_encoded = np.zeros(self.action_dim)
        action_encoded[int(action)]=1
        return action_encoded
        
    def select_action(self, state):
        state = PPO.encode_state(self, state)
        state = torch.FloatTensor(state.reshape(1,-1)).to(device)
        prob_u = self.actor(state).cpu().data.numpy()
        prob_u_rescaled = np.divide(prob_u,np.amin(prob_u)+0.01)
        for i in range(1,prob_u_rescaled.shape[1]):
            prob_u_rescaled[0,i]=prob_u_rescaled[0,i]+prob_u_rescaled[0,i-1]
        draw_u=np.divide(np.random.rand(),np.amin(prob_u)+0.01)
        temp = np.where(draw_u<=prob_u_rescaled)[1]
        if temp.size == 0:
            action = np.argmax(prob_u)
        else:
            action = np.amin(np.where(draw_u<=prob_u_rescaled)[1])
        return int(action)
        
    def GAE(self, env, GAIL = False, Discriminator = None, reset = 'random', init_state = np.array([0,0,0,8]), Mixed_GAIL = False):
        step = 0
        self.Total_iter += 1
        self.states = []
        self.actions = []
        self.returns = []
        self.advantage = []
        self.gammas = []
        while step < self.num_steps_per_rollout: 
            episode_states = []
            episode_actions = []
            episode_rewards = []
            episode_gammas = []
            episode_lambdas = []    
            state, done = env.reset(reset, init_state), False
            t=0
            episode_reward = 0

            while not done and step < self.num_steps_per_rollout:            
                action = PPO.select_action(self, state)
                
                state_encoded = PPO.encode_state(self, state.flatten())
            
                self.states.append(state_encoded)
                self.actions.append(action)
                episode_states.append(state_encoded)
                episode_actions.append(action)
                episode_gammas.append(self.gae_gamma**t)
                episode_lambdas.append(self.gae_lambda**t)
                
                state, reward, done, _ = env.step(action)
                
                episode_rewards.append(reward)
            
                t+=1
                step+=1
                episode_reward+=reward
                self.Total_t += 1
                        
            if done: 
                # +1 to account for 0 indexing. +0 on ep_timesteps since it will increment +1 even if done=True
                print(f"Total T: {self.Total_t}, Iter Num: {self.Total_iter}, Episode T: {t} Reward: {episode_reward:.3f}")
                
            episode_states = torch.FloatTensor(episode_states)
            episode_actions = torch.LongTensor(episode_actions)
            episode_rewards = torch.FloatTensor(episode_rewards)
            episode_gammas = torch.FloatTensor(episode_gammas)
            episode_lambdas = torch.FloatTensor(episode_lambdas)        
            
            if GAIL and Mixed_GAIL and self.Total_iter>1:
                episode_actions = F.one_hot(episode_actions, num_classes=self.action_dim)
                episode_rewards = episode_rewards - self.lambda_gail*torch.log(Discriminator(episode_states, episode_actions)).squeeze().detach()
            elif GAIL and self.Total_iter>1:
                episode_actions = F.one_hot(episode_actions, num_classes=self.action_dim)
                episode_rewards = -torch.log(Discriminator(episode_states, episode_actions)).squeeze().detach()
                
            episode_discounted_rewards = episode_gammas*episode_rewards
            episode_discounted_returns = torch.FloatTensor([sum(episode_discounted_rewards[i:]) for i in range(t)])
            episode_returns = episode_discounted_returns/episode_gammas
            
            self.returns.append(episode_returns)
            self.value_function.eval()
            current_values = self.value_function(episode_states).detach()
            next_values = torch.cat((self.value_function(episode_states)[1:], torch.FloatTensor([[0.]]))).detach()
            episode_deltas = episode_rewards.unsqueeze(-1) + self.gae_gamma*next_values - current_values
            episode_advantage = torch.FloatTensor([((episode_gammas*episode_lambdas)[:t-j].unsqueeze(-1)*episode_deltas[j:]).sum() for j in range(t)])
            
            self.advantage.append(episode_advantage)
            self.gammas.append(episode_gammas)
            
        rollout_states = torch.FloatTensor(self.states)
        rollout_actions = torch.LongTensor(np.array(self.actions))

        return rollout_states, rollout_actions
    
    
    def train(self, Entropy = False):
        
        rollout_states = torch.FloatTensor(self.states)
        rollout_actions = torch.LongTensor(np.array(self.actions))
        rollout_returns = torch.cat(self.returns)
        rollout_advantage = torch.cat(self.advantage)
        rollout_gammas = torch.cat(self.gammas)         
        
        rollout_advantage = ((rollout_advantage-rollout_advantage.mean())/rollout_advantage.std()).reshape(-1,1)
        
        self.actor.eval()
        old_log_prob, old_log_prob_rollout = self.actor.sample_log(rollout_states, rollout_actions)
        old_log_prob = old_log_prob.detach()
        old_log_prob_rollout = old_log_prob_rollout.detach()
        
        self.value_function.train()
        self.actor.train()
        
        max_steps = self.num_epochs * (self.num_steps_per_rollout // self.minibatch_size)
        
        for _ in range(max_steps):
            
            minibatch_indices = np.random.choice(range(self.num_steps_per_rollout), self.minibatch_size, False)
            batch_states=rollout_states[minibatch_indices]
            batch_actions = rollout_actions[minibatch_indices]
            batch_returns = rollout_returns[minibatch_indices]
            batch_advantage = rollout_advantage[minibatch_indices]
            batch_gammas = rollout_gammas[minibatch_indices]       
            
            log_prob, log_prob_rollout = self.actor.sample_log(batch_states, batch_actions)
            batch_old_log_pi = old_log_prob_rollout[minibatch_indices]
            
            r = torch.exp(log_prob_rollout - batch_old_log_pi)
            L_clip = torch.minimum(r*batch_advantage, torch.clip(r, 1-self.epsilon, 1+self.epsilon)*batch_advantage)
            L_vf = (self.value_function(batch_states).squeeze() - batch_returns)**2
            
            if Entropy:
                S = (-1)*torch.sum(torch.exp(log_prob)*log_prob, 1)
            else:
                S = torch.zeros_like(torch.sum(torch.exp(log_prob)*log_prob, 1))
                
            self.optimizer_value_function.zero_grad()
            self.optimizer_actor.zero_grad()
            loss = (-1) * (L_clip - self.c1 * L_vf + self.c2 * S).mean()
            loss.backward()
            self.optimizer_value_function.step()
            self.optimizer_actor.step()        
        
        
    def save_actor(self, filename):
        option = 0
        torch.save(self.actor.state_dict(), filename + f"_pi_lo_option_{option}")
        torch.save(self.optimizer_actor.state_dict(), filename + f"_pi_lo_optimizer_option_{option}")
    
    def load_actor(self, filename):
        option = 0
        self.actor.load_state_dict(torch.load(filename + f"_pi_lo_option_{option}"))
        self.optimizer_actor.load_state_dict(torch.load(filename + f"_pi_lo_optimizer_option_{option}"))

    def save_critic(self, filename):
        torch.save(self.value_function.state_dict(), filename + "_value_function")
        torch.save(self.optimizer_value_function.state_dict(), filename + "_value_function_optimizer")
    
    def load_critic(self, filename):
        self.value_function.load_state_dict(torch.load(filename + "_value_function"))      
        self.optimizer_value_function.load_state_dict(torch.load(filename + "_value_function_optimizer"))   
        
        

        
        
        
        
        
        

            
            
        
            
            
            

        
