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

device = torch.device("cpu")


class TRPO:
    def __init__(self, state_dim, action_dim, encoding_info = None, num_steps_per_rollout=15000, gae_gamma = 0.99, gae_lambda = 0.99, epsilon = 0.05, 
                 conj_grad_damping=0.1, lambda_ = 1e-3, lambda_gail = 1e-1):
        
        self.actor = SoftmaxHierarchicalActor.NN_PI_LO(state_dim, action_dim).to(device)
        self.value_function = Value_net(state_dim).to(device)
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.encoding_info = encoding_info

        self.num_steps_per_rollout = num_steps_per_rollout
        self.gae_gamma = gae_gamma
        self.gae_lambda = gae_lambda
        self.epsilon = epsilon
        self.conj_grad_damping = conj_grad_damping
        self.lambda_ = lambda_
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
        state = TRPO.encode_state(self, state)
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
                action = TRPO.select_action(self, state)
                
                state_encoded = TRPO.encode_state(self, state.flatten())
            
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
    
    def get_flat_grads(f, net):
        flat_grads = torch.cat([grad.view(-1) for grad in torch.autograd.grad(f, net.parameters(), create_graph=True)])
        return flat_grads
    
    def get_flat_params(net):
        return torch.cat([param.view(-1) for param in net.parameters()])
    
    def set_params(net, new_flat_params):
        start_idx = 0
        for param in net.parameters():
            end_idx = start_idx + np.prod(list(param.shape))
            param.data = torch.reshape(new_flat_params[start_idx:end_idx], param.shape)
            start_idx = end_idx
      
    def conjugate_gradient(Av_func, b, max_iter=10, residual_tol=1e-10):
        x = torch.zeros_like(b)
        r = b - Av_func(x)
        p = r
        rsold = r.norm() ** 2
    
        for _ in range(max_iter):
            Ap = Av_func(p)
            alpha = rsold / torch.dot(p, Ap)
            x = x + alpha * p
            r = r - alpha * Ap
            rsnew = r.norm() ** 2
            if torch.sqrt(rsnew) < residual_tol:
                break
            p = r + (rsnew / rsold) * p
            rsold = rsnew   
        return x
    
    def rescale_and_linesearch(self, g, s, Hs, L, kld, old_params, max_iter=20, success_ratio=1e-10):
        TRPO.set_params(self.actor, old_params)
        L_old = L().detach()
        max_kl = self.epsilon
        
        eta = torch.sqrt((2 * max_kl) / torch.dot(s, Hs))
    
        for _ in range(max_iter):
            new_params = old_params + eta * s
    
            TRPO.set_params(self.actor, new_params)
            kld_new = kld().detach()
    
            L_new = L().detach()
    
            actual_improv = L_new - L_old
            approx_improv = torch.dot(g, eta * s)
            ratio = actual_improv / approx_improv
    
            if ratio > success_ratio and actual_improv > 0 and kld_new < max_kl:
                return new_params, False
    
            eta *= 0.5
    
        print("The line search has failed!")
        return old_params, True
    
    def train(self, Entropy = False):
        
        rollout_states = torch.FloatTensor(self.states)
        rollout_actions = torch.LongTensor(np.array(self.actions))
        rollout_returns = torch.cat(self.returns)
        rollout_advantage = torch.cat(self.advantage)
        rollout_gammas = torch.cat(self.gammas)        
        
        rollout_advantage = ((rollout_advantage-rollout_advantage.mean())/rollout_advantage.std()).reshape(-1,1)
        
        self.value_function.train()
        old_params = TRPO.get_flat_params(self.value_function).detach()
        old_v = self.value_function(rollout_states).detach()
        
        def constraint():
            return ((old_v - self.value_function(rollout_states))**2).mean()
        
        gradient_constraint = TRPO.get_flat_grads(constraint(), self.value_function)
        
        def Hv(v):
            hessian_v = TRPO.get_flat_grads(torch.dot(gradient_constraint, v), self.value_function).detach()
            return hessian_v
        
        gradient = TRPO.get_flat_grads(((-1)*(self.value_function(rollout_states).squeeze() - rollout_returns)**2).mean(), self.value_function).detach()
        s = TRPO.conjugate_gradient(Hv, gradient).detach()
        Hessian_s = Hv(s).detach()
        alpha = torch.sqrt(2*self.epsilon/torch.dot(s,Hessian_s))
        new_params = old_params + alpha*s
        TRPO.set_params(self.value_function, new_params)
        
        self.actor.train()
        old_params = TRPO.get_flat_params(self.actor).detach()
        old_log_prob, old_log_prob_rollout = self.actor.sample_log(rollout_states, rollout_actions)
        
        def L():
            _, log_prob_rollout = self.actor.sample_log(rollout_states, rollout_actions)
            return (rollout_advantage*torch.exp(log_prob_rollout - old_log_prob_rollout.detach())).mean()
        
        def kld():
            prob = self.actor(rollout_states)
            divKL = F.kl_div(old_log_prob.detach(), prob, reduction = 'batchmean')
            return divKL
        
        grad_kld_old_param = TRPO.get_flat_grads(kld(), self.actor)
        
        def Hv(v):
            hessian_v = TRPO.get_flat_grads(torch.dot(grad_kld_old_param, v), self.actor).detach()
            return hessian_v + self.conj_grad_damping*v
        
        gradient = TRPO.get_flat_grads(L(), self.actor).detach()
        s = TRPO.conjugate_gradient(Hv, gradient).detach()
        Hs = Hv(s).detach()
        new_params, Failed = TRPO.rescale_and_linesearch(self, gradient, s, Hs, L, kld, old_params)
        
        if Entropy and not Failed:
            _, entropy_log_prob = self.actor.sample_log(rollout_states, rollout_actions)
            discounted_casual_entropy = ((-1)*rollout_gammas*entropy_log_prob).mean()
            gradient_discounted_casual_entropy = TRPO.get_flat_grads(discounted_casual_entropy, self.actor)
            new_params += self.lambda_*gradient_discounted_casual_entropy
            
        TRPO.set_params(self.actor, new_params)
        
    def save_actor(self, filename):
        option = 0
        torch.save(self.actor.state_dict(), filename + f"_pi_lo_option_{option}")
    
    def load_actor(self, filename):
        option = 0
        self.actor.load_state_dict(torch.load(filename + f"_pi_lo_option_{option}"))

    def save_critic(self, filename):
        torch.save(self.value_function.state_dict(), filename + "_value_function")
    
    def load_critic(self, filename):
        self.value_function.load_state_dict(torch.load(filename + "_value_function"))       
        
        
        
        
        
        

            
            
        
            
            
            

        