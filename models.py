#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  9 17:55:49 2021

@author: vittorio
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical


class SoftmaxHierarchicalActor:
    class NN_PI_LO(nn.Module):
        def __init__(self, state_dim, action_dim):
            super(SoftmaxHierarchicalActor.NN_PI_LO, self).__init__()
            
            self.l1 = nn.Linear(state_dim, 128)
            nn.init.uniform_(self.l1.weight, -0.5, 0.5)
            self.l2 = nn.Linear(128,128)
            nn.init.uniform_(self.l2.weight, -0.5, 0.5)
            self.l3 = nn.Linear(128,action_dim)
            nn.init.uniform_(self.l3.weight, -0.5, 0.5)
            self.lS = nn.Softmax(dim=1)
            
        def forward(self, state):
            a = self.l1(state)
            a = F.relu(self.l2(a))
            return self.lS(self.l3(a))
        
        def sample(self, state):
            self.log_Soft = nn.LogSoftmax(dim=1)
            a = self.l1(state)
            a = F.relu(self.l2(a))
            log_prob = self.log_Soft(self.l3(a)) 
            
            prob = self.forward(state)
            m = Categorical(prob)
            action = m.sample()
            
            log_prob_sampled = log_prob[torch.arange(len(action)),action]
            
            return action, log_prob_sampled.reshape(-1,1)
        
        def sample_log(self, state, action):
            self.log_Soft = nn.LogSoftmax(dim=1)
            a = self.l1(state)
            a = F.relu(self.l2(a))
            log_prob = self.log_Soft(self.l3(a)) 
                        
            log_prob_sampled = log_prob[torch.arange(len(action)), action]
            
            return log_prob, log_prob_sampled.reshape(-1,1)
            
class Critic_flat_discrete(nn.Module):
    def __init__(self, state_dim, action_cardinality):
        super(Critic_flat_discrete, self).__init__()
        # Q1 architecture
        self.l1 = nn.Linear(state_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, action_cardinality)

        # Q2 architecture
        self.l4 = nn.Linear(state_dim, 256)
        self.l5 = nn.Linear(256, 256)
        self.l6 = nn.Linear(256, action_cardinality)

    def forward(self, state):      
        q1 = F.relu(self.l1(state))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)
        
        q2 = F.relu(self.l4(state))
        q2 = F.relu(self.l5(q2))
        q2 = self.l6(q2)
        return q1, q2

    def Q1(self, state):    
        q1 = F.relu(self.l1(state))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)
        return q1
    
class Value_net(nn.Module):
    def __init__(self, state_dim):
        super(Value_net, self).__init__()
        # Value_net architecture
        self.l1 = nn.Linear(state_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, 1)

    def forward(self, state):
        q1 = F.relu(self.l1(state))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)    
        return q1
    
class Discriminator(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Discriminator, self).__init__()

        # architecture
        self.l1 = nn.Linear(state_dim + action_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, 1)

    def forward(self, state, action):
        return torch.sigmoid(self.get_logits(state, action))

    def get_logits(self, state, action):
        state = torch.FloatTensor(state)
        action = torch.LongTensor(action)
        sa = torch.cat([state, action], 1)
        d = F.relu(self.l1(sa))
        d = F.relu(self.l2(d))
        d = self.l3(d)
        return d