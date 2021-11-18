#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 11 11:32:32 2021

@author: vittorio
"""

import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models import Discriminator


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Implementation of Twin Delayed Deep Deterministic Policy Gradients (TD3)
# Paper: https://arxiv.org/abs/1802.09477

class Gail(object):
    def __init__(self, state_dim, action_dim, expert_states, expert_actions, l_rate = 1e-3):
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.expert_states = torch.FloatTensor(expert_states)
        self.expert_actions = torch.LongTensor(expert_actions)
        self.expert_actions = F.one_hot(self.expert_actions[:,0], num_classes=self.action_dim)
        self.learning_rate = l_rate
        self.discriminator = Discriminator(self.state_dim, self.action_dim)
        self.discriminator_optimizer = torch.optim.Adam(self.discriminator.parameters(), lr = self.learning_rate)
        
    def update(self, learner_states, learner_actions, iterations = 10):
        self.discriminator.train()
        learner_states = torch.FloatTensor(learner_states)
        learner_actions = torch.LongTensor(learner_actions)
        learner_actions = F.one_hot(learner_actions, num_classes=self.action_dim)
        
        for i in range(iterations):
            expert_scores = self.discriminator.get_logits(self.expert_states, self.expert_actions)
            learner_scores = self.discriminator.get_logits(learner_states, learner_actions)
            
            self.discriminator_optimizer.zero_grad()
            loss = F.binary_cross_entropy_with_logits(expert_scores, torch.zeros_like(expert_scores)) + F.binary_cross_entropy_with_logits(learner_scores, torch.ones_like(learner_scores))
            loss.backward()
            self.discriminator_optimizer.step()
            
        return expert_scores.mean(), learner_scores.mean()
        
        
        
        
        