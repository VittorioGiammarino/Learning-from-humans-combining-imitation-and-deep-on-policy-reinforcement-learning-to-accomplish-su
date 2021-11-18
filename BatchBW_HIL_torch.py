#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 23 12:06:27 2021

@author: vittorio
"""


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 19 17:40:50 2021

@author: vittorio
"""

import copy
import numpy as np
import World
from sklearn.preprocessing import OneHotEncoder

import torch
import torch.nn as nn
import torch.nn.functional as F

import os
import time

from models import SoftmaxHierarchicalActor

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class BatchBW(object):
    def __init__(self, state_dim, action_dim, option_dim, termination_dim, state_samples, action_samples, M_step_epoch, batch_size, l_rate, encoding_info = None):
        self.state_dim = state_dim
        self.action_dim = action_dim 
        self.option_dim = option_dim
        self.termination_dim = termination_dim
        self.TrainingSet = state_samples
        self.batch_size = batch_size
        self.mu = np.ones(option_dim)*np.divide(1,option_dim)
        self.action_space_discrete = len(np.unique(action_samples,axis=0))
        self.Labels = action_samples
        self.epochs = M_step_epoch

        # self.coordinates = 2
        self.encoding_info = encoding_info
        
        # self.observation_space_size_encoded = self.coordinates + self.view + self.closest_coin_dir
        # define hierarchical policy
        self.pi_hi = SoftmaxHierarchicalActor.NN_PI_HI(state_dim, option_dim).to(device)
        self.pi_b = [[None]*1 for _ in range(option_dim)] 
        self.pi_lo = [[None]*1 for _ in range(option_dim)] 
        pi_lo_temp = SoftmaxHierarchicalActor.NN_PI_LO(state_dim, self.action_space_discrete).to(device)
        pi_b_temp = SoftmaxHierarchicalActor.NN_PI_B(state_dim, termination_dim).to(device)
        for option in range(self.option_dim):
            self.pi_lo[option] = copy.deepcopy(pi_lo_temp)
            self.pi_b[option] = copy.deepcopy(pi_b_temp)
        # define optimizer 
        self.learning_rate = l_rate
        self.pi_hi_optimizer = torch.optim.Adam(self.pi_hi.parameters(), lr=self.learning_rate)
        self.pi_b_optimizer = [[None]*1 for _ in range(option_dim)] 
        self.pi_lo_optimizer = [[None]*1 for _ in range(option_dim)] 
        for option in range(self.option_dim):
            self.pi_lo_optimizer[option] = torch.optim.Adam(self.pi_lo[option].parameters(), lr=self.learning_rate)
            self.pi_b_optimizer[option] = torch.optim.Adam(self.pi_b[option].parameters(), lr=self.learning_rate) 
            
    def reset_learning_rate(self, new_rate):
        self.learning_rate = new_rate
        self.pi_hi_optimizer = torch.optim.Adam(self.pi_hi.parameters(), lr=self.learning_rate)
        self.pi_b_optimizer = [[None]*1 for _ in range(self.option_dim)] 
        self.pi_lo_optimizer = [[None]*1 for _ in range(self.option_dim)] 
        for option in range(self.option_dim):
            self.pi_lo_optimizer[option] = torch.optim.Adam(self.pi_lo[option].parameters(), lr=self.learning_rate)
            self.pi_b_optimizer[option] = torch.optim.Adam(self.pi_b[option].parameters(), lr=self.learning_rate) 
                 
    def pretrain_pi_hi(self, epochs, supervised_options):
        n_batches = np.int(self.TrainingSet.shape[0]/self.batch_size)
        supervised_options = supervised_options.reshape(len(supervised_options),1)
        criterion = torch.nn.CrossEntropyLoss()
        print(f"pretrain pi hi for {epochs} epochs")
        for t in range(epochs):
            for n in range(n_batches):
                TrainingSet = torch.FloatTensor(self.TrainingSet[n*self.batch_size:(n+1)*self.batch_size,:]).to(device)
                Labels = torch.LongTensor(supervised_options[n*self.batch_size:(n+1)*self.batch_size,0]).to(device)
                y_pred = self.pi_hi(TrainingSet)
                loss = criterion(y_pred, Labels)
                self.pi_hi_optimizer.zero_grad()
                loss.backward()
                self.pi_hi_optimizer.step()
            
            onehot_encoder = OneHotEncoder(sparse=False)
            onehot_encoded_psi = onehot_encoder.fit_transform(supervised_options)
            accuracy = 1 - np.sum(np.abs(self.pi_hi(torch.FloatTensor(self.TrainingSet)).detach().numpy()-onehot_encoded_psi))/(2*len(self.TrainingSet[:,2]))
        print(f"epoch {t}: accuracy {accuracy}")
            
    def prepare_labels_pretrain_pi_b(self, supervised_options):
        Labels_b = []
        for i in range(self.option_dim):
            Labels_b.append(np.ones((len(supervised_options),1)))                
        for i in range(len(supervised_options)):
            option = int(supervised_options[i])
            Labels_b[option][i,0] = 0
        return Labels_b
            
    def pretrain_pi_b(self, epochs, Labels, option):
        n_batches = np.int(self.TrainingSet.shape[0]/self.batch_size)
        Labels = Labels.reshape(len(Labels),1)
        criterion = torch.nn.CrossEntropyLoss()
        print(f"pretrain pi b (option {option+1}) for {epochs} epochs")
        for t in range(epochs):
            for n in range(n_batches):
                TrainingSet = torch.FloatTensor(self.TrainingSet[n*self.batch_size:(n+1)*self.batch_size,:]).to(device)
                Labels_b = torch.LongTensor(Labels[n*self.batch_size:(n+1)*self.batch_size,0]).to(device)
                y_pred = self.pi_b[option](TrainingSet)
                loss = criterion(y_pred, Labels_b)
                self.pi_b_optimizer[option].zero_grad()
                loss.backward()
                self.pi_b_optimizer[option].step()
            
            onehot_encoder = OneHotEncoder(sparse=False)
            onehot_encoded_lab_b = onehot_encoder.fit_transform(Labels)
            accuracy = 1 - np.sum(np.abs(self.pi_b[option](torch.FloatTensor(self.TrainingSet)).detach().numpy()-onehot_encoded_lab_b))/(2*len(self.TrainingSet[:,2]))
        print(f"epoch {t}: accuracy {accuracy}")   
        
    def select_action(self, state, option):
        state = BatchBW.encode_state(self, state)
        state = torch.FloatTensor(state.reshape(1,-1)).to(device)
        prob_u = self.pi_lo[option](state).cpu().data.numpy()
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
        
    def select_option(self, state, b, previous_option):
        state = BatchBW.encode_state(self, state)
        state = torch.FloatTensor(state.reshape(1,-1)).to(device)     
        if b == 1:
            b_bool = True
        else:
            b_bool = False

        o_prob_tilde = np.empty((1,self.option_dim))
        if b_bool == True:
            o_prob_tilde = self.pi_hi(state).cpu().data.numpy()
        else:
            o_prob_tilde[0,:] = 0
            o_prob_tilde[0,previous_option] = 1

        prob_o = o_prob_tilde
        prob_o_rescaled = np.divide(prob_o, np.amin(prob_o)+0.01)
        for i in range(1,prob_o_rescaled.shape[1]):
            prob_o_rescaled[0,i]=prob_o_rescaled[0,i]+prob_o_rescaled[0,i-1]
        draw_o=np.divide(np.random.rand(), np.amin(prob_o)+0.01)
        temp = np.where(draw_o<=prob_o_rescaled)[1]
        if temp.size == 0:
             option = np.argmax(prob_o)
        else:
             option = np.amin(np.where(draw_o<=prob_o_rescaled)[1])
             
        return int(option)
    
    def select_termination(self, state, option):
        state = BatchBW.encode_state(self, state)
        state = torch.FloatTensor(state.reshape(1,-1)).to(device)         
        self.pi_b[option].eval()
        # Termination
        prob_b = self.pi_b[option](state).cpu().data.numpy()
        prob_b_rescaled = np.divide(prob_b,np.amin(prob_b)+0.01)
        for i in range(1,prob_b_rescaled.shape[1]):
            prob_b_rescaled[0,i]=prob_b_rescaled[0,i]+prob_b_rescaled[0,i-1]
        draw_b = np.divide(np.random.rand(), np.amin(prob_b)+0.01)
        temp = np.where(draw_b<=prob_b_rescaled)[1]
        if temp.size == 0:
            b = np.argmax(prob_b)
        else:
            b = np.amin(np.where(draw_b<=prob_b_rescaled)[1])
            
        return int(b)  
    
    def encode_state(self, state):
        coordinates = state[0:2]
        psi = state[2]
        psi_encoded = np.zeros(self.encoding_info[0])
        psi_encoded[int(psi)]=1
        coin_dir_encoded = np.zeros(self.encoding_info[1])
        coin_dir = state[3]
        coin_dir_encoded[int(coin_dir)]=1
        current_state_encoded = np.concatenate((coordinates,psi_encoded,coin_dir_encoded))
        
        return current_state_encoded

    def Pi_hi(ot, Pi_hi_parameterization, state):
        Pi_hi = Pi_hi_parameterization(state).cpu()
        o_prob = Pi_hi[0,ot]
        return o_prob

    def Pi_hi_bar(b, ot, ot_past, Pi_hi_parameterization, state, option_space):
        if b == True:
            o_prob_tilde = BatchBW.Pi_hi(ot, Pi_hi_parameterization, state)
        elif ot == ot_past:
            o_prob_tilde =  torch.FloatTensor([1])
        else:
            o_prob_tilde =  torch.FloatTensor([0])
        
        return o_prob_tilde

    def Pi_lo(a, Pi_lo_parameterization, state):
        Pi_lo = Pi_lo_parameterization(state).cpu()
        a_prob = Pi_lo[0,int(a)]
    
        return a_prob

    def Pi_b(b, Pi_b_parameterization, state):
        Pi_b = Pi_b_parameterization(state).cpu()
        if b == True:
            b_prob = Pi_b[0,1]
        else:
            b_prob = Pi_b[0,0]
        return b_prob

    def Pi_combined(ot, ot_past, a, b, Pi_hi_parameterization, Pi_lo_parameterization, Pi_b_parameterization, state, option_space):
        Pi_hi_eval = BatchBW.Pi_hi_bar(b, ot, ot_past, Pi_hi_parameterization, state, option_space).clamp(0.0001,1)
        Pi_lo_eval = BatchBW.Pi_lo(a, Pi_lo_parameterization, state).clamp(0.0001,1)
        Pi_b_eval = BatchBW.Pi_b(b, Pi_b_parameterization, state).clamp(0.0001,1)
        output = Pi_hi_eval*Pi_lo_eval*Pi_b_eval
    
        return output

    def ForwardRecursion(alpha_past, a, Pi_hi_parameterization, Pi_lo_parameterization,
                         Pi_b_parameterization, state, option_space, termination_space):
        # =============================================================================
        #     alpha is the forward message: alpha.shape()= [option_space, termination_space]
        # =============================================================================
        alpha = np.empty((option_space, termination_space))
        for i1 in range(option_space):
            ot = i1
            for i2 in range(termination_space):
                if i2 == 1:
                    bt=True
                else:
                    bt=False
            
                Pi_comb = np.zeros(option_space)
                for ot_past in range(option_space):
                    Pi_comb[ot_past] = BatchBW.Pi_combined(ot, ot_past, a, bt, 
                                                           Pi_hi_parameterization, Pi_lo_parameterization[ot], Pi_b_parameterization[ot_past], 
                                                           state, option_space)
                alpha[ot,i2] = np.dot(alpha_past[:,0],Pi_comb)+np.dot(alpha_past[:,1],Pi_comb)  
        alpha = np.divide(alpha,np.sum(alpha))
            
        return alpha

    def ForwardFirstRecursion(mu, a, Pi_hi_parameterization, Pi_lo_parameterization,
                              Pi_b_parameterization, state, option_space, termination_space):
        # =============================================================================
        #     alpha is the forward message: alpha.shape()=[option_space, termination_space]
        #   mu is the initial distribution over options: mu.shape()=[1,option_space]
        # =============================================================================
        alpha = np.empty((option_space, termination_space))
        for i1 in range(option_space):
            ot = i1
            for i2 in range(termination_space):
                if i2 == 1:
                    bt=True
                else:
                    bt=False
            
                Pi_comb = np.zeros(option_space)
                for ot_past in range(option_space):
                    Pi_comb[ot_past] = BatchBW.Pi_combined(ot, ot_past, a, bt, 
                                                           Pi_hi_parameterization, Pi_lo_parameterization[ot], Pi_b_parameterization[ot_past], 
                                                           state, option_space)
                alpha[ot,i2] = np.dot(mu, Pi_comb[:])    
        alpha = np.divide(alpha, np.sum(alpha))
            
        return alpha

    def BackwardRecursion(beta_next, a, Pi_hi_parameterization, Pi_lo_parameterization,
                          Pi_b_parameterization, state, option_space, termination_space):
        # =============================================================================
        #     beta is the backward message: beta.shape()= [option_space, termination_space]
        # =============================================================================
        beta = np.empty((option_space, termination_space))
        for i1 in range(option_space):
            ot = i1
            for i2 in range(termination_space):
                for i1_next in range(option_space):
                    ot_next = i1_next
                    for i2_next in range(termination_space):
                        if i2_next == 1:
                            b_next=True
                        else:
                            b_next=False
                        beta[i1,i2] = beta[i1,i2] + beta_next[ot_next,i2_next]*BatchBW.Pi_combined(ot_next, ot, a, b_next, 
                                                                                                   Pi_hi_parameterization, Pi_lo_parameterization[ot_next], 
                                                                                                   Pi_b_parameterization[ot], state, option_space)
        beta = np.divide(beta,np.sum(beta))
    
        return beta

    def Alpha(self):
        alpha = np.empty((self.option_dim, self.termination_dim, len(self.TrainingSet)))
        for t in range(len(self.TrainingSet)):
            # print('alpha iter', t+1, '/', len(self.TrainingSet))
            if t ==0:
                state = torch.FloatTensor(self.TrainingSet[t,:].reshape(1,self.state_dim)).to(device)
                action = self.Labels[t]
                alpha[:,:,t] = BatchBW.ForwardFirstRecursion(self.mu, action, self.pi_hi, 
                                                             self.pi_lo, self.pi_b, 
                                                             state, self.option_dim, self.termination_dim)
            else:
                state = torch.FloatTensor(self.TrainingSet[t,:].reshape(1,self.state_dim)).to(device)
                action = self.Labels[t]
                alpha[:,:,t] = BatchBW.ForwardRecursion(alpha[:,:,t-1], action, self.pi_hi, 
                                                        self.pi_lo, self.pi_b, 
                                                        state, self.option_dim, self.termination_dim)
           
        return alpha

    def Beta(self):
        beta = np.empty((self.option_dim, self.termination_dim, len(self.TrainingSet)+1))
        beta[:,:,len(self.TrainingSet)] = np.divide(np.ones((self.option_dim, self.termination_dim)),2*self.option_dim)
    
        for t_raw in range(len(self.TrainingSet)):
            t = len(self.TrainingSet) - (t_raw+1)
            # print('beta iter', t_raw+1, '/', len(self.TrainingSet))
            state = torch.FloatTensor(self.TrainingSet[t,:].reshape(1,self.state_dim)).to(device)
            action = self.Labels[t]
            beta[:,:,t] = BatchBW.BackwardRecursion(beta[:,:,t+1], action, self.pi_hi, 
                                                    self.pi_lo, self.pi_b, state,
                                                    self.option_dim, self.termination_dim)
        
        return beta

    def Smoothing(option_space, termination_space, alpha, beta):
        gamma = np.empty((option_space, termination_space))
        for i1 in range(option_space):
            ot=i1
            for i2 in range(termination_space):
                gamma[ot,i2] = alpha[ot,i2]*beta[ot,i2]     
                
        gamma = np.divide(gamma,np.sum(gamma))
    
        return gamma

    def DoubleSmoothing(beta, alpha, a, Pi_hi_parameterization, Pi_lo_parameterization, 
                    Pi_b_parameterization, state, option_space, termination_space):
        gamma_tilde = np.empty((option_space, termination_space))
        for i1_past in range(option_space):
            ot_past = i1_past
            for i2 in range(termination_space):
                if i2 == 1:
                    b=True
                else:
                    b=False
                for i1 in range(option_space):
                    ot = i1
                    gamma_tilde[ot_past,i2] = gamma_tilde[ot_past,i2] + beta[ot,i2]*BatchBW.Pi_combined(ot, ot_past, a, b, 
                                                                                                        Pi_hi_parameterization, Pi_lo_parameterization[ot], 
                                                                                                        Pi_b_parameterization[ot_past], state, option_space)
                gamma_tilde[ot_past,i2] = gamma_tilde[ot_past,i2]*np.sum(alpha[ot_past,:])
        gamma_tilde = np.divide(gamma_tilde,np.sum(gamma_tilde))
    
        return gamma_tilde

    def Gamma(self, alpha, beta):
        gamma = np.empty((self.option_dim, self.termination_dim, len(self.TrainingSet)))
        for t in range(len(self.TrainingSet)):
            # print('gamma iter', t+1, '/', len(self.TrainingSet))
            gamma[:,:,t]=BatchBW.Smoothing(self.option_dim, self.termination_dim, alpha[:,:,t], beta[:,:,t])
        
        return gamma

    def GammaTilde(self, alpha, beta):
        gamma_tilde = np.zeros((self.option_dim, self.termination_dim, len(self.TrainingSet)))
        for t in range(1,len(self.TrainingSet)):
            # print('gamma tilde iter', t, '/', len(self.TrainingSet)-1)
            state = torch.FloatTensor(self.TrainingSet[t,:].reshape(1,self.state_dim)).to(device)
            action = self.Labels[t]
            gamma_tilde[:,:,t]=BatchBW.DoubleSmoothing(beta[:,:,t], alpha[:,:,t-1], action, 
                                                       self.pi_hi, self.pi_lo, self.pi_b, 
                                                       state, self.option_dim, self.termination_dim)
        return gamma_tilde

# functions M-step

    def GammaTildeReshape(gamma_tilde, option_space):
# =============================================================================
#         Function to reshape Gamma_tilde with the same size of NN_pi_b output
# =============================================================================
        T = gamma_tilde.shape[2]
        gamma_tilde_reshaped_array = np.empty((T-1,2,option_space))
        for i in range(option_space):
            gamma_tilde_reshaped = gamma_tilde[i,:,1:]
            gamma_tilde_reshaped_array[:,:,i] = gamma_tilde_reshaped.reshape(T-1,2)
            
        return gamma_tilde_reshaped_array

    def GammaReshapeActions(T, option_space, action_space, gamma, labels):
# =============================================================================
#         function to reshape gamma with the same size of the NN_pi_lo output
# =============================================================================
        gamma_actions_array = np.empty((T, action_space, option_space))
        for k in range(option_space):
            gamma_reshaped_option = gamma[k,:,:]    
            gamma_reshaped_option = np.sum(gamma_reshaped_option,0)
            gamma_actions = np.empty((int(T),action_space))
            for i in range(T):
                for j in range(action_space):
                    if int(labels[i])==j:
                        gamma_actions[i,j]=gamma_reshaped_option[i]
                    else:
                        gamma_actions[i,j] = 0
            gamma_actions_array[:,:,k] = gamma_actions
            
        return gamma_actions_array
    
    def GammaReshapeOptions(gamma):
# =============================================================================
#         function to reshape gamma with the same size of NN_pi_hi output
# =============================================================================
        gamma_reshaped_options = gamma[:,1,:]
        gamma_reshaped_options = np.transpose(gamma_reshaped_options)
        
        return gamma_reshaped_options


    def Loss(gamma_tilde_reshaped, gamma_reshaped_options, gamma_actions, auxiliary_vector,
                    NN_termination, NN_options, NN_actions, T, TrainingSet):
# =============================================================================
#         Compute batch loss function to minimize
# =============================================================================
        loss = 0
        option_space = len(NN_actions)
        for i in range(option_space):
            pi_b = NN_termination[i](TrainingSet).cpu()
            
            if not (len(pi_b) == len(gamma_tilde_reshaped[:,:,i])):
                pi_b = pi_b[0:len(gamma_tilde_reshaped[:,:,i]),:]
            
            loss = loss -torch.sum(torch.FloatTensor(gamma_tilde_reshaped[:,:,i])*torch.log(pi_b[:].clamp(1e-10,1.0)))/(T)
            pi_lo = NN_actions[i](TrainingSet).cpu()
            loss = loss -torch.sum(torch.FloatTensor(gamma_actions[:,:,i])*torch.log(pi_lo.clamp(1e-10,1.0)))/(T)
            
        pi_hi = NN_options(TrainingSet).cpu()
        loss_options = -torch.sum(torch.FloatTensor(gamma_reshaped_options)*torch.log(pi_hi.clamp(1e-10,1.0)))/(T)
        loss = loss + loss_options
    
        return loss    


    def OptimizeLossBatch(self, gamma_tilde_reshaped, gamma_reshaped_options, gamma_actions, auxiliary_vector):
# =============================================================================
#         optimize loss in mini-batches
# =============================================================================
        loss = 0
        n_batches = np.int(self.TrainingSet.shape[0]/self.batch_size)
                            
        for epoch in range(self.epochs):
            # print("\nStart of epoch %d" % (epoch,))
                
            for n in range(n_batches):
                # print("\n Batch %d" % (n+1,))
                TrainingSet = torch.FloatTensor(self.TrainingSet[n*self.batch_size:(n+1)*self.batch_size,:]).to(device)
                loss = BatchBW.Loss(gamma_tilde_reshaped[n*self.batch_size:(n+1)*self.batch_size,:,:], 
                                    gamma_reshaped_options[n*self.batch_size:(n+1)*self.batch_size,:], 
                                    gamma_actions[n*self.batch_size:(n+1)*self.batch_size,:,:], 
                                    auxiliary_vector[n*self.batch_size:(n+1)*self.batch_size,:],
                                    self.pi_b, self.pi_hi, self.pi_lo, self.batch_size, TrainingSet)
        
                for option in range(0,self.option_dim):
                    self.pi_lo_optimizer[option].zero_grad()
                    self.pi_b_optimizer[option].zero_grad()   
                self.pi_hi_optimizer.zero_grad()
                loss.backward()
                # for name, param in self.pi_hi.named_parameters():
                #     print(name, param.grad.nonzero())
                # for option in range(0,self.option_dim):    
                #     for name, param in self.pi_lo[option].named_parameters():
                #         print(name, param.grad)
                # for option in range(0,self.option_dim):    
                #     for name, param in self.pi_b[option].named_parameters():
                #         print(name, param.grad.nonzero())                                        
                for option in range(0,self.option_dim):
                    self.pi_lo_optimizer[option].step()
                    self.pi_b_optimizer[option].step()
                self.pi_hi_optimizer.step()
                # print('loss:', float(loss))
                
        T = self.TrainingSet.shape[0]
        TrainingSet = torch.FloatTensor(self.TrainingSet[0:T-1,:]).to(device)
        loss = BatchBW.Loss(gamma_tilde_reshaped[0:T-1,:,:], gamma_reshaped_options[0:T-1,:], gamma_actions[0:T-1,:,:], auxiliary_vector[0:T-1,:],
                            self.pi_b, self.pi_hi, self.pi_lo, T-1, TrainingSet)
    
        return loss   
                
    def Baum_Welch(self):
# =============================================================================
#         batch BW for HIL
# =============================================================================
        
        T = self.TrainingSet.shape[0]
            
        alpha = BatchBW.Alpha(self)
        beta = BatchBW.Beta(self)
        gamma = BatchBW.Gamma(self, alpha, beta)
        gamma_tilde = BatchBW.GammaTilde(self, alpha, beta)
    
        # print('Expectation done')
        # print('Starting maximization step')
        
        gamma_tilde_reshaped = BatchBW.GammaTildeReshape(gamma_tilde, self.option_dim)
        gamma_actions = BatchBW.GammaReshapeActions(T, self.option_dim, self.action_space_discrete, gamma, self.Labels)
        gamma_reshaped_options = BatchBW.GammaReshapeOptions(gamma)
        m,n,o = gamma_actions.shape
        auxiliary_vector = np.zeros((m,n))
        for l in range(m):
            for k in range(n):
                if gamma_actions[l,k,0]!=0:
                    auxiliary_vector[l,k] = 1


        loss = BatchBW.OptimizeLossBatch(self, gamma_tilde_reshaped, gamma_reshaped_options, gamma_actions, auxiliary_vector)
        # print('Maximization done, Loss:', float(loss)) #float(loss_options+loss_action+loss_termination))
 
        return loss

    def save(self, filename):
        torch.save(self.pi_hi.state_dict(), filename + "_pi_hi")
        torch.save(self.pi_hi_optimizer.state_dict(), filename + "_pi_hi_optimizer")
        
        for option in range(self.option_dim):
            torch.save(self.pi_lo[option].state_dict(), filename + f"_pi_lo_option_{option}")
            torch.save(self.pi_lo_optimizer[option].state_dict(), filename + f"_pi_lo_optimizer_option_{option}")
            torch.save(self.pi_b[option].state_dict(), filename + f"_pi_b_option_{option}")
            torch.save(self.pi_b_optimizer[option].state_dict(), filename + f"_pi_b_optimizer_option_{option}")      
            
    def load(self, filename):
        self.pi_hi.load_state_dict(torch.load(filename + "_pi_hi"))
        self.pi_hi_optimizer.load_state_dict(torch.load(filename + "_pi_hi_optimizer"))
        
        for option in range(self.option_dim):
            self.pi_lo[option].load_state_dict(torch.load(filename + f"_pi_lo_option_{option}"))
            self.pi_lo_optimizer[option].load_state_dict(torch.load(filename + f"_pi_lo_optimizer_option_{option}"))
            self.pi_b[option].load_state_dict(torch.load(filename + f"_pi_b_option_{option}"))
            self.pi_b_optimizer[option].load_state_dict(torch.load(filename + f"_pi_b_optimizer_option_{option}"))
            
