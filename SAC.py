import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from Buffer import ReplayBuffer
from models import SoftmaxHierarchicalActor
from models import Critic_flat_discrete

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    
class SAC(object):
    def __init__(self, state_dim, action_dim, encoding_info = None, l_rate=3e-4, discount=0.99, tau=0.005, alpha=0.2, critic_freq=2):

        self.actor = SoftmaxHierarchicalActor.NN_PI_LO(state_dim, action_dim).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=3e-4)
        
        self.critic = Critic_flat_discrete(state_dim, action_dim).to(device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=3e-4)
        
        self.target_entropy = -torch.FloatTensor([1]).to(device)
        self.log_alpha = torch.zeros(1, requires_grad=True).to(device)
        self.alpha_optim = torch.optim.Adam([self.log_alpha], lr = l_rate)     

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.encoding_info = encoding_info
        
        self.discount = discount
        self.tau = tau
        self.alpha = alpha
        self.critic_freq = critic_freq
        
        self.Buffer = ReplayBuffer(state_dim, 1)

        self.total_it = 0

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
        state = SAC.encode_state(self, state)
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

    def train(self, batch_size=256):
        self.total_it += 1

		# Sample replay buffer 
        state, action, next_state, reward, cost, not_done = self.Buffer.sample(batch_size)

        with torch.no_grad():
            # Select action according to policy and add clipped noise			
            next_action, _ = self.actor.sample(next_state)
            log_pi_next_state, _ = self.actor.sample_log(next_state, torch.LongTensor(next_action.detach().numpy().flatten()))
            next_action_prob = self.actor(next_state)

            # Compute the target Q value
            target_Q1, target_Q2 = self.critic_target(next_state)
            target_Q = next_action_prob*(torch.min(target_Q1, target_Q2) - self.alpha*log_pi_next_state)
            target_Q = reward-cost + not_done * self.discount * target_Q.sum(dim=1).unsqueeze(-1)

		# Get current Q estimates
        Q1, Q2 = self.critic(state)
        current_Q1 = Q1.gather(1, action.detach().long()) 
        current_Q2 = Q2.gather(1, action.detach().long()) 

		# Compute critic loss
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

		# Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        Q1, Q2 = self.critic(state)
        minQ = torch.min(Q1,Q2)
        action, _ = self.actor.sample(state)
        log_pi_state, _ = self.actor.sample_log(state, torch.LongTensor(action.detach().numpy().flatten()))
        action_prob = self.actor(state)
        
        actor_loss = ((action_prob*(self.alpha*log_pi_state-minQ)).sum(dim=1)).mean()
			
        # Optimize the actor 
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        alpha_loss = -(self.log_alpha * (torch.sum(log_pi_state * action_prob, dim=1) + self.target_entropy).detach()).mean()
        
        self.alpha_optim.zero_grad()
        alpha_loss.backward()
        self.alpha_optim.step()

        self.alpha = self.log_alpha.exp()

        # Update the frozen target models
        if self.total_it % self.critic_freq == 0:
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def save_actor(self, filename):
        option = 0
        torch.save(self.actor.state_dict(), filename + f"_pi_lo_option_{option}")
        torch.save(self.actor_optimizer.state_dict(), filename + f"_pi_lo_optimizer_option_{option}")
        
    def load_actor(self, filename):
        option = 0
        self.actor.load_state_dict(torch.load(filename + f"_pi_lo_option_{option}"))
        self.actor_optimizer.load_state_dict(torch.load(filename + f"_pi_lo_optimizer_option_{option}"))
        
    def save_critic(self, filename):
        torch.save(self.critic.state_dict(), filename + "_critic")
        torch.save(self.critic_optimizer.state_dict(), filename + "_critic_optimizer")
    
    def load_critic(self, filename):
        self.critic.load_state_dict(torch.load(filename + "_critic"))
        self.critic_optimizer.load_state_dict(torch.load(filename + "_critic_optimizer"))
        self.critic_target = copy.deepcopy(self.critic)
		