# sac_models.py
import torch
import torch.nn as nn
from torch.distributions import Normal
import config

class SacActor(nn.Module):
    """
    Decentralized Stochastic Actor network for SAC.
    Outputs mean and log_std vectors to sample an allocation score.
    """
    def __init__(self, state_dim, action_dim=1):
        super(SacActor, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU()
        )
        self.mu_layer = nn.Linear(64, action_dim)
        self.log_std_layer = nn.Linear(64, action_dim)
        
        # Enforce reasonable bounds for log standard deviations
        self.log_std_min = -20
        self.log_std_max = 2

    def forward(self, state):
        features = self.net(state)
        mu = self.mu_layer(features)
        log_std = self.log_std_layer(features)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        return mu, log_std

    def sample_action(self, state, reparameterize=True):
        mu, log_std = self.forward(state)
        std = torch.exp(log_std)
        dist = Normal(mu, std)
        
        if reparameterize:
            x_t = dist.rsample()  # Reparameterization trick (allows backprop through sampling)
        else:
            x_t = dist.sample()
            
        action = torch.tanh(x_t)  # Squashing to [-1, 1] range
        
        # Enforce correction to log-likelihood calculation due to Tanh squashing transformation
        log_prob = dist.log_prob(x_t) - torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        
        return action, log_prob

class CentralizedSacCritic(nn.Module):
    """
    Twin Critic networks for Centralized SAC training to solve overestimation bias.
    """
    def __init__(self, total_state_dim, total_action_dim):
        super(CentralizedSacCritic, self).__init__()
        # Twin Q1 Structure
        self.q1 = nn.Sequential(
            nn.Linear(total_state_dim + total_action_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        # Twin Q2 Structure
        self.q2 = nn.Sequential(
            nn.Linear(total_state_dim + total_action_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, joint_state, joint_action):
        x = torch.cat([joint_state, joint_action], dim=-1)
        return self.q1(x), self.q2(x)
