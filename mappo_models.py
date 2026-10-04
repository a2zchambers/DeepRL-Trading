# mappo_models.py
import torch
import torch.nn as nn
from torch.distributions import Beta
import config

class MappoActor(nn.Module):
    """
    Decentralized Stochastic Actor network for an individual asset agent.
    Outputs concentration parameters (alpha, beta) to sample an allocation score strictly in.
    """
    def __init__(self, state_dim):
        super(MappoActor, self).__init__()
        self.feature_extractor = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU()
        )
        # Softplus ensures concentration shapes are strictly > 0
        self.alpha_head = nn.Sequential(nn.Linear(64, 1), nn.Softplus())
        self.beta_head = nn.Sequential(nn.Linear(64, 1), nn.Softplus())

    def forward(self, state):
        features = self.feature_extractor(state)
        # Add 1.0 to ensure the distribution remains stable and unimodal
        alpha = self.alpha_head(features) + 1.0
        beta = self.beta_head(features) + 1.0
        return alpha, beta

    def get_action_and_log_prob(self, state, action=None):
        alpha, beta = self.forward(state)
        dist = Beta(alpha, beta)
        
        if action is None:
            action = dist.sample()
            
        log_prob = dist.log_prob(action).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        return action, log_prob, entropy

class CentralizedMappoCritic(nn.Module):
    """
    Centralized Critic network for MAPPO.
    Estimates a unified expected baseline value V(S) given the joint global states of ALL agents.
    """
    def __init__(self, joint_state_dim):
        super(CentralizedMappoCritic, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(joint_state_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)  # Centralized unified value scalar V(S)
        )

    def forward(self, joint_state):
        return self.net(joint_state)
