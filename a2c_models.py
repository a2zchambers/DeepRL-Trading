# a2c_models.py
import torch
import torch.nn as nn
from torch.distributions import Dirichlet
import config

class A2CActorCritic(nn.Module):
    """
    Synchronous Advantage Actor-Critic Network.
    Uses a Shared Backbone with separate Actor and Critic heads.
    """
    def __init__(self, state_dim, action_dim):
        super(A2CActorCritic, self).__init__()
        
        # Shared feature representation layer across heads
        self.shared = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU()
        )
        
        # Actor Head: Outputs Dirichlet concentration alpha parameters (>0)
        self.actor_head = nn.Sequential(
            nn.Linear(128, action_dim),
            nn.Softplus()
        )
        
        # Critic Head: Outputs a centralized expected state scalar value V(S)
        self.critic_head = nn.Linear(128, 1)

    def forward(self, state):
        features = self.shared(state)
        # Add 1.0 to guarantee numerical stability in the Dirichlet distribution boundaries
        alpha = self.actor_head(features) + 1.0
        value = self.critic_head(features)
        return alpha, value

    def evaluate_trajectory(self, state, action=None):
        alpha, value = self.forward(state)
        dist = Dirichlet(alpha)
        
        if action is None:
            action = dist.sample()
            
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        
        return action, log_prob, entropy, value
