# ppo_models.py
import torch
import torch.nn as nn
from torch.distributions import Dirichlet
import config

class PPOActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(PPOActorCritic, self).__init__()
        
        # Shared feature extractor layer
        self.shared = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU()
        )
        
        # Actor Head: Outputs continuous positive alpha concentration parameters for the Dirichlet Distribution
        # Softplus ensures alpha values are strictly > 0 (we add 1.0 to avoid small numbers near zero)
        self.actor_head = nn.Sequential(
            nn.Linear(128, action_dim),
            nn.Softplus()
        )
        
        # Critic Head: Estimates the expected state value baseline scalar V(s)
        self.critic_head = nn.Linear(128, 1)

    def forward(self, state):
        features = self.shared(state)
        alpha = self.actor_head(features) + 1.0 
        value = self.critic_head(features)
        return alpha, value

    def get_action_and_value(self, state, action=None):
        """
        Samples a stochastic continuous portfolio allocation matrix 
        and extracts log probabilities for policy gradients.
        """
        alpha, value = self.forward(state)
        dist = Dirichlet(alpha)
        
        if action is None:
            action = dist.sample()
            
        # Compute log probability of the generated allocation under the current policy
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        
        return action, log_prob, entropy, value
