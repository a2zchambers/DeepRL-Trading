# td3_models.py
import torch
import torch.nn as nn
import config

class TD3Actor(nn.Module):
    """
    Deterministic Actor network for TD3.
    Outputs continuous unbounded scores mapped between -1.0 and +1.0 via Tanh.
    """
    def __init__(self, state_dim, action_dim):
        super(TD3Actor, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
            nn.Tanh()  # Clamped boundary bounds for stock score weights
        )

    def forward(self, state):
        return self.net(state)

class TwinCritic(nn.Module):
    """
    Twin Critic layout containing Q1 and Q2 networks inside a single module
    to eliminate overestimation errors via lower-bound estimation checking.
    """
    def __init__(self, state_dim, action_dim):
        super(TwinCritic, self).__init__()
        
        # Q1 Architecture
        self.q1_net = nn.Sequential(
            nn.Linear(state_dim + action_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        
        # Q2 Architecture
        self.q2_net = nn.Sequential(
            nn.Linear(state_dim + action_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        return self.q1_net(x), self.q2_net(x)

    def Q1(self, state, action):
        """Helper method used exclusively during Actor policy optimization updates."""
        x = torch.cat([state, action], dim=-1)
        return self.q1_net(x)
