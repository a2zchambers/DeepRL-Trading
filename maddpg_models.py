# maddpg_models.py
import torch
import torch.nn as nn
import numpy as np
import config

class MaddpgActor(nn.Module):
    """
    Decentralized Actor network for an individual asset agent.
    Outputs a single continuous allocation score between -1.0 and +1.0.
    """
    def __init__(self, state_dim):
        super(MaddpgActor, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Tanh()  # Maps directly to allocation sentiment bound
        )
        
    def forward(self, state):
        return self.net(state)

class CentralizedCritic(nn.Module):
    """
    Centralized Critic network.
    Evaluates the joint state and the actions of ALL agents combined.
    """
    def __init__(self, total_state_dim, total_action_dim):
        super(CentralizedCritic, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(total_state_dim + total_action_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)  # Single centralized joint Q-value
        )
        
    def forward(self, joint_state, joint_action):
        x = torch.cat([joint_state, joint_action], dim=-1)
        return self.net(x)
