# models.py
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import config

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
            nn.Tanh()
        )
    def forward(self, state):
        return self.net(state)

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        self.state_layer = nn.Sequential(nn.Linear(state_dim, 128), nn.ReLU())
        self.action_layer = nn.Sequential(nn.Linear(action_dim, 128), nn.ReLU())
        
        self.joint_layer = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
    def forward(self, state, action):
        s_out = self.state_layer(state)
        a_out = self.action_layer(action)
        x = torch.cat([s_out, a_out], dim=-1)
        return self.joint_layer(x)

class DDPGAgent:
    def __init__(self, state_dim, action_dim):
        self.action_dim = action_dim
        
        # Instantiate networks on the designated Mac device profile
        self.actor = Actor(state_dim, action_dim).to(config.DEVICE)
        self.actor_target = Actor(state_dim, action_dim).to(config.DEVICE)
        self.critic = Critic(state_dim, action_dim).to(config.DEVICE)
        self.critic_target = Critic(state_dim, action_dim).to(config.DEVICE)
        
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())
        
        self.actor_opt = optim.AdamW(self.actor.parameters(), lr=config.LR_ACTOR)
        self.critic_opt = optim.AdamW(self.critic.parameters(), lr=config.LR_CRITIC)
        
        self.memory = deque(maxlen=20000)

    def select_action(self, state, noise_scale=0.05):
        # Ensure state tensor is on the matching acceleration device
        state = state.to(config.DEVICE).unsqueeze(0)
        self.actor.eval()
        with torch.no_grad():
            action = self.actor(state).squeeze(0).cpu().numpy()
        self.actor.train()
        
        noise = np.random.normal(0, noise_scale, size=self.action_dim)
        return np.clip(action + noise, -1.0, 1.0)

    def train_step(self):
        if len(self.memory) < config.BATCH_SIZE: 
            return
        
        batch = random.sample(self.memory, config.BATCH_SIZE)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        # Format tensors and cast onto Mac MPS device framework
        states = torch.stack(states).to(config.DEVICE)
        actions = torch.tensor(np.array(actions), dtype=torch.float32).to(config.DEVICE)
        rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1).to(config.DEVICE)
        next_states = torch.stack(next_states).to(config.DEVICE)
        dones = torch.tensor(dones, dtype=torch.float32).unsqueeze(1).to(config.DEVICE)

        # ---------------- CRITIC UPDATE ----------------
        with torch.no_grad():
            next_actions = self.actor_target(next_states)
            target_q = self.critic_target(next_states, next_actions)
            y = rewards + (config.GAMMA * target_q * (1 - dones))
            
        current_q = self.critic(states, actions)
        critic_loss = nn.MSELoss()(current_q, y)
        
        self.critic_opt.zero_grad()
        critic_loss.backward()
        self.critic_opt.step()

        # ---------------- ACTOR UPDATE ----------------
        actor_loss = -self.critic(states, self.actor(states)).mean()
        
        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()

        # ---------------- SOFT TARGET PARAMETER POLYAK UPDATES ----------------
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(config.TAU * param.data + (1.0 - config.TAU) * target_param.data)
        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(config.TAU * param.data + (1.0 - config.TAU) * target_param.data)
