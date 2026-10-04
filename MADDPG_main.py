# MADDPG_main.py
import json
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import ollama

import config
from database import load_financial_data
from maddpg_models import MaddpgActor, CentralizedCritic

# ==========================================
# LOCAL OLLAMA INTERACTION ENGINE
# ==========================================
class LocalOllamaAnalyzer:
    def __init__(self, model_name="llama3:8b"):
        self.model_name = model_name

    def analyze_event(self, text: str) -> float:
        prompt = f"""
        Analyze the stock market risk impact of this news event. 
        Rate it from -1.0 (highly negative impact) to +1.0 (highly positive).
        Respond ONLY with a valid JSON object containing the key "sentiment" and the float value.
        Text: "{text}"
        """
        try:
            res = ollama.generate(model=self.model_name, prompt=prompt)
            data = json.loads(res['response'].strip())
            return max(-1.0, min(1.0, float(data.get("sentiment", 0.0))))
        except Exception:
            return 0.0

# ==========================================
# MULTI-AGENT PORTFOLIO ENVIRONMENT
# ==========================================
class MultiAgentPortfolioEnv:
    def __init__(self, features_df, returns_df):
        self.features = features_df
        self.returns = returns_df
        self.num_agents = len(config.TICKERS)
        self.reset()

    def reset(self):
        self.current_step = config.LOOKBACK_WINDOW
        return self._get_state()

    def _get_state(self):
        window = self.features.iloc[self.current_step - config.LOOKBACK_WINDOW : self.current_step].values
        # Return a flat copy for each individual agent (homogeneous local observation space)
        state_tensor = torch.tensor(window, dtype=torch.float32).flatten()
        return [state_tensor.clone() for _ in range(self.num_agents)]

    def step(self, raw_agent_actions):
        """
        raw_agent_actions: A list of scalar scores output by each decentralized agent.
        """
        # Centralized Softmax normalization across agents to construct long-only portfolio allocations
        exp_weights = np.exp(raw_agent_actions)
        allocations = exp_weights / np.sum(exp_weights)
        
        # Calculate market step returns
        step_returns = self.returns.iloc[self.current_step].values
        portfolio_return = np.dot(allocations, step_returns)
        
        # Cooperative structural reward model (Global Shared Reward)
        reward = portfolio_return
        rewards = [reward for _ in range(self.num_agents)]
        
        self.current_step += 1
        done = self.current_step >= len(self.features) - 1
        
        # FIXED: Using scalar features.shape[1] logic to prevent PyTorch storage calculation overflow
        next_states = self._get_state() if not done else [
            torch.zeros(self.features.shape[1] * config.LOOKBACK_WINDOW) for _ in range(self.num_agents)
        ]
        return next_states, rewards, done

# ==========================================
# MADDPG TRAINING LOGIC MANAGER
# ==========================================
class MADDPGSystem:
    def __init__(self, state_dim, num_agents):
        self.num_agents = num_agents
        self.state_dim = state_dim
        
        # Lists of Actor Networks and Target Actor Networks (one per asset agent)
        self.actors = [MaddpgActor(state_dim).to(config.DEVICE) for _ in range(num_agents)]
        self.actor_targets = [MaddpgActor(state_dim).to(config.DEVICE) for _ in range(num_agents)]
        
        # Unified layout shapes for the Centralized Critic space
        total_state_dim = state_dim * num_agents
        total_action_dim = num_agents # Each agent contributes 1 action scalar
        
        self.critic = CentralizedCritic(total_state_dim, total_action_dim).to(config.DEVICE)
        self.critic_target = CentralizedCritic(total_state_dim, total_action_dim).to(config.DEVICE)
        
        # Sync target network architectures
        for i in range(num_agents):
            self.actor_targets[i].load_state_dict(self.actors[i].state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())
        
        # Optimizers
        self.actor_opts = [optim.AdamW(act.parameters(), lr=config.LR_ACTOR) for act in self.actors]
        self.critic_opt = optim.AdamW(self.critic.parameters(), lr=config.LR_CRITIC)
        
        self.memory = deque(maxlen=20000)

    def select_actions(self, states, noise_scale=0.05):
        actions = []
        for i, state in enumerate(states):
            state_t = state.to(config.DEVICE).unsqueeze(0)
            self.actors[i].eval()
            with torch.no_grad():
                act_val = self.actors[i](state_t).squeeze(0).cpu().item()
            self.actors[i].train()
            
            # Apply exploratory noise
            act_val = np.clip(act_val + np.random.normal(0, noise_scale), -1.0, 1.0)
            actions.append(act_val)
        return actions

    def train_step(self):
        if len(self.memory) < config.BATCH_SIZE:
            return
            
        batch = random.sample(self.memory, config.BATCH_SIZE)
        
        # Unpack structural experience tracking blocks
        s_list, a_list, r_list, ns_list, d_list = zip(*batch)
        
        # Process and construct consolidated centralized tensor maps
        states = [torch.stack([s[i] for s in s_list]).to(config.DEVICE) for i in range(self.num_agents)]
        actions = torch.tensor(np.array(a_list), dtype=torch.float32).to(config.DEVICE)
        next_states = [torch.stack([ns[i] for ns in ns_list]).to(config.DEVICE) for i in range(self.num_agents)]
        
        # FIXED: rewards is sliced column-wise to shape (64,), dones converted cleanly without slicing
        rewards = torch.tensor(np.array(r_list), dtype=torch.float32)[:, 0].to(config.DEVICE)
        dones = torch.tensor(np.array(d_list), dtype=torch.float32).to(config.DEVICE)
        
        joint_state = torch.cat(states, dim=-1)
        joint_next_state = torch.cat(next_states, dim=-1)

        # ---------------- CENTRALIZED CRITIC UPDATE ----------------
        with torch.no_grad():
            # Gather next actions from all target actors to evaluate next state value
            next_actions = torch.cat([self.actor_targets[i](next_states[i]) for i in range(self.num_agents)], dim=-1)
            target_q = self.critic_target(joint_next_state, next_actions).squeeze(-1)
            # FIXED: Aligned 1D vector addition to prevent non-singleton mismatch crash
            y = rewards + (config.GAMMA * target_q * (1 - dones))
            
        current_q = self.critic(joint_state, actions).squeeze(-1)
        critic_loss = nn.MSELoss()(current_q, y)
        
        self.critic_opt.zero_grad()
        critic_loss.backward()
        self.critic_opt.step()

        # ---------------- DECENTRALIZED ACTORS UPDATE ----------------
        for i in range(self.num_agents):
            curr_actions = []
            for j in range(self.num_agents):
                if j == i:
                    curr_actions.append(self.actors[j](states[j]))
                else:
                    curr_actions.append(actions[:, j].unsqueeze(-1).detach())
                    
            joint_curr_actions = torch.cat(curr_actions, dim=-1)
            actor_loss = -self.critic(joint_state, joint_curr_actions).mean()
            
            self.actor_opts[i].zero_grad()
            actor_loss.backward()
            self.actor_opts[i].step()

        # ---------------- SOFT TARGET POLYAK UPDATES ----------------
        for i in range(self.num_agents):
            for p, tp in zip(self.actors[i].parameters(), self.actor_targets[i].parameters()):
                tp.data.copy_(config.TAU * p.data + (1.0 - config.TAU) * tp.data)
        for p, tp in zip(self.critic.parameters(), self.critic_target.parameters()):
            tp.data.copy_(config.TAU * p.data + (1.0 - config.TAU) * tp.data)

# ==========================================
# SYSTEM RUNNER PIPELINE ENTRY
# ==========================================
if __name__ == "__main__":
    print("Loading SQLite asset databases...")
    fundamentals, asset_returns = load_financial_data()
    
    analyzer = LocalOllamaAnalyzer(model_name="llama3:8b")
    fundamentals['nlp_event_sentiment'] = analyzer.analyze_event(
        "Semiconductor manufacturing company announces breakthrough architecture for next-gen processing units."
    )
    
    env = MultiAgentPortfolioEnv(fundamentals, asset_returns)
    
    # FIXED: Extracting feature shape[1] integer directly to bypass structural tuple geometry error
    state_flat_dim = fundamentals.shape[1] * config.LOOKBACK_WINDOW
    num_assets = len(config.TICKERS)
    
    maddpg = MADDPGSystem(state_dim=state_flat_dim, num_agents=num_assets)
    
    print(f"\nLaunching Foundational MADDPG Framework for {num_assets} asset agents...")
    for ep in range(config.EPISODES):
        states = env.reset()
        episode_reward = 0
        done = False
        
        while not done:
            actions = maddpg.select_actions(states, noise_scale=config.NOISE_SCALE)
            next_states, rewards, done = env.step(actions)
            
            maddpg.memory.append((states, actions, rewards, next_states, done))
            maddpg.train_step()
            
            states = next_states
            # Sum up global rewards from agent 0 index tracking
            episode_reward += rewards[0]
            
        print(f"Episode {ep+1:02d}/{config.EPISODES} | Collective Team Portfolio Return: {episode_reward:+.4%}")
