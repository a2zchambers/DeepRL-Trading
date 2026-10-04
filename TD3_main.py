# main_td3.py
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
from td3_models import TD3Actor, TwinCritic

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
# CONTINUOUS PORTFOLIO ENVIRONMENT
# ==========================================
class TD3PortfolioEnv:
    def __init__(self, features_df, returns_df):
        self.features = features_df
        self.returns = returns_df
        self.num_assets = returns_df.shape[1]
        self.reset()

    def reset(self):
        self.current_step = config.LOOKBACK_WINDOW
        return self._get_state()

    def _get_state(self):
        window = self.features.iloc[self.current_step - config.LOOKBACK_WINDOW : self.current_step].values
        return torch.tensor(window, dtype=torch.float32).flatten()

    def step(self, raw_actions):
        # Softmax conversion enforces sum-to-one long portfolio targets
        exp_weights = np.exp(raw_actions)
        allocations = exp_weights / np.sum(exp_weights)
        
        step_returns = self.returns.iloc[self.current_step].values
        portfolio_return = np.dot(allocations, step_returns)
        
        reward = portfolio_return
        self.current_step += 1
        done = self.current_step >= len(self.features) - 1
        
        next_state = self._get_state() if not done else torch.zeros(self.features.shape[1] * config.LOOKBACK_WINDOW)
        return next_state, reward, done

# ==========================================
# CENTRALIZED TD3 AGENT ENGINE DEFINITION
# ==========================================
class TD3Agent:
    def __init__(self, state_dim, action_dim):
        self.action_dim = action_dim
        
        # Networks and Target instantiation on Mac Acceleration device
        self.actor = TD3Actor(state_dim, action_dim).to(config.DEVICE)
        self.actor_target = TD3Actor(state_dim, action_dim).to(config.DEVICE)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_opt = optim.AdamW(self.actor.parameters(), lr=config.LR_ACTOR)
        
        self.critic = TwinCritic(state_dim, action_dim).to(config.DEVICE)
        self.critic_target = TwinCritic(state_dim, action_dim).to(config.DEVICE)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_opt = optim.AdamW(self.critic.parameters(), lr=config.LR_CRITIC)
        
        self.memory = deque(maxlen=20000)
        self.total_train_steps = 0
        
        # TD3 Hyperparameters
        self.policy_noise = 0.2
        self.noise_clip = 0.5
        self.policy_delay = 2  # Delay actor updates: optimize policy every N critic updates

    def select_action(self, state, noise_scale=0.1):
        state = state.to(config.DEVICE).unsqueeze(0)
        self.actor.eval()
        with torch.no_grad():
            action = self.actor(state).squeeze(0).cpu().numpy()
        self.actor.train()
        
        # Add exploration noise appropriate for continuous allocation targets
        noise = np.random.normal(0, noise_scale, size=self.action_dim)
        return np.clip(action + noise, -1.0, 1.0)

    def train_step(self):
        if len(self.memory) < config.BATCH_SIZE:
            return
            
        self.total_train_steps += 1
        batch = random.sample(self.memory, config.BATCH_SIZE)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        states = torch.stack(states).to(config.DEVICE)
        actions = torch.tensor(np.array(actions), dtype=torch.float32).to(config.DEVICE)
        rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1).to(config.DEVICE)
        next_states = torch.stack(next_states).to(config.DEVICE)
        dones = torch.tensor(dones, dtype=torch.float32).unsqueeze(1).to(config.DEVICE)

        # ---------------- CRITIC UPDATE ----------------
        with torch.no_grad():
            # 1. Predict target action from the target actor network
            next_actions = self.actor_target(next_states)
            
            # 2. TARGET ACTION SMOOTHING NOISE: Apply small random perturbations
            noise = (torch.randn_like(next_actions) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
            smoothed_next_actions = (next_actions + noise).clamp(-1.0, 1.0)
            
            # 3. Take the minimum of twin target Q values to mitigate optimistic value tracking errors
            target_q1, target_q2 = self.critic_target(next_states, smoothed_next_actions)
            target_q = torch.min(target_q1, target_q2)
            y = rewards + (config.GAMMA * target_q * (1 - dones))
            
        curr_q1, curr_q2 = self.critic(states, actions)
        critic_loss = nn.MSELoss()(curr_q1, y) + nn.MSELoss()(curr_q2, y)
        
        self.critic_opt.zero_grad()
        critic_loss.backward()
        self.critic_opt.step()

        # ---------------- DELAYED ACTOR POLICY UPDATE ----------------
        if self.total_train_steps % self.policy_delay == 0:
            # Optimize actor policy via deterministic policy gradients using only Q1 head evaluations
            actor_loss = -self.critic.Q1(states, self.actor(states)).mean()
            
            self.actor_opt.zero_grad()
            actor_loss.backward()
            self.actor_opt.step()

            # Polyak updates for target networks parameter blocks
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(config.TAU * param.data + (1.0 - config.TAU) * target_param.data)
            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(config.TAU * param.data + (1.0 - config.TAU) * target_param.data)

# ==========================================
# MAIN ENTRY SYSTEM PIPELINE
# ==========================================
if __name__ == "__main__":
    print("Loading fundamental and pricing tables via SQLite connectors...")
    fundamentals, asset_returns = load_financial_data()
    
    analyzer = LocalOllamaAnalyzer(model_name="llama3:8b")
    fundamentals['nlp_event_sentiment'] = analyzer.analyze_event(
        "Company rolls out high margin SaaS structural upgrade across their global enterprise portfolio channels."
    )
    
    env = TD3PortfolioEnv(fundamentals, asset_returns)
    
    # Clean scalar product extraction avoiding tuple configuration bugs
    state_flat_dim = fundamentals.shape[1] * config.LOOKBACK_WINDOW
    num_assets = asset_returns.shape[1]
    
    agent = TD3Agent(state_dim=state_flat_dim, action_dim=num_assets)
    
    print(f"\nLaunching Continuous TD3 Optimization Pipeline across {config.EPISODES} runs...")
    for episode in range(config.EPISODES):
        state = env.reset()
        episode_reward = 0.0
        done = False
        
        while not done:
            action = agent.select_action(state, noise_scale=config.NOISE_SCALE)
            next_state, reward, done = env.step(action)
            
            agent.memory.append((state, action, reward, next_state, done))
            agent.train_step()
            
            state = next_state
            episode_reward += reward
            
        print(f"Episode {episode+1:02d}/{config.EPISODES} | Strategy TD3 Accumulation Return: {episode_reward:+.4%}")
