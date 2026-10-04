# main_ppo.py
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import ollama

import config
from database import load_financial_data
from ppo_models import PPOActorCritic

# ==========================================
# LOCAL OLLAMA INTERACTION ENGINE
# ==========================================
class LocalOllamaAnalyzer:
    def __init__(self, model_name="llama3.1:72b"):
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
# PORTFOLIO STOCHASTIC ENVIRONMENT
# ==========================================
class PPOPortfolioEnv:
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

    def step(self, allocations):
        """
        allocations: Pre-normalized continuous weights from Dirichlet model array (sums to 1.0)
        """
        # Calculate continuous cross-sectional step returns
        step_returns = self.returns.iloc[self.current_step].values
        portfolio_return = np.dot(allocations, step_returns)
        
        # Absolute structural returns reward signal
        reward = portfolio_return
        
        self.current_step += 1
        done = self.current_step >= len(self.features) - 1
        
        next_state = self._get_state() if not done else torch.zeros(self.features.shape[1] * config.LOOKBACK_WINDOW)
        return next_state, reward, done

# ==========================================
# PPO POLICY OPTIMIZATION RUNNER
# ==========================================
def train_ppo():
    print("Loading fundamental and pricing databases...")
    fundamentals, asset_returns = load_financial_data()
    
    # Process local text elements
    analyzer = LocalOllamaAnalyzer(model_name="llama3:8b")
    sample_news = "Regulators open up antitrust lawsuit against top semiconductor manufacturers."
    fundamentals['nlp_event_sentiment'] = analyzer.analyze_event(sample_news)
    
    env = PPOPortfolioEnv(fundamentals, asset_returns)
    state_dim = fundamentals.shape[1] * config.LOOKBACK_WINDOW
    action_dim = asset_returns.shape[1]
    
    # Model and Parameter Optimization Initialization
    model = PPOActorCritic(state_dim, action_dim).to(config.DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=config.LR_ACTOR, eps=1e-5)
    
    # PPO Hyperparameters
    PPO_EPOCHS = 10
    CLIP_EPSILON = 0.2
    CRITIC_COEFF = 0.5
    ENTROPY_COEFF = 0.01
    
    print(f"\nStarting Stochastic PPO Loop across {config.EPISODES} runs...")
    
    for episode in range(config.EPISODES):
        state = env.reset()
        done = False
        
        # Experience storage lists for current episode trajectory
        states_b, actions_b, log_probs_b, rewards_b, values_b, dones_b = [], [], [], [], [], []
        episode_reward = 0
        
        # 1. Trajectory Data Collection Phase
        model.eval()
        while not done:
            state_t = state.to(config.DEVICE)
            with torch.no_grad():
                action_t, log_prob_t, _, value_t = model.get_action_and_value(state_t)
                
            action = action_t.cpu().numpy()
            next_state, reward, done = env.step(action)
            
            # Record structural step values
            states_b.append(state)
            actions_b.append(action_t.cpu())
            log_probs_b.append(log_prob_t.cpu())
            rewards_b.append(reward)
            values_b.append(value_t.item())
            dones_b.append(done)
            
            state = next_state
            episode_reward += reward

        # 2. Compute Expected Returns & Advantages (Generalized Advantage Estimation)
        # Compute Monte Carlo value tracking baselines
        returns_b = []
        discounted_sum = 0
        for r, d in zip(reversed(rewards_b), reversed(dones_b)):
            if d: discounted_sum = 0
            discounted_sum = r + config.GAMMA * discounted_sum
            returns_b.insert(0, discounted_sum)
            
        # Convert list tracks securely to tensor arrays
        states_t = torch.stack(states_b).to(config.DEVICE)
        actions_t = torch.stack(actions_b).to(config.DEVICE)
        old_log_probs_t = torch.stack(log_probs_b).to(config.DEVICE)
        returns_t = torch.tensor(returns_b, dtype=torch.float32).unsqueeze(1).to(config.DEVICE)
        old_values_t = torch.tensor(values_b, dtype=torch.float32).unsqueeze(1).to(config.DEVICE)
        
        # Compute policy choices base metrics
        advantages_t = returns_t - old_values_t
        # Normalize structural advantage arrays to stabilize gradients
        advantages_t = (advantages_t - advantages_t.mean()) / (advantages_t.std() + 1e-8)

        # 3. PPO Optimization Updates
        model.train()
        for _ in range(PPO_EPOCHS):
            _, new_log_probs, entropy, new_values = model.get_action_and_value(states_t, actions_t)
            
            # Compute tracking ratios for probability changes
            ratios = torch.exp(new_log_probs - old_log_probs_t).unsqueeze(1)
            
            # Policy Surrogates Objective functions
            surr1 = ratios * advantages_t
            surr2 = torch.clamp(ratios, 1.0 - CLIP_EPSILON, 1.0 + CLIP_EPSILON) * advantages_t
            actor_loss = -torch.min(surr1, surr2).mean()
            
            # Critic MSE Loss calculation
            critic_loss = nn.MSELoss()(new_values, returns_t)
            
            # Total Loss Objective: minimize Actor & Critic error, maximize Entropy exploration
            total_loss = actor_loss + CRITIC_COEFF * critic_loss - ENTROPY_COEFF * entropy.mean()
            
            optimizer.zero_grad()
            total_loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            optimizer.step()
            
        print(f"Episode {episode+1:02d}/{config.EPISODES} | Strategy Stochastic Return: {episode_reward:+.4%}")

if __name__ == "__main__":
    train_ppo()
