# A2C_main.py
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import ollama

import config
from database import load_financial_data
from a2c_models import A2CActorCritic

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
# PORTFOLIO ALLOCATION ENVIRONMENT
# ==========================================
class A2CPortfolioEnv:
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
        allocations: Dirichlet action array which natively sums to exactly 1.0
        """
        step_returns = self.returns.iloc[self.current_step].values
        portfolio_return = np.dot(allocations, step_returns)
        
        reward = portfolio_return
        
        self.current_step += 1
        done = self.current_step >= len(self.features) - 1
        
        next_state = self._get_state() if not done else torch.zeros(self.features.shape[1] * config.LOOKBACK_WINDOW)
        return next_state, reward, done

# ==========================================
# MAIN SYNCHRONOUS A2C ENGINE EXECUTION
# ==========================================
if __name__ == "__main__":
    print("Loading fundamental and pricing tables via SQLite connectors...")
    fundamentals, asset_returns = load_financial_data()
    
    analyzer = LocalOllamaAnalyzer(model_name="llama3:8b")
    fundamentals['nlp_event_sentiment'] = analyzer.analyze_event(
        "Industry-wide semiconductor supply constraints ease as production yields hit record highs."
    )
    
    env = A2CPortfolioEnv(fundamentals, asset_returns)
    state_flat_dim = fundamentals.shape[1] * config.LOOKBACK_WINDOW
    num_assets = asset_returns.shape[1]
    
    # Initialize A2C Model on your Mac accelerated hardware device
    model = A2CActorCritic(state_flat_dim, num_assets).to(config.DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=config.LR_ACTOR, eps=1e-5)
    
    # Structural Loss Coefficients
    CRITIC_COEFF = 0.5
    ENTROPY_COEFF = 0.01

    print(f"\nLaunching Synchronous On-Policy A2C Processing Loop...")
    for episode in range(config.EPISODES):
        state = env.reset()
        done = False
        
        # On-Policy Episode Buffers
        states_b, actions_b, log_probs_b, rewards_b, values_b, entropies_b = [], [], [], [], [], []
        episode_reward = 0.0
        
        # 1. Collect Trajectory Rollout
        # FIXED: Model set to train mode and torch.no_grad removed to track gradients natively
        model.train() 
        while not done:
            state_t = state.to(config.DEVICE)
            
            # Forward pass tracks gradients directly during rollout tracking
            action_t, log_prob_t, entropy_t, value_t = model.evaluate_trajectory(state_t)
                
            action = action_t.detach().cpu().numpy().flatten() # Detach here only for environment step interaction
            next_state, reward, done = env.step(action)
            
            states_b.append(state_t)
            actions_b.append(action_t)
            log_probs_b.append(log_prob_t)
            rewards_b.append(reward)
            values_b.append(value_t) # Graph connection kept intact
            entropies_b.append(entropy_t)
            
            state = next_state
            episode_reward += reward
            
        # 2. Process Synchronous On-Policy Updates
        returns_b = []
        discounted_sum = 0.0
        for r in reversed(rewards_b):
            discounted_sum = r + config.GAMMA * discounted_sum
            returns_b.insert(0, discounted_sum)
            
        # Convert lists to device-backed tensors
        returns_t = torch.tensor(returns_b, dtype=torch.float32).unsqueeze(1).to(config.DEVICE)
        log_probs_t = torch.stack(log_probs_b).unsqueeze(1)
        
        # FIXED: Squeezed multi-agent value outputs to prevent broadcasting mismatch warning
        values_t = torch.stack(values_b).squeeze(-1)
        entropies_t = torch.stack(entropies_b)
        
        # Advantage calculation: Actual Returns - Estimated Values Baselines
        # Detached from target graph to isolate policy tracking adjustments
        advantages_t = returns_t - values_t.detach()
        
        # 3. Compute Synchronous Network Gradients
        # Policy Loss: maximize advantage-weighted log-probabilities
        actor_loss = -(log_probs_t * advantages_t).mean()
        
        # Value Baseline Loss: minimize Mean Squared Error against Monte Carlo targets
        critic_loss = nn.MSELoss()(values_t, returns_t)
        
        # Total combined on-policy loss optimization layout
        total_loss = actor_loss + CRITIC_COEFF * critic_loss - ENTROPY_COEFF * entropies_t.mean()
        
        optimizer.zero_grad()
        total_loss.backward() # FIXED: Tracks backward trace through functional graphs cleanly
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
        optimizer.step()
        
        print(f"Episode {episode+1:02d}/{config.EPISODES} | Synchronous A2C Return: {episode_reward:+.4%}")
