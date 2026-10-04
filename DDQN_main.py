# main.py
import json
import numpy as np
import pandas as pd
import torch
import ollama

import config
from database import load_financial_data
from models import DDPGAgent

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
            return 0.0 # Neutral fallback for missing local endpoints or parsing issues

# ==========================================
# PORTFOLIO ALLOCATION ENVIRONMENT
# ==========================================
class PortfolioEnv:
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
        return torch.tensor(window, dtype=torch.float32)

    def step(self, raw_actions):
        # Scale outputs via Softmax to guarantee long-only allocations summing to 1.0
        exp_weights = np.exp(raw_actions)
        allocations = exp_weights / np.sum(exp_weights)
        
        # Compute portfolio return for the step
        step_returns = self.returns.iloc[self.current_step].values
        portfolio_return = np.dot(allocations, step_returns)
        
        # Reward design targeting structural absolute returns
        reward = portfolio_return
        
        self.current_step += 1
        done = self.current_step >= len(self.features) - 1
        
        next_state = self._get_state() if not done else torch.zeros((config.LOOKBACK_WINDOW, self.features.shape[1]))
        return next_state, reward, done

# ==========================================
# MAIN EXECUTION RUNNER
# ==========================================
if __name__ == "__main__":
    print("Loading fundamental and pricing feeds from local SQLite databases...")
    fundamentals, asset_returns = load_financial_data()
    
    # --- Example Local Ollama News Processing Loop ---
    # In practice, map your historical narrative calendar alongside tracking dates.
    analyzer = LocalOllamaAnalyzer(model_name="llama3:8b")
    print("Processing local macro/industry events via Ollama pipeline...")
    
    sample_news = "Regulators open up antitrust lawsuit against top semiconductor manufacturers."
    event_sentiment_score = analyzer.analyze_event(sample_news)
    
    # Broadcast NLP scores into environment features dataframe
    fundamentals['nlp_event_sentiment'] = event_sentiment_score
    
    # Initialize structural tracking components
    env = PortfolioEnv(fundamentals, asset_returns)
    
    # Flatten lookback windows across dimensions for input space calculation
    state_flat_dim = fundamentals.shape[1] * config.LOOKBACK_WINDOW
    action_dim = asset_returns.shape[1]
    
    agent = DDPGAgent(state_dim=state_flat_dim, action_dim=action_dim)
    
    print(f"\nStarting optimization loop across {config.EPISODES} training runs...")
    for episode in range(config.EPISODES):
        state_tensor = env.reset().flatten()
        episode_reward = 0
        done = False
        
        while not done:
            action = agent.select_action(state_tensor, noise_scale=config.NOISE_SCALE)
            next_state_tensor, reward, done = env.step(action)
            next_state_flat = next_state_tensor.flatten()
            
            # Store transition state steps
            agent.memory.append((state_tensor, action, reward, next_state_flat, done))
            agent.train_step()
            
            state_tensor = next_state_flat
            episode_reward += reward
            
        print(f"Episode {episode+1:02d}/{config.EPISODES} | Strategy Accumulated Return: {episode_reward:+.4%}")
