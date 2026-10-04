# MAPPO_main.py
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import ollama

import config
from database import load_financial_data
from mappo_models import MappoActor, CentralizedMappoCritic

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
# MULTI-AGENT STOCHASTIC ENVIRONMENT
# ==========================================
class MultiAgentStochasticEnv:
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
        state_tensor = torch.tensor(window, dtype=torch.float32).flatten()
        return [state_tensor.clone() for _ in range(self.num_agents)]

    def step(self, raw_agent_actions):
        """
        raw_agent_actions: A list of continuous scales generated bounded inside.
        """
        actions_array = np.array(raw_agent_actions).flatten()
        
        # Enforce sum-to-one fully invested constraint cleanly via cross-agent normalization
        sum_actions = np.sum(actions_array)
        allocations = actions_array / sum_actions if sum_actions > 0 else np.ones(self.num_agents) / self.num_agents
        
        # Calculate joint portfolio return performance step
        step_returns = self.returns.iloc[self.current_step].values
        portfolio_return = np.dot(allocations, step_returns)
        
        # Cooperative global shared reward signal
        reward = portfolio_return
        rewards = [reward for _ in range(self.num_agents)]
        
        self.current_step += 1
        done = self.current_step >= len(self.features) - 1
        
        next_states = self._get_state() if not done else [
            torch.zeros(self.features.shape[1] * config.LOOKBACK_WINDOW) for _ in range(self.num_agents)
        ]
        return next_states, rewards, done

# ==========================================
# MAIN EXECUTION RUNNER
# ==========================================
if __name__ == "__main__":
    print("Loading SQLite relational asset databases...")
    fundamentals, asset_returns = load_financial_data()
    
    analyzer = LocalOllamaAnalyzer(model_name="llama3:8b")
    fundamentals['nlp_event_sentiment'] = analyzer.analyze_event(
        "Regulators issue highly protective infrastructure rules bolstering domestic semiconductor capacity."
    )
    
    env = MultiAgentStochasticEnv(fundamentals, asset_returns)
    
    # FIXED: Extracting feature shape integer directly to bypass structural tuple geometry error
    state_flat_dim = fundamentals.shape[1] * config.LOOKBACK_WINDOW
    num_assets = len(config.TICKERS)
    
    # Initialize MAPPO elements
    actors = [MappoActor(state_flat_dim).to(config.DEVICE) for _ in range(num_assets)]
    critic = CentralizedMappoCritic(state_flat_dim * num_assets).to(config.DEVICE)
    
    # Centralized parameter optimization group
    all_params = list(critic.parameters())
    for act in actors:
        all_params += list(act.parameters())
    optimizer = optim.AdamW(all_params, lr=config.LR_ACTOR, eps=1e-5)
    
    # MAPPO Hyperparameters
    PPO_EPOCHS = 5
    CLIP_EPSILON = 0.2
    CRITIC_COEFF = 0.5
    ENTROPY_COEFF = 0.01

    print(f"\nLaunching Multi-Agent MAPPO Stochastic Processing Loop...")
    for episode in range(config.EPISODES):
        states = env.reset()
        done = False
        
        # Trajectory tracking stores
        states_b = [[] for _ in range(num_assets)]
        actions_b = [[] for _ in range(num_assets)]
        log_probs_b = [[] for _ in range(num_assets)]
        joint_states_b = []
        rewards_b = []
        values_b = []
        dones_b = []
        
        episode_reward = 0.0
        
        # 1. Trajectory Rollout Phase
        for act in actors: act.eval()
        critic.eval()
        
        while not done:
            joint_state = torch.cat(states, dim=-1).to(config.DEVICE)
            with torch.no_grad():
                val_pred = critic(joint_state.unsqueeze(0)).squeeze(0)
                
            step_actions = []
            
            for i in range(num_assets):
                state_i = states[i].to(config.DEVICE)
                with torch.no_grad():
                    act_t, lp_t, _ = actors[i].get_action_and_log_prob(state_i.unsqueeze(0))
                
                step_actions.append(act_t.cpu().item())
                
                states_b[i].append(states[i])
                actions_b[i].append(act_t.squeeze(0).cpu())
                log_probs_b[i].append(lp_t.squeeze(0).cpu())
                
            next_states, rewards, done = env.step(step_actions)
            
            joint_states_b.append(joint_state.cpu())
            rewards_b.append(rewards)
            values_b.append(val_pred.item())
            dones_b.append(done)
            
            states = next_states
            # FIXED: Slice reward array from the first agent position to eliminate list accumulation error
            episode_reward += rewards[0]
            
        # 2. Compute Expected Target Baselines
        returns_b = []
        discounted_sum = 0
        for r, d in zip(reversed(rewards_b), reversed(dones_b)):
            if d: discounted_sum = 0
            # rewards is structural shape (3,), extract index 0 shared reward component
            discounted_sum = r[0] + config.GAMMA * discounted_sum
            returns_b.insert(0, discounted_sum)
            
        # Convert rollout storage to PyTorch tensors
        joint_states_t = torch.stack(joint_states_b).to(config.DEVICE)
        returns_t = torch.tensor(returns_b, dtype=torch.float32).unsqueeze(1).to(config.DEVICE)
        old_values_t = torch.tensor(values_b, dtype=torch.float32).unsqueeze(1).to(config.DEVICE)
        
        advantages_t = returns_t - old_values_t
        advantages_t = (advantages_t - advantages_t.mean()) / (advantages_t.std() + 1e-8)
        
        # 3. Synchronous MAPPO Optimization Phase
        for act in actors: act.train()
        critic.train()
        
        for _ in range(PPO_EPOCHS):
            new_values = critic(joint_states_t)
            critic_loss = nn.MSELoss()(new_values, returns_t)
            
            total_actor_loss = 0
            total_entropy_loss = 0
            
            for i in range(num_assets):
                states_i_t = torch.stack(states_b[i]).to(config.DEVICE)
                actions_i_t = torch.stack(actions_b[i]).to(config.DEVICE)
                old_lps_i_t = torch.stack(log_probs_b[i]).to(config.DEVICE)
                
                _, new_lps, entropy = actors[i].get_action_and_log_prob(states_i_t, actions_i_t)
                
                ratios = torch.exp(new_lps - old_lps_i_t).unsqueeze(1)
                surr1 = ratios * advantages_t
                surr2 = torch.clamp(ratios, 1.0 - CLIP_EPSILON, 1.0 + CLIP_EPSILON) * advantages_t
                
                total_actor_loss += -torch.min(surr1, surr2).mean()
                total_entropy_loss += entropy.mean()
                
            loss = total_actor_loss + CRITIC_COEFF * critic_loss - ENTROPY_COEFF * total_entropy_loss
            
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(all_params, max_norm=0.5)
            optimizer.step()
            
        print(f"Episode {episode+1:02d}/{config.EPISODES} | Collective MAPPO Return: {episode_reward:+.4%}")
