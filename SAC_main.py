# main_sac.py
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
from sac_models import SacActor, CentralizedSacCritic

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
        state_tensor = torch.tensor(window, dtype=torch.float32).flatten()
        return [state_tensor.clone() for _ in range(self.num_agents)]

    def step(self, raw_agent_actions):
        exp_weights = np.exp(raw_agent_actions)
        allocations = exp_weights / np.sum(exp_weights)
        
        step_returns = self.returns.iloc[self.current_step].values
        portfolio_return = np.dot(allocations, step_returns)
        
        reward = portfolio_return
        rewards = [reward for _ in range(self.num_agents)]
        
        self.current_step += 1
        done = self.current_step >= len(self.features) - 1
        
        next_states = self._get_state() if not done else [
            torch.zeros(self.features.shape[1] * config.LOOKBACK_WINDOW) for _ in range(self.num_agents)
        ]
        return next_states, rewards, done

# ==========================================
# COOPERATIVE MULTI-AGENT SAC SYSTEM
# ==========================================
class MultiAgentSAC:
    def __init__(self, state_dim, num_agents):
        self.num_agents = num_agents
        self.state_dim = state_dim
        
        # Instantiate localized actors
        self.actors = [SacActor(state_dim).to(config.DEVICE) for _ in range(num_agents)]
        self.actor_opts = [optim.AdamW(act.parameters(), lr=config.LR_ACTOR) for act in self.actors]
        
        total_state_dim = state_dim * num_agents
        total_action_dim = num_agents
        
        # Critic & target networks
        self.critic = CentralizedSacCritic(total_state_dim, total_action_dim).to(config.DEVICE)
        self.critic_target = CentralizedSacCritic(total_state_dim, total_action_dim).to(config.DEVICE)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_opt = optim.AdamW(self.critic.parameters(), lr=config.LR_CRITIC)
        
        # Temperature-weighted adaptive automatic entropy configuration
        self.target_entropy = -float(num_agents)
        self.log_alpha = torch.zeros(1, requires_grad=True, device=config.DEVICE)
        self.alpha_opt = optim.AdamW([self.log_alpha], lr=config.LR_ACTOR)
        
        self.memory = deque(maxlen=20000)

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def select_actions(self, states):
        actions = []
        for i, state in enumerate(states):
            state_t = state.to(config.DEVICE).unsqueeze(0)
            self.actors[i].eval()
            with torch.no_grad():
                act_t, _ = self.actors[i].sample_action(state_t)
            self.actors[i].train()
            actions.append(act_t.cpu().item())
        return actions

    def train_step(self):
        if len(self.memory) < config.BATCH_SIZE:
            return
            
        batch = random.sample(self.memory, config.BATCH_SIZE)
        s_list, a_list, r_list, ns_list, d_list = zip(*batch)
        
        # Parse tensor shapes
        states = [torch.stack([s[i] for s in s_list]).to(config.DEVICE) for i in range(self.num_agents)]
        actions = torch.tensor(np.array(a_list), dtype=torch.float32).to(config.DEVICE)
        next_states = [torch.stack([ns[i] for ns in ns_list]).to(config.DEVICE) for i in range(self.num_agents)]
        
        rewards = torch.tensor(np.array(r_list), dtype=torch.float32)[:, 0].to(config.DEVICE)
        dones = torch.tensor(np.array(d_list), dtype=torch.float32).to(config.DEVICE)
        
        joint_state = torch.cat(states, dim=-1)
        joint_next_state = torch.cat(next_states, dim=-1)

        # ---------------- CRITIC UPDATE ----------------
        with torch.no_grad():
            next_actions_list = []
            next_log_probs_list = []
            for i in range(self.num_agents):
                next_act, next_lp = self.actors[i].sample_action(next_states[i])
                next_actions_list.append(next_act)
                next_log_probs_list.append(next_lp)
                
            joint_next_action = torch.cat(next_actions_list, dim=-1)
            joint_next_log_prob = torch.cat(next_log_probs_list, dim=-1).sum(dim=-1)
            
            target_q1, target_q2 = self.critic_target(joint_next_state, joint_next_action)
            # SAC objective handles the minimum of twin q values adjusted by policy entropy values
            target_q = torch.min(target_q1, target_q2).squeeze(-1) - self.alpha * joint_next_log_prob
            y = rewards + (config.GAMMA * target_q * (1 - dones))
            
        curr_q1, curr_q2 = self.critic(joint_state, actions)
        critic_loss = nn.MSELoss()(curr_q1.squeeze(-1), y) + nn.MSELoss()(curr_q2.squeeze(-1), y)
        
        self.critic_opt.zero_grad()
        critic_loss.backward()
        self.critic_opt.step()

        # ---------------- ACTOR & ALPHA UPDATE ----------------
        curr_actions_list = []
        curr_log_probs_list = []
        for i in range(self.num_agents):
            curr_act, curr_lp = self.actors[i].sample_action(states[i])
            curr_actions_list.append(curr_act)
            curr_log_probs_list.append(curr_lp)
            
        joint_curr_action = torch.cat(curr_actions_list, dim=-1)
        joint_curr_log_prob = torch.cat(curr_log_probs_list, dim=-1).sum(dim=-1)
        
        q1_new, q2_new = self.critic(joint_state, joint_curr_action)
        q_new = torch.min(q1_new, q2_new).squeeze(-1)
        
        actor_loss = (self.alpha.detach() * joint_curr_log_prob - q_new).mean()
        
        for opt in self.actor_opts: opt.zero_grad()
        actor_loss.backward()
        for opt in self.actor_opts: opt.step()
        
        # Automatically tune entropy temperature alpha parameter
        alpha_loss = -(self.log_alpha * (joint_curr_log_prob + self.target_entropy).detach()).mean()
        self.alpha_opt.zero_grad()
        alpha_loss.backward()
        self.alpha_opt.step()

        # ---------------- TARGET NETWORK POLYAK SOFT UPDATES ----------------
        for p, tp in zip(self.critic.parameters(), self.critic_target.parameters()):
            tp.data.copy_(config.TAU * p.data + (1.0 - config.TAU) * tp.data)

# ==========================================
# MAIN EXECUTION ENTRY PIPELINE
# ==========================================
if __name__ == "__main__":
    print("Loading SQLite data frames into local session layers...")
    fundamentals, asset_returns = load_financial_data()
    
    analyzer = LocalOllamaAnalyzer(model_name="llama3:8b")
    fundamentals['nlp_event_sentiment'] = analyzer.analyze_event(
        "Unexpected shift in corporate leadership triggers rapid operational restructuring across core nodes."
    )
    
    env = MultiAgentPortfolioEnv(fundamentals, asset_returns)
    state_flat_dim = fundamentals.shape[1] * config.LOOKBACK_WINDOW
    num_assets = len(config.TICKERS)
    
    sac_system = MultiAgentSAC(state_dim=state_flat_dim, num_agents=num_assets)
    
    print(f"\nLaunching Multi-Agent Off-Policy SAC Pipeline across {config.EPISODES} runs...")
    for episode in range(config.EPISODES):
        states = env.reset()
        episode_reward = 0.0
        done = False
        
        while not done:
            actions = sac_system.select_actions(states)
            next_states, rewards, done = env.step(actions)
            
            sac_system.memory.append((states, actions, rewards, next_states, done))
            sac_system.train_step()
            
            states = next_states
            episode_reward += rewards[0]
            
        print(f"Episode {episode+1:02d}/{config.EPISODES} | Collective Team SAC Return: {episode_reward:+.4%}")
