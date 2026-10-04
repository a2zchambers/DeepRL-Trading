# Multi-Agent Proximal Policy Optimization (MAPPO) for Market Making

## Technical Overview
Combines the multi-agent setup of MADDPG with the clipping objective stability of PPO. It utilizes a Centralized Critic to evaluate the global state advantage parameter tracking, while decentralized stochastic Actors sample allocations using a bounded **Beta Distribution**.

## Pros & Cons
* **Pros:** Highly stable multi-agent optimization updates; the Beta distribution keeps outputs natively bounded between `0.0` and `1.0` without requiring clipping hacks.
* **Cons:** Requires large trajectory batches to stabilize the centralized advantage calculations; high sample overhead.

## Financial Use Case
* **When to Use:** Ideal for **stochastic continuous market-making or thematic multi-asset trading systems** where you need strict risk limits and stable multi-agent performance.
* **How to Execute:**
  ```bash
  python3 -m MAPPO_main
  ```
