# Multi-Agent Deep Deterministic Policy Gradient (MADDPG) for Portfolio Management

## Technical Overview
An off-policy framework designed for multi-agent systems, following the **Centralized Training with Decentralized Execution (CTDE)** paradigm. Each stock ticker operates as an autonomous agent with a localized Actor, while a shared Centralized Critic tracks the joint states and combined actions of all agents during training.

## Pros & Cons
* **Pros:** Explicitly models cross-asset relationships; allows agents to coordinate allocations to manage sector concentration risk.
* **Cons:** Computational overhead scales aggressively as you add tickers; the centralized critic's input space expands quadratically with larger asset classes.

## Financial Use Case
* **When to Use:** Use this for **highly correlated sector rotation** (such as Semiconductors or Energy) where individual assets must dynamically adjust their weightings based on the actions of their direct competitors.
* **How to Execute:**
  ```bash
  python3 -m MADDPG_main
  ```
