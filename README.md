# 🚀 DeepRL-Sector-Quant: Autonomous Sector Portfolio Optimization Engine

![Company Logo](https://www.a2zchambers.com)

An enterprise-ready, modular deep reinforcement learning (DRL) framework optimized for M-series MacBook Pros (`mps` hardware acceleration). This repository parses raw corporate text events via local **Ollama LLMs** and cross-references them against structured relational **yfinance SQLite tables**. The objective is to train autonomous agents capable of continuous or discrete asset allocation within highly correlated sector universes (e.g., Semiconductors, Software, Energy).

---

## 📊 1. Quantitative Event Impact & KPI Architecture

The project features a unique state-space architecture that bridges qualitative unstructured news with quantitative macroeconomic data.

+------------------------------------------+
|            STATE SPACE MATRIX            |
+------------------------------------------+
|  [Qualitative NLP Vectors]               | --> Local Ollama (llama3)
|   - Earnings Surprises, Litigation       |
|   - Management Changes, Regulation       |
+------------------------------------------+
|  [Quantitative Financial KPIs]           | --> SQLite Relational Tables
|   - Gross Margin, YoY Growth, Leverage   |
+------------------------------------------+

### Unstructured Event Impact Extraction (Ollama Pipeline)
Rather than relying on basic dictionary-based sentiment tools, the system passes raw corporate news timelines directly into a local MacBook-hosted **Ollama model (`llama3:8b`)**. The pipeline converts complex, ambiguous events into a standardized risk-impact scale spanning `[-1.0, 1.0]`:
*   **Earnings Surprises:** Models immediate sentiment shocks against consensus guidance.
*   **Management Restructuring:** Captures operational risk vectors (e.g., sudden COO/CEO departures).
*   **Product Launches:** Gauges market-entry efficacy and future revenue pipelines.
*   **Litigation & Regulation:** Maps structural downside threats (e.g., DOJ antitrust probes or supply restrictions).

### Tracking KPI Drivers across Stock Universes
Different companies inside the same sector are driven by entirely different financial ratios. Our environment automatically calculates trailing variables to see how individual asset prices respond to distinct balance-sheet parameters:
*   **Growth-Driven Assets (e.g., NVDA):** The agent discovers high correlations to `YoY Growth` and `Gross Margin`, tilting weights heavily toward these assets during macro tech expansion phases.
*   **Value/Stability-Driven Assets (e.g., INTC):** The agent tracks downside protection metrics such as the `Leverage Ratio`, learning to reallocate funds to defensively positioned firms during contraction regimes.

---

## 🎯 2. Model Selection Matrix: Use Case Mapping

This platform includes **8 cutting-edge reinforcement learning paradigms**. Each has been fine-tuned for a specific operational or research mandate:

| Use Case / Strategic Mandate | Optimal Model Choice | Rationale |
| :--- | :--- | :--- |
| **Strict Filtering & Stock Screening** | **DDQN** | Perfect for discrete Buy/Hold binary classifications without structural weighting overhead. |
| **Long-Term Institutional Wealth & Index Tracking** | **SAC** | Optimizes for Maximum Entropy. Automatically forces diversification to prevent the portfolio from hyper-concentrating into a single asset. |
| **Active Cross-Sectional Daily Rebalancing** | **TD3** | Twin critics block overestimation bias while target action smoothing eliminates execution noise. |
| **High-Risk Macro & Regulatory Regimes** | **PPO / MAPPO** | Bounded policy clipping prevents catastrophic strategy decay when processing volatile textual news events. |
| **Competitive Alpha Generation & Market Making** | **MADDPG / MAPPO** | Centralized Training with Decentralized Execution allows assets to actively manage cross-asset correlation risks as cooperative/competitive agents. |
| **Edge Deployment & Low-Memory Constraints** | **A2C** | Synchronous on-policy calculations eliminate the need for a memory-heavy replay buffer, optimizing processing speeds. |

---

## 🛠️ 3. Repository Organization & Execution

### Project Folder File Trees
```text
/
├── config.py              # Centralized hyperparameters & absolute DB routing paths
├── database.py            # SQLite parsing engine, dynamic KPI calculation, and date union logic
├── fetch_prices.py        # 5-Year daily pricing yfinance scraper pipeline
├── models.py              # DDPG and DDQN network deep neural layers
├── ppo_models.py          # Stochastic PPO network architectures (Dirichlet)
├── maddpg_models.py       # Multi-agent centralized critic networks
├── mappo_models.py        # Multi-agent PPO stochastic networks (Beta)
├── sac_models.py          # Twin critic Maximum Entropy SAC networks
├── a2c_models.py          # Shared backbone A2C architectures
├── td3_models.py          # Delayed policy twin-critic TD3 structures
├── main.py                # DDPG default optimizer script
├── main_ppo.py            # PPO optimization run orchestrator
├── MADDPG_main.py         # Multi-agent deterministic executor
├── MAPPO_main.py          # Multi-agent stochastic policy executor
├── main_sac.py            # Soft Actor-Critic execution script
├── A2C_main.py            # Advantage Actor-Critic on-policy executor
└── main_td3.py            # Twin Delayed execution script
```

### Initial Execution Workflow
1.  **Pull the Local Inference Model:**
    ```bash
    ollama pull llama3:8b
    ```
2.  **Generate Price Data Matrices:**
    ```bash
    python3 fetch_prices.py
    ```
3.  **Run Your Chosen Portfolio Optimization Script:**
    ```bash
    python3 -m MAPPO_main
    ```

---

## ⚖️ 4. Copyright, Compliance & Disclaimers

### Legal Disclaimer
**CRITICAL NOTICE: FOR EDUCATIONAL AND ACADEMIC RESEARCH PURPOSES ONLY.**  
The source code, mathematical representations, data integration samples, and optimization models contained within this project folder are provided exclusively for instructional and developer testing purposes. They do not constitute financial advice, investment recommendations, or an endorsement to buy or sell any financial asset or instrument. 

Quantitative trading strategies and reinforcement learning architectures carry substantial financial risk. Past simulated performance yields no guarantees of future actual returns. You must consult with a qualified, licensed financial professional, certified accountant, or investment advisor before allocating capital or executing any financial markets strategy based upon this system's architecture. The authors and maintainers assume zero liabilities for financial or computational losses incurred through direct or indirect use of this repository.

### Copyright Notice
© 2026 A2Z Chambers Inc. All Rights Reserved.  
Licensed under the Apache License, Version 2.0 (the "License"); you may not use these components except in structural compliance with corporate authorization bounds.
