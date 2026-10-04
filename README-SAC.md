# Soft Actor-Critic (SAC) Portfolio Allocation Engine

## Technical Overview
SAC is a highly efficient, off-policy stochastic actor-critic framework. It stands apart from standard policy gradients by implementing **Maximum Entropy Reinforcement Learning**. Instead of optimizing purely for maximum historical absolute returns, SAC maximizes a dual objective: absolute returns plus policy randomness (entropy: \(H(\pi(\cdot\vert{}s))\)). 

This maximum entropy constraint encourages continuous exploration and prevents the policy from collapsing prematurely into single, hyper-concentrated stock options (e.g., placing 100% of the weight on a single ticker like NVDA). It implements an automated temperature balancing module to scale exploration limits continuously, alongside Twin Centralized Critics to mitigate value overestimation bias.

### Key Metrics Handled
- **Objective Function:** Maximum Entropy Off-Policy Value Minimization
- **Action Space Transformation:** Continuous Squashed Gaussian Distribution scaled via Tanh and Softmax
- **Overestimation Protection:** Minimum expectation targets calculated across two independent centralized critics (Q₁, Q₂)

## Pros & Cons
### Pros
* **Highest Diversification Enforcements:** The explicit entropy penalty forces the model to actively seek diversified asset balances.
* **Superb Sample Efficiency:** Replay buffer architectures allow reuse of past historical trajectory vectors.
* **Reparameterization Stability:** Uses `dist.rsample()` to map backpropagation pathways down to shared linear configurations smoothly.

### Cons
* **Complex Multi-Parameter Dependencies:** Balancing the relationship between the base reward function and the automated entropy temperature parameter (α) can require heavy tuning.
* **High Memory Footprint:** The combination of twin centralized critics and an extensive experience replay memory structure utilizes significant system resources.

## Financial Use Case & Implementation
* **When to Use:** The prime candidate for **Long-Term Institutional Wealth Management, Index Tracking, and Core Equity Portfolios** where diversification is required, and hyper-concentration constitutes unacceptable tail risk.
* **How to Execute:** Ensure `sac_models.py`, `main_sac.py`, `config.py`, and `database.py` occupy the same folder, then execute:
  ```bash
  python3 -m main_sac
  ```
