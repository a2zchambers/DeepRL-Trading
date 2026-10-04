# Proximal Policy Optimization (PPO) Portfolio Engine

## Technical Overview
PPO is a synchronous, on-policy stochastic policy gradient method. It solves the performance instability common in standard policy gradients by utilizing a **clipped surrogate objective function**. This optimization strategy limits policy update parameter shifts to a strict boundary constraint (typically 1 ± ε, or 0.8 to 1.2), preventing destructive policy drift in noisy environments. 

In this implementation, the Actor network outputs the parameters of a **Dirichlet Distribution**. The Dirichlet distribution naturally outputs a continuous probability vector that sums exactly to `1.0`, satisfying long-only portfolio constraint allocations out-of-the-box.

### Key Metrics Handled
- **Objective Function:** Clipped Surrogate Advantage Optimization
- **Action Space Transformation:** Stochastic Dirichlet Distribution Sampling
- **State space integration:** Flattened lookback window tracking sequences of combined fundamental data and local Ollama news sentiment scores.

## Pros & Cons
### Pros
* **Exceptional Update Stability:** The policy clipping constraint limits massive swings in asset allocations, making it highly reliable.
* **Native Portfolio Constraint Enforcements:** The Dirichlet distribution prevents allocations from drifting outside legal asset boundaries.
* **Low Sensitivity to Random Seeds:** Converges far more reliably than traditional off-policy continuous algorithms.

### Cons
* **High Sample Inefficiency:** Being an on-policy method, data samples collected during environment exploration are discarded immediately after an update, requiring a large number of rows to converge.
* **Vulnerability to Sparse Data Trajectories:** Performance degrades if data alignments contain wide gaps between filing intervals.

## Financial Use Case & Implementation
* **When to Use:** Best deployed for **Macro Asset Allocation and Thematic Sector Rotation** where sudden, volatile sentiment shifts (e.g., regulatory changes parsed via Ollama) could destabilize off-policy agents.
* **How to Execute:** Ensure `ppo_models.py`, `main_ppo.py`, `config.py`, and `database.py` occupy the same folder, then execute:
  ```bash
  python3 -m main_ppo
  ```
