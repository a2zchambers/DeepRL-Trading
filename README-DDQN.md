# Double Deep Q-Network (DDQN) for Asset Selection

## Technical Overview
DDQN addresses the overestimation bias inherent in vanilla DQN by decoupling action selection from action evaluation. It uses the online policy network ($\theta$) to select the greedy action, but evaluates that action using the target network ($\bar{\theta}$) to compute the target Q-value.

## Pros & Cons
* **Pros:** Highly stable discrete convergence; low computational overhead; excellent at strict binary/trinary decisions (e.g., Hold vs. Buy).
* **Cons:** Cannot natively handle continuous action spaces; requires asset allocation discretization, which scales poorly when handling large asset universes due to the curse of dimensionality.

## Financial Use Case
* **When to Use:** Use this for **binary stock screening** (e.g., classifying stocks as Outperform vs. Underperform) or managing a **single stock's position limits** (e.g., long vs. flat exposure).
* **How to Execute:**
  ```bash
  python3 -m main
  ```
