# Advantage Actor-Critic (A2C) for Continuous Rebalancing

## Technical Overview
A synchronous, on-policy actor-critic algorithm. Trajectories are collected across the current baseline allocation timeline using a shared neural backbone, and policy updates are calculated instantly using an advantage value metric ($A(s,a) = Q(s,a) - V(s)$).

## Pros & Cons
* **Pros:** Zero replay buffer overhead means low system memory utilization; simpler codebase; fast execution passes on Apple Silicon devices.
* **Cons:** Highly vulnerable to local minima traps; training steps can become unstable if gradient sizes are not clipped aggressively.

## Financial Use Case
* **When to Use:** Best for **lightweight, real-time edge deployment pipelines** running on consumer laptops where tracking data memory structures are highly limited.
* **How to Execute:**
  ```bash
  python3 -m A2C_main
  ```
