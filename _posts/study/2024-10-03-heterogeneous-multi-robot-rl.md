---
layout: post
title: Heterogeneous Multi-Robot Reinforcement Learning
subtitle: "Paper notes: HetGPPO vs. GPPO (Bettini et al., 2023)"
thumbnail-img: /assets/img/GPPO_HetGPPPO.png
tags: [multi-agent-rl, multi-robot, gnn, paper-review]
category: study
mathjax: true
---

## Introduction

Most multi-agent reinforcement learning (MARL) methods share parameters across agents, which limits how heterogeneous the learned policies can be and makes training effectively centralized. Bettini et al. propose **HetGPPO** (Heterogeneous Graph Neural Network Proximal Policy Optimization), which overcomes these limitations in partially observable environments while enabling fully decentralized learning.

The paper reports two main results:
1. **Superior Performance in Strong Heterogeneity**: When homogeneous methods fail under strong heterogeneous requirements, HetGPPO succeeds.
2. **Higher Resilience in Weak Heterogeneity**: When homogeneous methods can learn *apparently* heterogeneous behaviors, both models reach the same reward without noise, but HetGPPO achieves higher resilience to both training and deployment noise.

## Classification of Heterogeneous Systems

The paper first defines what a heterogeneous system is, using two criteria: **P (Physical)** and **B (Behavioral)**:

### Physical Heterogeneity (P)
Refers to systems where agents/robots are fundamentally different types. Examples include:
- Drones + quadruped robots + underwater vehicles
- Different sensor capabilities and hardware specifications

### Behavioral Heterogeneity (B)
Divided into two subcategories:

#### B_s (Same Objective)
- Agents share the same objective function
- Use identical reward functions
- Coordinate toward common goals

#### B_d (Different Objectives)  
- Agents have different objective functions
- Use different reward functions
- Can be applied to adversarial settings, non-cooperative environments, or hierarchical RL

### System Classifications

Using these criteria, we can classify heterogeneous systems into five types:

1. **P\B**: Physical differences only
   - Different hardware but same behavioral policies
   - *My note: this case seems unlikely to be optimal, since physical differences should usually call for different behaviors.*

2. **P ∩ B_d**: Complete heterogeneity 
   - Different robots with different objectives
   - Most complex form

3. **P ∩ B_s**: Physical diversity, shared goals
   - Different robots working toward same objective
   - Example: Mixed robot types exploring an area

4. **B_s\P**: Behavioral diversity, same hardware
   - Identical robots with different behaviors for shared goals
   - Example: Some drones for fast/wide search, others for slow/detailed search

5. **B_d\P**: Same hardware, different objectives
   - Identical robots with different missions
   - Example: Wildfire response drones (some for rescue, others for fire mapping)

**Important Note**: "Same objective" doesn't mean "same behavior" - it assumes different behaviors toward a common goal.

## POMDP and Communication Graph

### POMDP Formulation

The system is modeled as a Partially Observable Markov Decision Process (POMDP), defined by the tuple:

$$(\mathcal{V}, \mathcal{S}, \mathcal{O}, \{\sigma_i\}_{i \in \mathcal{V}}, \mathcal{A}, \{R_i\}_{i \in \mathcal{V}}, T, \gamma)$$

Where:
- $\mathcal{V}$: Set of agents
- $\mathcal{S}$: State space
- $\mathcal{O}$: Observation space  
- $\{\sigma_i\}_{i \in \mathcal{V}}$: Observation function for agent $i$
- $\mathcal{A}$: Action space
- $\{R_i\}_{i \in \mathcal{V}}$: Reward function for agent $i$, $R_i: \mathcal{S} \times \mathcal{A} \times \mathcal{S} \rightarrow \mathbb{R}$
- $T$: State transition function, $T: \mathcal{S} \times \mathcal{A} \times \mathcal{S} \rightarrow [0,1]$
- $\gamma$: Discount factor

### Communication Graph

Additionally, the authors define a communication graph $G = (\mathcal{V}, \mathcal{E})$ where:
- $\mathcal{V}$: Set of agents
- $\mathcal{E}$: Communication links (time-varying based on agent proximity)

This represents the maximum communication range and changes dynamically over time.

![GPPO vs HetGPPO](/assets/img/GPPO_HetGPPPO.png)
*Architecture comparison: GPPO (left) with shared parameters vs HetGPPO (right) with individual parameters*

## GPPO (Graph Neural Network Proximal Policy Optimization)

### Evolution from IPPO

GPPO builds upon IPPO (Independent Proximal Policy Optimization), where each agent maintains its own critic and actor networks based solely on local observations. IPPO's main issues:

- **Non-stationarity**: Other agents appear as part of the environment
- **Learning instability**: When other agents' policies change, the environment becomes non-stationary

### GPPO Solution

GPPO inherits IPPO's advantages while addressing its limitations through **GNN communication layers** that enable:
- Information sharing between neighboring agents
- Mitigation of partial observability
- Improved learning stability

### Architecture Details

**Critic and Actor**: $V_i(o_{N_i})$ and $\pi_i(o_{N_i})$ consider neighborhood observations $o_{N_i}$

**Feature Processing**:
- **Absolute features**: Position $p_i$ and velocity $v_i$ (used for edge computation)
- **Relative features**: Other observations embedded as $z_i$ through MLP encoder

**Edge Features**: For agents $i$ and $j$:
$$p_{ij} = p_i - p_j, \quad v_{ij} = v_i - v_j$$
$$e_{ij} = p_{ij} \| v_{ij}$$

**Message-Passing GNN Kernel**:
$$h_i = \psi_{\theta_i}(z_i) + \bigoplus_{j \in N_i} \phi_{\theta_i}(z_j \| e_{ij})$$

Where:
- $\psi_{\theta_i}$ and $\phi_{\theta_i}$: Two MLPs
- $\bigoplus$: Aggregation operation (sum)
- $\theta_i$: **Shared parameters across all agents**

### Limitations

- **Centralized training**: Due to parameter sharing
- **Homogeneous behavior**: All agents become similar due to shared $\theta_i$

## HetGPPO (Heterogeneous Graph Neural Network Proximal Policy Optimization)

### Key Innovation: Removing Parameter Sharing

HetGPPO **eliminates parameter sharing**, which has profound implications:

**Lost Property**: **Permutation Equivariance**
- GPPO: Input node order doesn't affect output (due to shared parameters)  
- HetGPPO: Different parameters → order matters → no permutation equivariance

### Trade-offs

**Disadvantages**:
- Reduced generalization ability
- Lower sample efficiency
- Loss of permutation equivariance

**Advantages** (outweighing disadvantages):
- **Truly decentralized learning**: No global information required
- **Local gradient backpropagation**: Learning through neighborhood interactions
- **Individual specialization**: Each agent develops unique message encoding/interpretation

### Decentralized Training and Execution

HetGPPO is decentralized in both phases:
- **Decentralized training**: no central coordinator is required
- **Decentralized execution**: each agent acts on local and neighborhood information

## Experimental Results

The experiments show that:

1. **Strong Heterogeneity Scenarios**: HetGPPO succeeds where homogeneous methods fail
2. **Weak Heterogeneity Scenarios**: both models solve the task without noise, but HetGPPO degrades far more gracefully as observation noise increases
3. **Real-World Transfer**: the give-way policy is deployed on physical robots, not only evaluated in simulation

## Key Takeaways

### Theoretical Contributions
- **Novel heterogeneity classification** (P, B_s, B_d framework)
- **Decentralized GNN-based MARL** architecture
- **Analysis of permutation equivariance** trade-offs

### Practical Implications
- **Real-world applicability**: Truly decentralized systems
- **Scalable heterogeneous teams**: No central coordination bottleneck
- **Flexible deployment**: Supports various heterogeneity requirements

### Future Directions
- Dynamic heterogeneity (battery levels, hardware degradation)
- Larger-scale deployments
- Integration with other MARL paradigms

## References

[1] Bettini, M., Shankar, A., & Prorok, A. (2023). Heterogeneous Multi-Robot Reinforcement Learning. 

Paper Link: [https://arxiv.org/abs/2301.07137](https://arxiv.org/abs/2301.07137)
