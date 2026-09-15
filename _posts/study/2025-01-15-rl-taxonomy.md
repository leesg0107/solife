---
layout: post
title: "The Architecture of Reinforcement Learning: A Structural Guide"
subtitle: "Understanding RL through its fundamental dichotomies and design choices"
tags: [reinforcement-learning, taxonomy]
category: study
mathjax: true
---
## Introduction

New reinforcement learning algorithms appear constantly, but almost all of them are recombinations of a small number of structural choices. This post lays out those choices as I understand them: three axes that define the computational structure of an algorithm, plus a set of orthogonal concepts that can be layered on top of any of them. Knowing the axes makes a new algorithm much easier to place.

## 1. Model-Free vs Model-Based

An agent interacts with an environment, receives rewards, and learns. The first structural question is: **does the agent have access to a model of how the environment works?**

### Model-Based RL

**The agent knows (or learns) the environment dynamics and reward structure.**

In mathematical terms, the agent has access to:

- **Transition dynamics**: \\(P(s'|s,a)\\) - the probability of reaching state \\(s'\\) when taking action \\(a\\) in state \\(s\\)
- **Reward function**: \\(R(s,a)\\) - the immediate reward for taking action \\(a\\) in state \\(s\\)

With a model the agent can **plan**: it can simulate candidate action sequences before committing to one.

### Model-Free RL

**The agent doesn't know the environment dynamics or reward structure.**

The agent learns purely from **trial and error**: it improves its value estimates, its policy, or both from sampled experience, without representing the dynamics explicitly.

### Important Clarification: Prior Data ≠ Model-Based

Having demonstration data does not make an algorithm model-based. These are orthogonal concepts:

- **Model-Based/Free**: Whether you know the environment's transition dynamics
- **Imitation Learning**: Whether you use expert demonstrations (a different dimension entirely)

## 2. On-Policy vs Off-Policy

A policy maps states to actions. To define this axis we need to distinguish two policies:

- **Target Policy** (\\(\pi_{target}\\)): The policy the agent is trying to learn and improve
- **Behavior Policy** (\\(\pi_{behavior}\\)): The policy the agent actually uses to collect data

### On-Policy: Target = Behavior

**The agent learns from its own experiences using the same policy it's trying to improve.**

\\[
\pi_{target} = \pi_{behavior}
\\]

### Off-Policy: Target ≠ Behavior

**The agent can learn one policy while following a different policy to collect data.**

Data collected by another policy, including earlier versions of the agent's own policy, can be reused for learning.

\\[
\pi_{target} \neq \pi_{behavior}
\\]

## 3. Value-Based vs Policy-Based vs Actor-Critic

The third axis concerns what the algorithm learns. Two objects are involved:

- **Value Function** (\\(V(s)\\) or \\(Q(s,a)\\)): Evaluates how good a state (or state-action pair) is
- **Policy** (\\(\pi(a|s)\\)): Decides which action to take

### Value-Based: Learning Values, Deriving Actions

**Learn value functions, then derive the policy implicitly from those values.**

The action-value function \\(Q(s,a)\\) estimates the expected return of taking action \\(a\\) in state \\(s\\). Given \\(Q\\), the policy follows directly:

\\[
\pi(s) = \arg\max_a Q(s,a)
\\]

**The Q-function update formula:**

\\[
Q(s,a) \leftarrow Q(s,a) + \alpha \left[ r + \gamma \max_{a'} Q(s',a') - Q(s,a) \right]
\\]

where:

- \\(\alpha\\) is the learning rate
- \\(\gamma\\) is the discount factor
- \\(r\\) is the immediate reward

### Policy-Based: Learning Actions Directly

**Directly learn the policy \\(\pi_\theta(a|s)\\) without explicitly computing values.**

The policy is parameterized and optimized directly. It can be:

- **Stochastic**: \\(\pi_\theta(a|s)\\) outputs a probability distribution over actions
- **Deterministic**: \\(\pi_\theta(s)\\) directly outputs a single action

**The policy gradient theorem:**

\\[
\nabla_\theta J(\theta) = \mathbb{E}_{\pi_\theta} \left[ \nabla_\theta \log \pi_\theta(a|s) \cdot G_t \right]
\\]

where:

- \\(J(\theta)\\) is the expected return
- \\(G_t = \sum_{k=0}^{\infty} \gamma^k r_{t+k}\\) is the return from time \\(t\\)

### Actor-Critic: Learning Values and a Policy Together

**Use two networks: Actor (policy) + Critic (value function).**

The **actor** is the policy and the **critic** is a value function. The critic's estimate replaces the high-variance Monte Carlo return in the policy gradient, and the actor is updated in the direction the critic indicates.

\\[
\begin{align}
\text{Actor: } & \pi_\theta(a|s) \\\\
\text{Critic: } & V_\phi(s) \text{ or } Q_\phi(s,a)
\end{align}
\\]

**The advantage function (TD error):**

\\[
A(s,a) = r + \gamma V(s') - V(s)
\\]

The advantage measures whether an action produced a better or worse outcome than the critic expected.

## 4. Beyond the Core: Concepts in RL

The three axes above define the computational structure of an algorithm. A separate set of **concepts** can be combined with any of them. They do not change the structure; they change how the learning problem is posed, in order to improve sample efficiency, exploit the structure of a task, or handle a specific difficulty.

### Curriculum Learning

**Curriculum learning orders training tasks from easy to hard.** The agent acquires basic competence on simple versions of the task before being exposed to the full difficulty.

For example, teaching a robot to organize a warehouse:

1. **Easy**: Pick up a single object
2. **Medium**: Lift and carry the object
3. **Hard**: Navigate while carrying
4. **Very Hard**: Sort and organize multiple objects efficiently

\\[
\text{Task Difficulty: } \tau_1 \rightarrow \tau_2 \rightarrow \cdots \rightarrow \tau_n
\\]

The agent builds foundational skills before facing the full task, which reduces the chance of getting stuck in early exploration and typically shortens training.

### Meta-Learning (Learning to Learn)

**Meta-learning trains an agent to adapt quickly to new tasks, using experience on a distribution of related tasks.**

Instead of learning a single task, the agent learns a **learning strategy** that works across many tasks:

\\[
\theta^* = \text{Meta-Learner}(\mathcal{T}_1, \mathcal{T}_2, \ldots, \mathcal{T}_n)
\\]

where \\(\theta^*\\) are meta-parameters that can quickly adapt to any new task \\(\mathcal{T}_{new}\\).

**Example**: an agent trained on many manipulation tasks (grasping, opening doors, turning knobs) can adapt to a new one (opening a jar) in a few trials, because the meta-parameters already encode what manipulation tasks have in common.

### Hierarchical RL

Some tasks decompose naturally into levels. For "making breakfast":

- **High-level**: Decide to make eggs, then toast, then coffee
- **Mid-level**: For eggs—get pan, crack eggs, cook, serve
- **Low-level**: Motor commands—move arm 20cm, rotate wrist 30°, apply force

**Hierarchical RL decomposes complex tasks into subtasks and sub-goals:**

\\[
\pi_{\text{high}}(g|s) \quad \text{and} \quad \pi_{\text{low}}(a|s,g)
\\]

where \\(g\\) is a goal or subgoal, \\(\pi_{\text{high}}\\) chooses goals, and \\(\pi_{\text{low}}\\) executes low-level actions to achieve those goals.

### Multi-Agent RL (MARL)

In **multi-agent RL**, several agents learn simultaneously in the same environment, so each agent's environment includes the other learners. It is a large field with its own considerations; only the main challenges are listed here.

**Key Challenges**:

- **Non-stationarity**: The environment keeps changing as other agents learn
- **Credit assignment**: Which agent contributed to success/failure?
- **Scalability**: How many agents can be effectively trained?
- **Equilibrium selection**: Which equilibrium should be selected? This depends on game-theoretic considerations and reward structure.

### Other Important Concepts

**Offline RL (Batch RL)**: Learn from a fixed dataset without environment interaction. Critical for safety-critical domains (healthcare, autonomous driving) where exploration can be dangerous.

**Imitation Learning**: Learn from expert demonstrations. Includes behavioral cloning (supervised learning from demos) and inverse RL (infer the expert's reward function).

**Transfer Learning**: Use knowledge from one task to jumpstart learning on a related task.

**Partial Observability (POMDPs)**: The agent doesn't see the full state. Requires memory and belief state estimation.

## Algorithm Classification Map

The major algorithms can be placed on the three axes:

| Algorithm                  | Model | Policy | Learning                   |
| -------------------------- | ----- | ------ | -------------------------- |
| **Q-Learning**       | Free  | Off    | Value                      |
| **SARSA**            | Free  | On     | Value                      |
| **DQN**              | Free  | Off    | Value                      |
| **Double DQN**       | Free  | Off    | Value                      |
| **Dueling DQN**      | Free  | Off    | Value                      |
| **Rainbow**          | Free  | Off    | Value                      |
| **IQN**              | Free  | Off    | Value (Distributional)     |
| **C51**              | Free  | Off    | Value (Distributional)     |
| **REINFORCE**        | Free  | On     | Policy                     |
| **TRPO**             | Free  | On     | Policy                     |
| **PPO**              | Free  | On     | Actor-Critic               |
| **A2C**              | Free  | On     | Actor-Critic               |
| **A3C**              | Free  | On     | Actor-Critic               |
| **SAC**              | Free  | Off    | Actor-Critic               |
| **TD3**              | Free  | Off    | Actor-Critic               |
| **DDPG**             | Free  | Off    | Actor-Critic               |
| **QMIX**             | Free  | Off    | Value (Multi-Agent)        |
| **VDN**              | Free  | Off    | Value (Multi-Agent)        |
| **MADDPG**           | Free  | Off    | Actor-Critic (Multi-Agent) |
| **Value Iteration**  | Based | -      | Value                      |
| **Policy Iteration** | Based | -      | Policy                     |
| **MCTS**             | Based | -      | Search + Value             |
| **MuZero**           | Based | -      | Search + Value (Learned Model) |
| **AlphaZero**        | Based | -      | Search + Policy + Value    |
| **Dyna-Q**           | Based | Off    | Value (Model + RL)         |
| **PILCO**            | Based | On     | Policy                     |
| **MBPO**             | Based | Off    | Actor-Critic (Model-Based) |

## Understanding the Bigger Picture

**RL research is modular.** The three core axes define the computational structure, and the concepts can be mixed and matched:

\\[
\text{Any Algorithm} = \text{(Model Choice)} \times \text{(Policy Type)} \times \text{(Learning Method)} \times \text{(Optional Concepts)}
\\]

**Examples of combined approaches**:

- **Hierarchical Multi-Agent Curriculum RL with Imitation**: Teaching multiple robots to cooperate on complex tasks using expert demonstrations and progressive difficulty
- **Meta-Learning for Offline Actor-Critic**: Quick adaptation to new tasks using only logged data
- **Model-Based Hierarchical RL with Curriculum**: Planning over abstract goals with learned models, gradually increasing task complexity

## Conclusion

The variety of reinforcement learning algorithms comes from combinations of these axes and concepts. For any new algorithm, four questions locate it:

1. **Does it use a model of the environment?** (Model-Based vs Model-Free)
2. **Does it learn from its own actions or can it learn from other policies?** (On-Policy vs Off-Policy)
3. **Does it learn values, policies, or both?** (Value-Based vs Policy-Based vs Actor-Critic)
4. **What additional concepts does it employ?** (Curriculum, Meta-Learning, Hierarchical, etc.)

Answering them gives the algorithm's structure, its likely strengths and weaknesses, and its nearest relatives. New methods will keep appearing, but they answer the same questions in different combinations.

---

## References

1. **Sutton, R. S., & Barto, A. G. (2018).** *Reinforcement Learning: An Introduction* (2nd ed.). MIT Press.
2. **Mnih, V., et al. (2015).** Human-level control through deep reinforcement learning. *Nature*, 518(7540), 529-533.
3. **Schulman, J., et al. (2017).** Proximal Policy Optimization Algorithms. *arXiv preprint arXiv:1707.06347*.
4. **Lillicrap, T. P., et al. (2015).** Continuous control with deep reinforcement learning. *arXiv preprint arXiv:1509.02971*.
5. **Haarnoja, T., et al. (2018).** Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor. *ICML*.
6. **Silver, D., et al. (2017).** Mastering the game of Go without human knowledge. *Nature*, 550(7676), 354-359.

