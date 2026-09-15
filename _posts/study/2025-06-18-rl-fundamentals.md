---
layout: post
title: "From Markov Processes to Reinforcement Learning"
subtitle: "How MP, MRP, MDP, and Monte Carlo methods build on one another"
tags: [reinforcement-learning, mdp, monte-carlo]
category: study
mathjax: true
---

## Introduction

Most introductions to reinforcement learning start from algorithms. This note takes the other route: it starts from the Markov process and adds one ingredient at a time, showing which limitation each extension removes. The sequence is MP, MRP, MDP, then the split between known and unknown models, which is where learning begins.

## Step 1: Markov Process (MP)

A **Markov process** is the simplest case. Take a weather model with three states (sunny, cloudy, rainy) in which tomorrow's weather depends only on today's, not on earlier days.

A Markov Process is defined by:
- **S**: A set of states (sunny, cloudy, rainy)
- **P**: Transition probabilities P(s'|s)

```
MP = (S, P)
```

**Key insight**: The Markov property captures the idea that "the future depends only on the present, not the past." This is both a limitation and a strength—it simplifies modeling but assumes away potentially important historical dependencies.

### What's missing?
We can model state transitions, but we can't express preferences or goals. We need a way to distinguish between "good" and "bad" states.

## Step 2: Markov Reward Process (MRP)

Adding **rewards** to the process gives a Markov reward process. In the weather example, sunny days might carry +10 and rainy days -5.

An MRP extends MP with:
- **R**: Reward function R(s) giving immediate reward for being in state s
- **γ**: Discount factor (0 ≤ γ ≤ 1) for future rewards

```
MRP = (S, P, R, γ)
```

Now we can define the **value** of a state:

```
V(s) = E[G_t | S_t = s]
where G_t = R_t+1 + γR_t+2 + γ²R_t+3 + ...
```

**Key insight**: The value function captures the long-term desirability of being in a state, balancing immediate rewards with future possibilities.

### The Bellman equation
```
V(s) = R(s) + γ ∑_{s'} P(s'|s) V(s')
```

The value of a state equals its immediate reward plus the discounted expected value of its successor states.

### What's still missing?
We can evaluate states, but we can't influence the process. We need the ability to make choices.

## Step 3: Markov Decision Process (MDP)

Adding **actions** gives a Markov decision process. The agent now influences the process, for example by choosing whether to carry an umbrella.

An MDP extends MRP with:
- **A**: Set of actions available in each state
- **P**: Transition probabilities P(s'|s,a) now depend on actions
- **R**: Reward function R(s,a) or R(s,a,s') now depends on actions

```
MDP = (S, A, P, R, γ)
```

Now we need a **policy** π(a|s) that tells us what action to take in each state.

The value functions become:
```
V^π(s) = E_π[G_t | S_t = s]
Q^π(s,a) = E_π[G_t | S_t = s, A_t = a]
```

**Key insight**: Actions turn evaluation into control—the agent must now choose, not just predict. The exploration vs. exploitation dilemma appears only later, once the model is unknown.

### The Bellman Equations for MDPs:
```
V^π(s) = ∑_a π(a|s) ∑_{s'} P(s'|s,a) [R(s,a,s') + γV^π(s')]
Q^π(s,a) = ∑_{s'} P(s'|s,a) [R(s,a,s') + γ ∑_{a'} π(a'|s') Q^π(s',a')]
```

## The Fundamental Dichotomy: Known vs Unknown MDPs

This is the point where the framework splits. Everything we've discussed so far assumes we **know** the MDP—we know P(s'|s,a) and R(s,a,s'). But what happens when we don't?

### Case 1: Known MDP - Dynamic Programming

When we know the MDP completely, we can solve it using **Dynamic Programming**:

**Policy Evaluation** (computing V^π):
```python
# Iterative approach
for s in states:
    V_new[s] = sum(π(a|s) * sum(P(s'|s,a) * [R(s,a,s') + γ*V[s']] 
                                 for s' in states) 
                   for a in actions)
```

**Policy Improvement**:
```python
# Greedy policy improvement
π_new(s) = argmax_a sum(P(s'|s,a) * [R(s,a,s') + γ*V[s']] for s' in states)
```

**Value Iteration** combines both:
```python
V_new[s] = max_a sum(P(s'|s,a) * [R(s,a,s') + γ*V[s']] for s' in states)
```

### Case 2: Unknown MDP - The Learning Problem

But what if we don't know P(s'|s,a) or R(s,a,s')? We can only **experience** the MDP by taking actions and observing outcomes. This is where **Reinforcement Learning** truly begins.

## Monte Carlo Methods

**Monte Carlo methods** are the first approach to unknown MDPs. The key insight: instead of computing expected returns using transition probabilities, we can **sample** actual returns and average them.

### First-Visit Monte Carlo Prediction:
```python
def mc_prediction(policy, episodes):
    returns = defaultdict(list)
    V = defaultdict(float)
    
    for episode in episodes:
        G = 0
        for t in reversed(range(len(episode))):
            state, action, reward = episode[t]
            G = reward + gamma * G
            if state not in [x[0] for x in episode[:t]]:  # first visit
                returns[state].append(G)
                V[state] = np.mean(returns[state])
    return V
```

**Key point**: We don't need to know P(s'|s,a) to compute V^π(s). We just need to follow π and average the returns we observe.

### Monte Carlo Control (ε-greedy):
```python
def mc_control_epsilon_greedy(episodes, epsilon=0.1):
    Q = defaultdict(lambda: defaultdict(float))
    returns = defaultdict(lambda: defaultdict(list))
    policy = {}
    
    for episode in episodes:
        G = 0
        for t in reversed(range(len(episode))):
            state, action, reward = episode[t]
            G = reward + gamma * G
            if not any(x[0] == state and x[1] == action for x in episode[:t]):
                returns[state][action].append(G)
                Q[state][action] = np.mean(returns[state][action])
        
        # Policy improvement: make the policy ε-greedy w.r.t. the current Q
        for state in Q:
            best_action = max(Q[state], key=Q[state].get)
            n = len(Q[state])
            policy[state] = {a: epsilon / n + (1 - epsilon if a == best_action else 0)
                             for a in Q[state]}

    return Q, policy
```

## Strengths and Limitations of Monte Carlo

### Strengths
1. **Model-free**: No need to know transition probabilities
2. **Unbiased**: Given enough samples, converges to true values  
3. **Simple**: Just average observed returns
4. **No bootstrapping**: Each estimate is independent

### Limitations
1. **High variance**: Individual episodes can be very different
2. **Episode-based**: Must wait for episode completion
3. **Slow convergence**: Needs many episodes for good estimates
4. **Memory intensive**: Must store all returns

## Beyond Monte Carlo

The later developments of the field each address one of the limitations above:

### Temporal Difference Learning
- **Key insight**: Why wait for episode completion? Update estimates online!
- **SARSA**: On-policy TD control
- **Q-Learning**: Off-policy TD control
- **TD(λ)**: Bridging MC and TD methods

### Function Approximation
- **Problem**: What about continuous or very large state spaces?
- **Solution**: Approximate value functions with neural networks
- **DQN**: deep Q-networks, value functions approximated by neural networks

### Policy Gradient Methods
- **Alternative approach**: Instead of learning values, learn policies directly
- **REINFORCE**: Monte Carlo policy gradients
- **Actor-Critic**: Combining value and policy learning

## Summary of the Progression

Each step removes one limitation of the previous framework:

1. **MP → MRP**: From transitions to preferences
2. **MRP → MDP**: From evaluation to control  
3. **Known MDP → Unknown MDP**: From planning to learning
4. **Monte Carlo → TD**: From episodic to online learning
5. **Tabular → Function Approximation**: From finite to infinite spaces

Each extension keeps the earlier formulation as a special case while enlarging the class of solvable problems.

## Open Problems

### Sample Efficiency
The core challenge: How do we learn effectively from limited experience?
- **Meta-learning**: Learning to learn faster
- **Model-based RL**: Learning world models for planning
- **Transfer learning**: Leveraging past experience

### Exploration vs Exploitation
How do we balance trying new things with doing what we know works?
- **Upper Confidence Bounds**: Principled exploration
- **Thompson Sampling**: Bayesian exploration
- **Curiosity-driven learning**: Intrinsic motivation

### Safety and Robustness  
Real-world deployment concerns: How do we ensure safe learning?
- **Safe RL**: Constraints during learning
- **Robust RL**: Handling uncertainty and distribution shift
- **Interpretable RL**: Understanding agent decisions

## Conclusion

Reinforcement learning follows from a sequence of simple questions:

- "How do we model sequential processes?" → Markov Processes
- "How do we express preferences?" → Markov Reward Processes  
- "How do we make decisions?" → Markov Decision Processes
- "What if we don't know the model?" → Monte Carlo Methods
- "How do we learn more efficiently?" → Temporal Difference Learning

Each answer introduces the limitation that the next question addresses. Seen this way, the algorithms are not a catalogue to memorize but a sequence of responses to the same underlying problem: decision-making under uncertainty when the model is unknown.

---

## References

1. **Sutton, R. S., & Barto, A. G. (2018).** *Reinforcement Learning: An Introduction*. MIT Press. 
   - The definitive textbook covering the conceptual journey outlined here

2. **Bellman, R. (1957).** *Dynamic Programming*. Princeton University Press.
   - The foundational work on optimal control and dynamic programming

3. **Howard, R. A. (1960).** *Dynamic Programming and Markov Processes*. MIT Press.
   - Early formalization of Markov Decision Processes

4. **Robbins, H., & Monro, S. (1951).** A stochastic approximation method. *The Annals of Mathematical Statistics*, 22(3), 400-407.
   - Theoretical foundations for stochastic optimization in RL
 