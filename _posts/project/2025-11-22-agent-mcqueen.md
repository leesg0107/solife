---
layout: post
title: Agent Mcqueen
subtitle: Head-to-Head Competitive Racing using PPO
tags: [reinforcement-learning, ppo, f1tenth, multi-agent-rl]
category: project
mathjax: true
mermaid: true
thumbnail-img: "/assets/img/Agent-Mcqueen/agent-mcqueen-f1tenth.png"
---
Racing behaviors such as overtaking, defending, and positioning are difficult to capture with hand-designed controllers, because the right action depends on a fast-changing interaction with another vehicle. Reinforcement learning can acquire such behavior from experience. Agent Mcqueen is a PPO-based racing agent for the F1tenth simulator, trained first to drive alone and then to overtake a competitor.

---

## Why PPO?

PPO (Proximal Policy Optimization) was chosen for the following reasons.

| Algorithm Type            | Examples       | Characteristics                         |
| ------------------------- | -------------- | --------------------------------------- |
| **Value-based**     | DQN, DDQN      | Discrete actions only, sample efficient |
| **Policy Gradient** | REINFORCE, A2C | High variance, continuous actions       |
| **Actor-Critic**    | PPO, SAC, TD3  | Balanced stability and efficiency       |

PPO belongs to the Actor-Critic family, using a learned value function to reduce the variance of the policy gradient:

<div class="mermaid">
flowchart LR
    subgraph AC["Actor-Critic Architecture"]
        STATE[State s]
        ACTOR["Actor<br/>π(a|s)"]
        CRITIC["Critic<br/>V(s)"]
        ACTION[Action a]
        ADV["Advantage<br/>A = R - V(s)"]
    end

    STATE --> ACTOR
    STATE --> CRITIC
    ACTOR --> ACTION
    CRITIC --> ADV
    ADV -->|"Updates"| ACTOR

</div>

### Key PPO Properties

| Property                     | Description                                 |
| ---------------------------- | ------------------------------------------- |
| **On-Policy**          | Uses data from current policy only          |
| **Clipping Mechanism** | Prevents destructively large policy updates |
| **Trust Region**       | Keeps new policy close to old policy        |

The clipping mechanism is what makes PPO stable:

$$
L^{CLIP}(\theta) = \mathbb{E}_t \left[ \min \left( r_t(\theta) A_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) A_t \right) \right]
$$

where $r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)}$ is the probability ratio.

---

## Two-Stage Training Approach

Training directly on competitive racing is unlikely to succeed, since the agent must first learn to drive at all. Training was therefore split into two stages:

| Stage             | Goal                  | Environment       | Agents |
| ----------------- | --------------------- | ----------------- | ------ |
| **Stage 1** | Solo track completion | 450 random tracks | 1      |
| **Stage 2** | Competitive racing    | F1tenth tracks    | 2      |

---

## Stage 1: Solo Track Completion

<video width="70%" controls>
  <source src="/assets/img/Agent-Mcqueen/agent-mcqueen-stage1-render.webm" type="video/webm">
  Your browser does not support the video tag.
</video>

I referenced [this GitHub repository](https://github.com/meraccos/f1tenth_reinforcement_learning) to build Stage 1. Some pieces I needed appeared to be missing — environment initialization among them — so I implemented those myself.

### Observation and Action Space

<div class="mermaid">
flowchart LR
    subgraph OBS["Observation Space"]
        LIDAR["LiDAR Scan<br/>1080 rays"]
        VEL["Linear Velocity<br/>1 value"]
    end

    subgraph AGENT["PPO Agent"]
        ACTOR["Actor Network"]
        CRITIC["Critic Network"]
    end

    subgraph ACT["Action Space"]
        STEER["Steering Angle<br/>[-0.4, 0.4] rad"]
        SPEED["Target Speed<br/>[0, 8] m/s"]
    end

    OBS --> AGENT
    AGENT --> ACT

</div>

| Component          | Specification              |
| ------------------ | -------------------------- |
| **LiDAR**    | 1080 rays, 270° FOV       |
| **Velocity** | Scalar linear velocity     |
| **Steering** | Continuous [-0.4, 0.4] rad |
| **Speed**    | Continuous [0, 8] m/s      |

### Reward Structure

The reward function uses **Frenet coordinates** to measure progress along the track centerline:

<div class="mermaid">
flowchart LR
    POSE["Agent Pose<br/>(x, y, θ)"] --> KD["K-D Tree<br/>Nearest Waypoint"] --> FRENET["Frenet Transform"]

    FRENET --> PROG["Progress<br/>+Δs"]
    FRENET --> LAT["Lateral<br/>-|d|"]
    FRENET --> COL["Collision<br/>-10"]
</div>

| Reward Component            | Formula       | Purpose                    |
| --------------------------- | ------------- | -------------------------- |
| **Progress**          | $+\Delta s$ | Encourage forward movement |
| **Lateral Deviation** | $-\|d\|$    | Stay near centerline       |
| **Collision**         | $-10$       | Avoid crashes              |

### Domain Randomization

<img src="/assets/img/Agent-Mcqueen/stage1-domain-randomization.png" alt="Domain Randomization with Obstacles" style="width:70%;">

Obstacles were placed at random positions on the tracks during training to improve generalization across track layouts.

| Randomization                | Range                             |
| ---------------------------- | --------------------------------- |
| **Track Selection**    | 450 procedurally generated tracks |
| **Obstacle Placement** | Random positions and sizes        |
| **Starting Position**  | Random spawn along centerline     |

### Training Results

<img src="/assets/img/Agent-Mcqueen/stage1-tensorboard.png" alt="Stage 1 Training Progress" style="width:70%;">

| Metric                   | Value                      |
| ------------------------ | -------------------------- |
| **Training Steps** | 10 million                 |
| **Training Time**  | ~13 hours                  |
| **Success Rate**   | 91% (21/23 F1tenth tracks) |

<video width="70%" controls>
  <source src="/assets/img/Agent-Mcqueen/agent-mcqueen-stage1-eval.webm" type="video/webm">
  Your browser does not support the video tag.
</video>

Most of the debugging effort went into environment initialization. The centerline dataset must be reloaded for each randomly selected track; when it was not, the agent learned to drive only on map #50. The reference implementation had been trusted too readily on this point.

---

## Stage 2: Competitive Racing

The initial plan was to switch from PPO to MAPPO (multi-agent PPO). This turned out to be much harder than expected.

### The MARL Challenge

Moving from single-agent to multi-agent RL introduces three well-known difficulties:

<div class="mermaid">
flowchart LR
    subgraph CHALLENGES["MARL Challenges"]
        NS["Non-Stationarity"]
        CA["Credit Assignment"]
        EQ["Equilibrium Selection"]
    end

    subgraph MAPPO["MAPPO"]
        COOP["Cooperative Design"]
    end

    subgraph RACING["Racing"]
        COMP["Zero-Sum Competition"]
    end

    CHALLENGES --> MAPPO
    MAPPO -.->|"Incompatible"| RACING
</div>

| Challenge                       | Description                         | Impact on Racing                       |
| ------------------------------- | ----------------------------------- | -------------------------------------- |
| **Non-Stationarity**      | Other agents change during training | Moving "obstacles" disrupt learning    |
| **Credit Assignment**     | Hard to attribute rewards           | Who caused the collision?              |
| **Equilibrium Selection** | Multiple optimal strategies         | Agents may converge to suboptimal play |

After restructuring the code for zero-sum rewards, joint training of both agents consistently degraded: even when initialized from the Stage 1 models, both agents eventually lost basic driving ability.

### Solution: Residual Learning with Frozen Expert

Joint training was abandoned in favor of a **residual learning** approach against a frozen expert.

<div class="mermaid">
flowchart LR
    subgraph AGENT0["Agent 0 (frozen)"]
        OBS0["LiDAR+Vel"] --> F0["feature_net (frozen)"] --> M0["mean/log_std (frozen)"] --> ACT0["Action"]
    end

    subgraph AGENT1["Agent 1 (trainable)"]
        OBS1["LiDAR+Vel"] --> LN1["lidar_net (frozen)"]
        OPPINFO["Opponent Info"] --> ON["opponent_net (trainable)"]
        LN1 --> OAN["adjustment_net (trainable)"]
        ON --> OAN
        OAN --> ACT1["Action"]
    end
</div>

### Extended Observation Space for Agent 1

| Observation               | Dimension | Description                  |
| ------------------------- | --------- | ---------------------------- |
| **LiDAR Scan**      | 1080      | Same as Stage 1              |
| **Linear Velocity** | 1         | Same as Stage 1              |
| **Delta S (Δs)**   | 1         | Longitudinal gap to opponent |
| **Delta Vs (Δvs)** | 1         | Relative velocity            |
| **Ahead Flag**      | 1         | 1 if ahead, 0 otherwise      |

This lets Agent 1 observe its opponent and learn competitive behavior while the frozen Stage 1 networks preserve its driving ability.

### Stage 2 Reward Structure

<div class="mermaid">
flowchart LR
    subgraph REWARDS["Stage 2 Rewards"]
        BASE["Base Driving<br/>(from Stage 1)"]
        GAP["Gap Reward<br/>+Δs (close the gap)"]
        PASS["Overtake Bonus<br/>+50 (when ahead changes)"]
        SAFE["Safety Penalty<br/>-10 (collision)"]
    end
</div>

| Reward                | Value               | Condition                  |
| --------------------- | ------------------- | -------------------------- |
| **Progress**    | $+\Delta s$       | Always                     |
| **Gap Closing** | $+\Delta s_{gap}$ | When behind opponent       |
| **Overtake**    | $+50$             | Successfully pass opponent |
| **Collision**   | $-10$             | Any collision              |

### Training Configuration

| Parameter                       | Value                |
| ------------------------------- | -------------------- |
| **Frozen Agent Speed**    | 80% of trained speed |
| **Opponent Info Scaling** | 0.01 (very small)    |
| **Training Steps**        | 5 million            |

The opponent features are scaled to very small values. With larger scaling factors the agent over-weighted the opponent and its driving deteriorated.

<video width="70%" controls>
  <source src="/assets/img/Agent-Mcqueen/IMG_6667.mp4" type="video/mp4">
  Your browser does not support the video tag.
</video>

---

## ROS2 Integration

The entire system is integrated with ROS2 for deployment on the F1tenth platform:

<div class="mermaid">
flowchart LR
    GYM["F1tenth Gym / ForzaETH"] <--> BRIDGE["gym_bridge_node"]

    BRIDGE --> SCAN["/scan"]
    BRIDGE --> ODOM["/odom"]

    SCAN --> AGENT["agent_node"]
    ODOM --> AGENT

    AGENT --> CMD["/cmd_vel"] --> BRIDGE
</div>

The same node structure is intended for deployment on the physical F1tenth platform.

---

## Results

### Stage 1 Performance

<video width="70%" controls>
  <source src="/assets/img/Agent-Mcqueen/f1tenth_stage1.webm" type="video/webm">
  Your browser does not support the video tag.
</video>

| Track Dataset                | Success Rate |
| ---------------------------- | ------------ |
| **F1tenth Racetracks** | 91% (21/23)  |
| **Training Tracks**    | 95%+         |

### Stage 2 Performance

<video width="70%" controls>
  <source src="/assets/img/Agent-Mcqueen/f1tenth_overtake.webm" type="video/webm">
  Your browser does not support the video tag.
</video>

The agent learned overtaking behavior while maintaining stable driving.

---

## An Observation Before Training

A script written to render the initial state (to check spawn positions and wall collisions) showed that, with Agent 0 running at 80% speed, Agent 1 — still on its Stage 1 weights, before any Stage 2 training — already overtook it in some episodes. This suggested that overtaking did not need a heavy reward weight, and the opponent-information scaling was reduced further before training proceeded.

---

## Lessons Learned

| Challenge             | Solution                             |
| --------------------- | ------------------------------------ |
| MARL non-stationarity | Freeze one agent                     |
| Credit assignment     | Residual learning architecture       |
| Opponent awareness    | Extended observation with Δs, Δvs  |
| Driving stability     | Small opponent info scaling (0.01)   |
| Generalization        | Domain randomization with 450 tracks |

---

## Limitations

While Agent Mcqueen successfully demonstrates competitive racing behavior, several limitations remain:

| Limitation                              | Description                                                                     | Impact                                          |
| --------------------------------------- | ------------------------------------------------------------------------------- | ----------------------------------------------- |
| **Lack of Strategy Diversity**    | Both agents load the same Stage 1 model, resulting in identical base strategies | Homogeneous racing without unexpected variables |
| **Incomplete Stage 2 Evaluation** | No rigorous quantitative evaluation of overtaking improvement                   | Difficult to measure exact performance gains    |
| **Centerline vs Racing Line**     | Training follows centerline rather than optimal racing trajectory               | Suboptimal lap times and cornering              |
| **Exploration over Mastery**      | Training on 450 random tracks instead of mastering specific tracks              | More like exploration than true racing          |

### Missing Racing Elements

Real racing involves far more complexity than what Agent Mcqueen currently handles:

<div class="mermaid">
flowchart LR
    subgraph CURRENT["Current"]
        DRIVE["Basic Driving"]
        OVERTAKE["Simple Overtaking"]
    end

    subgraph MISSING["Not Modeled"]
        ACCEL["Acceleration"]
        BRAKE["Braking"]
        CORNER["Cornering"]
        DEFEND["Defense"]
    end

    CURRENT -.->|"Future Work"| MISSING
</div>

Professional drivers learn a specific track in detail; Agent Mcqueen is trained on constantly changing tracks, so it generalizes rather than specializes. Braking points, position defense, and corner entry are not modeled.

A potential improvement would be combining traditional control algorithms for low-level vehicle dynamics with RL for high-level strategic decisions. This hierarchical approach could enable more sophisticated competitive behavior.

---

## Conclusion

Agent Mcqueen shows that a competitive racing agent can be trained without a full MARL framework, by combining:

1. **Solid foundation** from Stage 1 solo training
2. **Residual learning** with frozen expert
3. **Carefully designed observations** for opponent awareness
4. **Appropriate reward shaping** for competitive behavior

The result is dynamic overtaking behavior on top of stable driving.

The project also made clear how much of real racing remains unmodeled. The current approach produces functional overtaking but not strategic depth. Future iterations could add racing-line optimization, track-specific training, and a hierarchical control architecture.

The code will be released on GitHub after cleanup.
