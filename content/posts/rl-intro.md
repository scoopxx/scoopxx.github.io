---
title: "Introduction to Reinforcement Learning with Cliff Walking"
date: 2025-04-10
ShowToc: true
TocOpen: true
tags: ["Reinforcement Learning", "Monte Carlo", "Dynamic Programming", "Temporal Difference"]
---

## 1. Getting Started with Reinforcement Learning

I've been fascinated by Reinforcement Learning (RL) for some time, particularly after seeing its recent successes in refining Large Language Models (LLMs) through post-training. However, diving into RL can feel like entering a different world compared to supervised learning. The core concepts, mathematical notation, and terminology—terms like `on-policy`, `off-policy`, `reward`, `value function`, `model-free`, and `agent`—often seem unfamiliar and initially confusing for newcomers like me, who are more accustomed to the standard machine learning vocabulary of `model`, `data`, and `loss function`.

Recently, I dedicated time to exploring RL more deeply. I was pleasantly surprised to find Richard Sutton and Andrew Barto's book, *Reinforcement Learning: An Introduction*, remarkably accessible, even for beginners. This post aims to summarize and share some fundamental concepts I've learned.

In this article, we'll cover the basic elements of RL and explore the `Cliff Walking` problem—a classic RL task—using three foundational approaches:
1.  **Dynamic Programming (DP)**
2.  **Monte Carlo (MC) Methods**
3.  **Temporal-Difference (TD) Learning**

This discussion is primarily based on Chapters 1 through 6 of Sutton & Barto's book.

## 2. What is Reinforcement Learning?

With established paradigms like supervised and unsupervised learning, why do we need RL? How does it stand apart?

**Machine Learning**, as defined by [Wikipedia](https://en.wikipedia.org/wiki/Machine_learning), centers on algorithms that learn patterns from data ($X$) and generalize to new, unseen data. The goal is often to understand the underlying distribution of $X$ or predict outcomes based on it.

-   **Supervised Learning**: Learns from labeled data, typically pairs of input features and corresponding outputs $<x_i, y_i>$. The objective is to learn a mapping function or the conditional probability distribution $p(y | x)$. Examples include image classification (predicting a label $y$ for an image $x$) and sequence prediction tasks like those performed by GPT models (predicting the next token based on previous tokens).

-   **Unsupervised Learning**: Works with unlabeled data, consisting only of input features $x_i$. The goal is to uncover hidden structures, patterns, or distributions within the data itself, without predefined categories or outcomes. Clustering and dimensionality reduction are common examples.

-   **Reinforcement Learning**: Differs significantly because it deals with learning optimal behaviors through interaction within a dynamic environment over time. Unlike supervised learning with static $<x_i, y_i>$ pairs or unsupervised learning focusing solely on $x_i$, RL involves an **agent** learning a **policy** $\\pi(a | s)$. This policy maps situations (**states**, $s$) to **actions** ($a$). The agent learns by receiving feedback (**rewards**) from the **environment** based on its actions, influencing future states and subsequent rewards. The temporal dimension and the agent's active role in shaping its experience are key distinctions.

Consider a self-driving car navigating traffic. The car (agent) must constantly make decisions (accelerate, brake, turn - actions) based on its current situation (traffic, road conditions - state). Supervised learning struggles here because creating labeled data for every conceivable driving scenario is impractical. RL provides a framework for the car to learn effective driving strategies by directly interacting with its environment (simulated or real) and learning from the consequences (rewards or penalties) of its actions.

### 2.1 Core Components of RL

RL problems model the interaction between two main entities:

![Agent Environment Interaction](/images/ae1.png)

-   **Agent**: The learner and decision-maker. It perceives the environment's state and chooses actions.
-   **Environment**: Everything external to the agent with which it interacts. It responds to the agent's actions by changing its state and providing rewards.

The interaction unfolds through a sequence of discrete time steps:

1.  At time $t$, the agent observes the environment's state $s_t$.
2.  Based on $s_t$, the agent selects an action $a_t$ according to its policy.
3.  The environment transitions to a new state $s_{t+1}$ as a result of action $a_t$.
4.  The environment provides a reward $r_{t+1}$ to the agent, indicating the immediate consequence of the transition.

This cycle repeats. The **goal** of the RL agent is typically to learn a policy that maximizes the **cumulative reward** over the long run.

An RL agent's behavior and learning process are often characterized by three key components:

-   **Policy ($\pi$)**: The agent's strategy or behavior function. It dictates how the agent chooses actions based on the observed state. $a_t \sim \pi(\cdot|s_t)$.
-   **Value Function ($V$ or $Q$)**: A prediction of future rewards. It estimates how good it is for the agent to be in a particular state ($V(s)$) or to take a specific action in a state ($Q(s, a)$), assuming the agent follows a certain policy thereafter. Value functions are crucial for evaluating policies and guiding action selection towards maximizing cumulative reward.
-   **Model (Optional)**: The agent's internal representation of how the environment works. A model predicts what the next state and reward will be given a current state and action. $P(s_{t+1}, r_{t+1} | s_t, a_t)$. Agents that use a model are called **model-based**, while those that learn directly from experience without building an explicit model are **model-free**.

#### 2.1.1 Policy ($\pi$)

The policy defines the agent's way of behaving at a given time. Mathematically, it's a mapping from perceived states to actions to be taken when in those states.

-   **Stochastic Policy**: Outputs a probability distribution over actions. The agent samples an action from this distribution.
    $\pi(a | s) = P(A_t = a | S_t = s)$
    This is common during learning to encourage exploration.

-   **Deterministic Policy**: Directly outputs a specific action for each state.
    $a_t = \pi(s_t)$
    Often, the goal is to learn an optimal deterministic policy.

#### 2.1.2 Value Function

In an RL system, the end goal is to find a strategy to win games like Chess or Go. At any time $t$, we want a metric to evaluate the effectiveness of the current policy.

The reward $r_t$ only refers to the immediate reward at time $t$, not the overall winning chance. Thus, we define $V_t$ as the overall reward at time $t$, including the current immediate reward $R_t$ and its expected future reward $V_{t+1}$.

$V_t = r_t + \gamma V_{t+1}$

The gamma ($\gamma$) discount factor is a crucial component in reinforcement learning. It determines the importance of future rewards compared to immediate rewards. A value of $\gamma$ close to 0 makes the agent short-sighted by prioritizing immediate rewards, while a value close to 1 encourages the agent to consider long-term rewards.

The reward at time $t$ is determined by its current state $s_t$, the action taken $a_t$, and its next state $s_{t+1}$.

$r_t = R(s_t, a_t, s_{t+1})$

Assuming we take $T$ steps from time $0$ to $T-1$, these steps form a trajectory $\tau$, and the sum reward is represented as the reward over the trajectory.

$$
\tau = (s_0, a_0, r_0, s_1, a_1, r_1, ..., s_{T-1}, a_{T-1}, r_{T-1}) \newline

R(\tau) = \sum_{t=0}^{T-1}r_t \
$$

#### 2.1.3 Model

A model defines how the environment interacts with the agent, comprising $<S, A, P, R>$:

- $S$: The space of all possible states.
- $A$: The space of all possible actions.
- $P$: The transformation function of how states change, $P(s_{t+1}|s_t, a_t)$.
- $R$: How rewards are calculated for each state, $R(s_t, a_t)$.

If these four elements are known, we can model the interaction without actual interaction, known as **model-based learning**.

In reality, while $S$ and $A$ are often known, the transformation and reward parts are either fully unknown or hard to estimate, so agents need to interact with the real environment to observe states and rewards, known as **model-free learning**.

Comparing the two, **model-free learning** relies on real interaction to get the next state and reward, while **model-based learning** models these without specific interaction.

### 2.2 Optimization of RL

The optimization of an RL task can be divided into two parts:
- **Value Estimation**: Given a strategy $\pi$, evaluate its effectiveness, computing $V_\pi$.
- **Policy Optimization**: Given the value function $V_\pi$, optimize to get a better policy $\pi$.

This is similar to k-Means clustering, where we have two optimization targets and optimize them iteratively to get the optimal answer. In k-means:

- Given the current cluster assignment of each point, compute the optimal centroid. Similar to *value estimation*.
- Given the current optimal centroid, find a better cluster assignment for each point. Similar to *policy optimization*.

Are both steps necessary in RL optimization? The answer is no.

#### 2.2.1 Value-based Agent

An agent can learn only the value function $V_\pi$, maintaining a table mapping $<S, A>$ to $V$. In each step, it picks the action that will maximize the ultimate value. In this type of work, the agent doesn't explicitly have a strategy or policy.

$a_t = \text{argmax}\ V(a | s_t)$

#### 2.2.2 Policy-based Agent

A policy agent directly learns the policy, and each step it outputs the distribution of the next action without knowing the value function.

$a_t \sim P(a|s_t)$

#### 2.2.3 Actor-Critic

An agent can learn both $\pi$ and $V_{\pi}$ as described above.
- **Actor**: Learning of policy $\pi$.
- **Critic**: Learning of value function $V_\pi$.

## 3. Example Problem, Cliff Walking Problem

Let's explore a problem to illustrate the various components and optimization methods in RL. The Cliff Walking problem is a classic reinforcement learning environment introduced in Sutton & Barto’s book, *Reinforcement Learning: An Introduction*.

### 3.1 Problem Statement:
![Cliff Walking Figure](/images/cliff-walking.png)

- The world is represented as a 4×12 grid.
- The start state is at the bottom-left corner $(3, 0)$, and the goal state is at the bottom-right corner $(3, 11)$.
- The bottom row between the start and goal is referred to as the cliff $(3, 1-10)$. If the agent steps into any of these cliff cells, it falls off, receives a large negative reward (e.g., -100), and is reset to the start.
- Each non-terminal move incurs a reward of -1.

Mapping this problem to RL components:
- **Agent**: The robot navigating the grid, aiming to find a path from the start to the goal while maximizing rewards.
- **Environment**: The 4x12 grid world, including transition dynamics and rewards for each move. This environment is fully observable and deterministic.
- **State**: The state space comprises all possible locations of the agent on the grid. There are 48 grid cells, excluding the 10 cliff cells, resulting in 38 possible states.
- **Action**: The action space consists of four possible moves: `UP`, `DOWN`, `LEFT`, `RIGHT`.
- **Reward**: A scalar signal from the environment for each move:
    - -1 for each normal move.
  - -100 if the agent falls into the cliff.
  - 0 upon reaching the goal.
- **Policy**: The strategy the agent should adopt to reach the goal state. It maps states to actions, indicating the direction the agent should take in each grid cell, e.g., `(3, 0) -> MOVE UP`. The objective is to discover such a policy.

### 3.2 Python Implementation
#### 3.2.1 Cliff Walking Environment
The environment can be defined as follows:
```python
class CliffWalk:
    def __init__(self, height, width, start, end, cliff):
        self.height = height
        self.width = width
        self.start = start
        self.end = end
        self.cliff = set(cliff)
        
        self.actions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # up, down, left, right
```
Initialize the environment with specified start, end, and cliff cells:
```python
start = (3, 0)
end = (3, 11)
height = 4
width = 12
cliff_cells = [(3, i) for i in range(1, 11)]

env = CliffWalk(height, width, start, end, cliff_cells)

# Check GitHub for visualization code.
env.render_plot()
```
![Cliff Walking Visualization](/images/CliffWalking-04-08-2025_10_12_AM.png)

#### 3.2.2 Define the Step Function
Next, define the rules and rewards for the agent's actions:
- `self.actions` defines four types of actions, each representing a step in a direction.
- The `step` function takes the current location `(i, j)` and action index, executes it, and returns a tuple representing:
  - The agent's new position after the action.
  - The reward for the current action.

```python
    def step(self, i: int, j: int, a: int) -> tuple[tuple[int, int], int]:
        ni, nj = i + self.actions[a][0], j + self.actions[a][1]
        # Fell into Cliff, get -100 reward, back to start point
        if (ni, nj) in self.cliff:
            return self.start, -100

        if (ni, nj) == self.end:
            return (ni, nj), 0
        
        # Move, get -1 reward
        if 0 <= ni < self.height and 0 <= nj < self.width:
            return (ni, nj), -1
        # Move out of grid, get -1 reward
        return (i, j), -1
```

The movement rules and rewards are:
- The agent receives -1 for each normal move.
- The agent receives -100 if it falls off the cliff.
- The agent receives 0 upon reaching the goal.
- If the agent moves out of the grid, it remains in the same cell and still receives -1.

With the environment set up, we can now solve the problem using three RL methods: dynamic programming, Monte Carlo, and temporal-difference methods. A comparison of these methods will be provided at the end of the article.

## 4. Dynamic Programming Methods
### 4.1 Introduction

Dynamic programming (DP) is a method used in reinforcement learning to solve Markov decision processes (MDPs) by breaking them down into simpler subproblems. It works by iteratively improving the value function, which estimates the expected return of states, and deriving an optimal policy from these values. DP requires a complete model of the environment, including the transition probabilities and reward functions.

### 4.1.1 Markov Property

A Markov decision process (MDP) is defined by five elements:
$$
MDP = <S, A, P, R, \gamma>
$$
- $S$: state space
- $A$: action space
- $P$: **transition probability** from a state and action to its next state, $p(s_{t+1}|s_t, a_t)$
- $R$: **reward function** immediate reward after a transition, $r(s_{t+1}, s_t, a_t)$
- $\gamma$: **discount factor** that weights the importance of future rewards

A decision process has the **Markov Property** if its next state and reward depend only on its current state and action, not the full history. The cliff walking problem satisfies the **Markovian property**.

### 4.1.2 Bellman Optimal Function

The Bellman optimality equation defines the optimal value function as: 
$$
V^*(s) = \max V_\pi(s)
$$

Here, we search for a policy $\pi$ that maximizes the value of state $V$. The resulting policy is our optimal policy.
$$
\pi^*(s) = argmax V_\pi(s)
$$

For each state $s$, we search over its possible actions to maximize the value:
$$
\pi^*(a | s) = 1, if\ a = argmax\ Q(s, a)
$$

Dynamic programming utilizes these equations to iteratively update the value function and improve the policy until convergence, ensuring that the policy becomes optimal.

### 4.2 Value Iteration

Value iteration is a dynamic programming method used to compute the optimal policy and value function for a Markov decision process (MDP). It iteratively updates the value function by considering the expected returns of all possible actions at each state and selecting the action that maximizes this return.

#### 4.2.1 Value Iteration for Cliff Walking
- Initialize each state's value function to 0: $V(s) = 0$
- For each state $s_t$:
  - Search over its possible actions $a_t$, each $a_t$ leads to a new state $s_{t+1}$
  - Update the current state's value function with the action that bears the largest reward: $V(s_t) = \underset{a}{\max}(r_a + V(s_{t+1}))$
- Repeat the previous steps until convergence

#### 4.2.2 Python Implementation of Value Iteration for Cliff Walking
##### 4.2.2.1 One Epoch Update

Below is the implementation for one iteration of value update:
```python
def value_iterate(env, V, gamma=0.9):
    """Run one epoch of value iteration"""
    V_new = V.copy()
    policy = defaultdict(list)

    delta = 0.
    for i in range(env.height):
        for j in range(env.width):
            if (i, j) not in cliff_cells and (i, j) != (end):
                values = []
                for a in range(4):
                    (i_new, j_new), reward= env.step(i, j, a)
                    values.append(V[(i_new, j_new)] * gamma + reward)
                
                max_value = np.max(values)
                best_actions = [a for a, v in enumerate(values) if v == max_value]
                
                V_new[(i, j)] = max_value
                policy[(i, j)] = best_actions
                delta = max(delta, abs(V_new[(i, j)] - V[(i, j)]))

    return V_new, policy, delta
```
- `V` is the value function, its key is a grid location of `(i, j)`, and its value is initialized to 0.
- We iterate over all grid locations that are not cliffs or goal locations. For each grid, we iterate over its 4 actions and pick the action with the largest value to update the current value.
- We use `max(abs(V_new[(i, j)] - V[i, j]))` as the difference between value iterations.

Value iteration does not explicitly optimize the policy; instead, it is learned implicitly by selecting the action that maximizes its next value state.

##### 4.2.2.2 Training

To train the value iteration until convergence:
```python
def value_iteration_train(env, gamma=0.9, tolerance=1e-6):
    progress_data = []
    V = defaultdict(float)
    policy = defaultdict(list)
    progress_data.append({"V": V.copy(), "policy": policy.copy(), "delta": float("inf")})

    while True:
        V, policy, delta = value_iterate(env, V, gamma)
        progress_data.append({"V": V.copy(), "policy": policy.copy(), "delta": delta})
        if delta < tolerance:
            break
    return progress_data
```

- `gamma` is a discount factor that defines the current value of future rewards.
- The training iteration stops when the difference between two value functions is `<tolerance`.
- We return the `V` and `policy` data during training for evaluation purposes.

##### 4.2.2.3 Result and Visualization
We run training using `gamma=0.9`, and it converges in 15 epochs.
```python
dp_progress = value_iteration_train(env, gamma=0.9, tolerance=1e-6)
print(f"Trained {len(dp_progress)} epochs")
```
```text
Trained 15 epochs
```
We visualize both the value function and policy in epochs `1, 7, 14`:
```python
epoches = [1, 7, 14]
for i, ax in enumerate(axs):
    iter = epoches[i // 2]
    if i % 2:
        env.render_plot(policy=dp_progress[iter]['policy'], title = f'Cliff Walking Policy in Epoch {iter}', ax=ax)
    else:
        env.render_plot(value=dp_progress[iter]['V'], title = f'Cliff Walking Value in Epoch {iter}', ax=ax)

plt.tight_layout()
plt.show()
```
From the visualization:
- In epoch 1, the agent learns to avoid the cliff.
- In epoch 7, the agent learns the best actions on the right side of the grid, which are either to take `DOWN` or `RIGHT` actions to reach the goal grid.
- In epoch 14, the value function converges, and the optimal path is to take `UP` from the start and then always take `RIGHT` until close to the goal.

![Value Iteration Visualization](/images/CliffWalking-04-08-2025_01_17_PM.png)

### 4.3 Policy Iteration
In the previous **Value Iteration** method, during iterations, we only updated the value function until convergence. The policy is derived implicitly from the value function. So, can we optimize the policy directly? This leads us to another dynamic programming method in MDPs, **Policy Iteration**.

A policy iteration consists of two parts:
1. **Policy Evaluation**: Given a policy $\pi$, compute its state-value function $V^{\pi}(s)$, which is the expected return of following the policy $\pi$.
$$
V(s_t) = \sum P(s_{t+1} | s_t, \pi) * [r(s_t, \pi, s_{t+1}) + \gamma * V(s_{t+1})]
$$

2. **Policy Improvement**: Update the agent's policy with respect to the current value function.
$$
\pi_{new}(s) = \underset{a}{argmax}\ \sum P(s_{t+1} | s_t, \pi) * [r(s_t, \pi, s_{t+1}) + \gamma * V(s_{t+1})]
$$

#### 4.3.1 Policy Iteration for Cliff Walking
- Step 0, Initialization:
    - Initialize each state's value function to 0: $V(s) = 0$
    - Initialize the policy to take all 4 actions in all states.
- Step 1, **policy evaluation**, for each state $s_t$:
    - Search over its policy's actions $a_t$, each $a_t$ leads to a new state $s_{t+1}$.
    - Update the current state's value function with the mean reward of policy actions. $V(s_t) = \underset{a}{mean}(r_a + \gamma * V(s_{t+1}))$
    - Repeat until the value function converges.
- Step 2, **policy improvement**, for each state $s_t$:
    - Search over the current policy's actions $a_t$ at each $s_t$, compute its value function.
    - Update the policy $\pi(s_t)$ by only keeping actions with the largest value function.
- Repeat steps 1 and 2 until the policy doesn't change.

#### 4.3.2 Python Implementation of Policy Iteration for Cliff Walking
##### 4.3.2.1 Policy Evaluation

Let's first implement the **policy evaluation** function:
```python
def policy_eval(env, V, policy, gamma=0.9, tolerance=1e-4):
    V_new = V.copy()

    while True:
        delta = 0
        for i in range(env.height):
            for j in range(env.width):
                if (i, j) not in env.cliff and (i, j) != env.end:
                    values = []
                    for a in policy[(i, j)]:
                        (i_new, j_new), reward = env.step(i, j, a)
                        values.append(V[(i_new, j_new)] * gamma + reward)
                    
                    V_new[(i, j)] = np.mean(values)
                    delta = max(delta, abs(V_new[(i, j)] - V[(i, j)]))

        if delta < tolerance:
            break
        V = V_new.copy()
    return V_new
```
- This function is very similar to the `value_iterate` function in value iteration, except for one major difference:
    - In `value_iterate`, we compute value functions among all actions and use `np.max(values)` to pick the best action, which means we are implicitly changing the policy using `argmax`.
    - In `policy_eval`, we only iterate actions in the existing policy `policy[(i, j)]`, and use `np.mean` to calculate the expected value function, which means we are only doing evaluation instead of policy optimization here.
- The function returns a new value function `V_new` after convergence.

##### 4.3.2.2 Policy Improvement

Then let's implement the **policy improvement** step:
```python
def policy_improve(env, V, policy, gamma=0.9):
    policy_new = defaultdict(list)

    policy_stable = True
    for i in range(env.height):
        for j in range(env.width):
            current_state = (i, j)
            # Skip cliff cells and the terminal goal state
            if current_state in env.cliff or current_state == env.end:
                continue

            action_values = []
            for a in range(4):
                (i_new, j_new), reward = env.step(i, j, a)
                action_values.append(V[(i_new, j_new)] * gamma + reward)
            
            max_val = np.max(action_values)
            best_actions = [a for a, v in enumerate(action_values) if v == max_val]
            
            if set(best_actions) != set(policy[(i, j)]):
                policy_stable = False
            policy_new[(i, j)] = best_actions
    return policy_new, policy_stable
```
- The `policy_improve` is a one-step optimization, it takes in the current value function `V`, picked the `argmax` action to update the policy, it also takes in the current policy `policy` to compare whether there is any change between the two policies. It returns both the updated policy `policy_new` and a boolean indicating whether the policy changed during optimization.

##### 4.3.2.3 Training

Combining these two sub-steps, we can train using policy iteration:
```python
def policy_iteration_train(env, gamma=0.9, tolerance=1e-6):
    V = defaultdict(float)
    policy = defaultdict(lambda : range(4))    
    progress_data = [{"V": V.copy(), "policy": policy.copy()}]

    while True:
        # Value evaluation
        V = policy_eval(env, V, policy, gamma)
        # Policy improvement
        policy, policy_stable = policy_improve(env, V, policy, gamma)
        progress_data.append({"V": V.copy(), "policy": policy.copy()})

        if policy_stable:
            break
        idx += 1
    return progress_data
```
- For all states, the value function `V` is default to 0, and the policy is default to all 4 actions.
- The training iteration stops until the policy no longer changes.

##### 4.3.2.4 Results and Evaluation
We run training using `gamma=0.9`, and it converges in 6 epochs.
```python
dp_progress = policy_iteration_train(env, gamma=0.9, tolerance=1e-6)
print(f"Trained {len(dp_progress)} epochs")
```
```text
Trained 6 epochs
```
We also visualize the value function and policy in epochs `1, 3, 5`:
- In epoch 1, because the initialized policy includes all actions, this leads to grid in the `i=2` row having a low value function as it has 25% of falling into the cliff and incurring `-100` reward, so the learned policy for most grids is to move upward and avoid the cliff.
- In later epochs, since the policy no longer includes actions that lead to falling off the cliff, the value function improves for all grids, and it learns the optimal path towards the goal grid.

![Policy Iteration Visualization](/images/CliffWalking-04-08-2025_02_49_PM.png)


## 5. Monte-Carlo Methods
### 5.1 Introduction

It's nice that we solved the cliff walking problem with DP methods, and what's more? Remember in DP we assumed full knowledge of the environment - specifically:
- The transition probability: $P(s_{t+1} | s_t, a_t)$
- The reward function : $r(s_{t+1}, a_t, s_t)$

What if the agent is in another environment that itself doesn't know any of such information ahead? Assume the agent was placed in the `start` location, with no knowledge about:
- Where is the `goal` grid, and how to reach it.
- Which grid it will go to if taking an action and what reward it will get.

Then the agent needs to interact with the environment to generate **episodes** (sequence of states, actions, rewards) until it reached the `goal` grid, and learn these information and optimize the policy during the interaction.

Compare the two methods, DP is like a **planner** who knows the full map and computes the best path. Monte-Carlo is like an **explorer** that tries different routes and keeps optimizing the policy.

### 5.2 Interaction Environment
We first need to change the `CliffWalk` environment to mimic an interaction environment.

```python
class CliffWalk:
    def __init__(self, height, width, start, end, cliff):
        ... # Ignore previous codes
        
        self.actions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # up, down, left, right
        self.agent_pos = self.start

    def reset(self):
        self.agent_pos = self.start
        return self.agent_pos

    def step(self, a: int) -> tuple[tuple[int, int], int, bool]:
        i, j = self.agent_pos
        ni, nj = i + self.actions[a][0], j + self.actions[a][1]
        # Fell into Cliff, get -100 reward, back to start point
        if (ni, nj) in self.cliff:
            self.reset()
            return self.start, -100, False

        if (ni, nj) == self.end:
            return (ni, nj), 0, True
        
        # Move, get -1 reward
        if 0 <= ni < self.height and 0 <= nj < self.width:
            self.agent_pos = ni, nj
        else:
            # Move out of grid, get -1 reward
            self.agent_pos = i, j
        done, reward = False, -1
        return self.agent_pos, reward, done
```

Compare the new implementation of `step` function with the previous one:
- The new implementation only takes an `action` index, it tracks the agent's state using `self.agent_pos`
- We are forbidden to compute the state and reward for any $<state, action>$ now. The agent has to reach a specific $s_t$ and take an $a_t$, call `step` to finally get the $s_{t+1}, r_t$ from interaction.
- The `step` function returns a boolean variable `done` indicating whether the agent reached the `goal` grid

### 5.3 Monte-Carlo Simulation
Monte Carlo (MC) methods learn from complete episodes of interaction with the environment. The core idea is to estimate the value of a $<state, action>$ pair by averaging the total return observed after visiting a state across multiple episodes.

#### 5.3.1 $\epsilon$-search algorithm
In RL systems, it's very common to face the exploration vs exploitation dilemma:
- **Exploitation**: Pick the best known action so far (greedy)
- **Exploration**: Try other actions to discover potentially better ones

If the agent always acts greedily, it may get stuck in suboptimal paths, without getting opportunities to discover potentially better paths. The $\epsilon$-search tries to balance this by introducing a random $\epsilon$, in each step:
- **Exploration**: With probability $\epsilon$, choose a random action.
- **Exploitation**: With probability $1 - \epsilon$, choose the action with the highest value: $a = argmax\ Q(s, a)$

#### 5.3.2 Monte-Carlo method in Cliff Walking
- Step 0, Initialization:
    - Initialize a random value function: $Q(s, a)$
    - Initialize an $\epsilon$-greedy policy

- Step 1, Generate episodes:
    - From the `start` state, follow the current policy to generate a full episode until the agent reaches the `goal` grid, we will get a sequence of $<s_t, a_t, r_t>$
- Step 2, Update value function: $Q(s, a)$
    - For each $<s_t, a_t>$ pair in the episode trace, compute its return by $G_t = r\ + \gamma*G_{t+1}$
    - Update $Q(s, a)$ by averaging returns across multiple episodes.
- Step 3, Improve policy:
    - The new policy is the $\epsilon$-greedy policy with the updated value function $Q(s, a)$.
- Repeat step 1-3 until the policy converges.

#### 5.3.3 Python Implementation of MC in Cliff Walking
##### 5.3.3.1 $\epsilon$-greedy search

This function implements the $\epsilon$-search to pick the action, 

```python
def epsilon_greedy(action_values, epsilon):
    if np.random.rand() < epsilon:
        # Random action
        action = np.random.randint(0, 4)
    else:
        # Optimized action
        max_val = np.max(action_values)
        best_actions = [i for i in range(4) if action_values[i] == max_val]
        
        # Random pick among best actions
        action = np.random.choice(best_actions)
    return action
```

##### 5.3.3.2 MC-Simulation
This function simulates 1 episode of MC simulation.
- At the beginning, `env.reset()` sets the agent to the `start` state.
- `Q` is the Value table, with the key being the current location `(i, j)`, and the value being a list of size 4, the value at index `k` represents the value for action `k`.
- The simulation stops after it reaches the `goal` state.

```python
def mc_simulation(env, Q, epsilon=0.1):
    state = env.reset()
    done = False
    curr_eps = epsilon
    episode_data = []

    while not done:
        action = epsilon_greedy(Q[state], epsilon)
        
        next_state, reward, done = env.step_interactive(action)
        episode_data.append((state, action, reward))
        state = next_state

    return episode_data, done
```

##### 5.3.3.3 Value Function Update
```python
def improve_policy(Q, returns, episode_data, gamma=0.9):
    # Compute reward
    visited = set()
    G = 0
    for t in reversed(range(len(episode_data))):
        state_t, action_t, reward_t = episode_data[t]
        G = gamma*G + reward_t

        # First-time update
        if (state_t, action_t) not in visited:
            visited.add((state_t, action_t))
            returns[(state_t, action_t)].append(G)
            Q[state_t][action_t] = np.mean(returns[(state_t, action_t)])

    policy = defaultdict(int)
    for k, v in Q.items():
        policy[k] = np.argmax(v)

    return Q, policy
```
- `returns` is a dictionary, with the key being a `<state, action>` combination, and the value being a list that stores its expected reward in each episode.
- For each episode, we traversed backwards, iteratively computing each state's value using the function: $G_t = r_t + \gamma * G_{t+1}$.

##### 5.3.3.4 Monte-Carlo training
Combining the previous steps, we can train the agent:
```python
def monte_carlo_training(env, num_episodes=1000, gamma=0.9, epsilon=0.1):
    Q = defaultdict(lambda: [0.1] * 4)
    returns = defaultdict(list)
    progress_data = []
    
    for episode in tqdm.tqdm(range(num_episodes)):
        epsilon = max(0.01, epsilon*0.99)
        # Run MC simulation
        episode_data, finished = mc_simulation(env, Q, epsilon)
        if not finished:
            continue

        # Policy improvement
        Q, policy = improve_policy(Q, returns, episode_data, gamma)
        progress_data.append({"Q": copy.deepcopy(Q), "policy": copy.deepcopy(policy), "episode": episode})

    return progress_data
```
- `Q` is initialized by giving equal weights to each action.
- We set $\epsilon$ to decay over episodes, `epsilon = max(0.01, epsilon*0.99)`. In earlier episodes, the agent has no prior knowledge, so we encourage more exploration, then in later episodes focus more on exploitation.

##### 5.3.3.5 Results and Visualizations
We run MC sampling for 5000 episodes:
```python
progress_data = monte_carlo_training(env, num_episodes=5000, gamma=0.9, epsilon=0.3)
```
![Monte-Carlo Visualization](/images/CliffWalking-04-09-2025_04_41_PM.png)

We can see the learned policy is not ideally the optimal shortest path, and the agent is trying to avoid the grid next to the cliff in its first several steps, why? This is explainable:
- The $Q$ value function is averaged over episodes, an early cliff fall trace will drag the **average return** for those cliff-adjacent grids.
- Also, we used $\epsilon$-greedy policy, so even in later episodes when the agent learned a good policy, they will still randomly explore and occasionally fall off the cliff in cliff-adjacent grids.

## 6. Temporal-Difference Methods
### 6.1 Introduction

In previous illustrations of Monte Carlo methods, it estimates the value function using complete episodes. While this is intuitively simple and unbiased, it's very sample-inefficient. The value function updates only happen at the end of episodes, learning can be slow—especially in environments with long or variable episode lengths. 

Temporal-Difference (TD) methods address these limitations by updating value estimates after each time step using bootstrapped predictions, leading to faster and more stable learning.

I found an intuitive way to understand the difference between TD and MC methods are compare this to **Gradient Descent** and **SGD** in neural network optimization, but in the temporal axis, view one step in RL as one batch in supervised model training.
- Gradient descent computes the gradient using the full dataset, while *SGD* compute using only data points in the current batch, update the parameters, then move to the next batch.
- Monte-Carlo methods generate a full episode, backpropagate along the episode to update the value function. While TD methods run one step, use its TD difference to update the value function, then move to the next step.

Then how is TD-difference computed, remember we want to estimate value function using:
$$
V(s_t)\ = r_{t+1} + \gamma\ V(s_{t+1})
$$

So we can bootstrap at $s_t$, execute one more step and compute the value estimates and use it to update the value function:
$$
G(s_t) = r_{t+1} + \gamma\ V(s_{t+1}) 
\newline
\text{TD Error} = G(s_t) - V(s_t) 
\newline
V(s_t) \leftarrow V(s_t) + \alpha \cdot (G(s_t) - V(s_t))
$$
- $\gamma$ is the discount factor
- $\alpha$ is the single step learning rate

### 6.2 SARSA
SARSA is one of the most straightforward ways in TD-methods. The idea is intuitive, using next step's $Q(s_{t+1}, a_{t+1})$ to subtract current step's $Q(s_{t}, a_{t})$ as the TD error, and update value function.
$$
G(s_t) = r_{t+1} + \gamma\ Q(s_{t+1}, a_{t+1}) \newline
Q(s_{t}, a_{t}) \leftarrow Q(s_{t}, a_{t}) + \alpha \cdot (G(s_t) - Q(s_t, a_t))
$$
In every step, we need to get its current state $s_t$, action $a_t$, bootstrap one step forward, get the reward $r_{t+1}$, the new state $s_{t+1}$ and action $a_{t+1}$. In each step, we need the sequence of $<s_t, a_t, r_{t+1}, s_{t+1}, a_{t+1}>$, and this is why this method is called *SARSA*.

#### 6.2.1 SARSA method in Cliff Walking
- Step 0, Initialization:
    - Initialize a random value function: $Q(s, a)$
    - Initialize an $\epsilon$-greedy policy

- Step 1, Bootstrap a step:
    - Agent in state $s_t$ and action $a_t$
    - Bootstrap $a_t$, get the reward $r_{t+1}$ and new state $s_{t+t}$
    - Use the same policy to get the new action $a_{t+1}$
- Step 2, Update value function for the step: $Q(s_t, a_t)$
    - $Q(s_{t}, a_{t}) = Q(s_{t}, a_{t}) + \alpha \cdot (r_{t+1} + \gamma\ Q(s_{t+1}, a_{t+1}) - Q(s_t, a_t))$
- Finish 1 episode by repeated running step 1-2 until the agent reached `goal` state.
- Run above algorithm multiple times until the policy converges.

This looks very similar to SARSA, the only difference is:
- We no longer need to sample $s_{t+1}$ in each step.
- We used $max\ Q(s_{t+1})$ instead of $Q(s_{t+1}, a_{t+1})$ for value update.

#### 6.2.2 Python Implementation of SARSA
##### 6.1.2.1 SARSA
```python
def sarsa_one_epoch(env, Q, gamma=0.9, epsilon=0.1, alpha=0.1):
    # Get init action
    state = env.reset()
    action = epsilon_greedy(Q[state], epsilon)
    done = False

    while not done:
        next_state, reward, done = env.step_interactive(action)

        # Sample next action
        next_action = epsilon_greedy(Q[next_state], epsilon)

        # TD-Update current function
        Q[state][action] += alpha*(reward + gamma*Q[next_state][next_action] - Q[state][action])
        
        state, action = next_state, next_action

    policy = defaultdict(int)
    for k, v in Q.items():
        best_actions = [a for a, i in enumerate(v) if i == np.max(v)]
        
        # Random pick among best actions
        policy[k] = best_actions
    return Q, policy
```
- We use the same $\epsilon$-greedy search to get the action
- The $Q$ value function is updated within each step of the epoch, this is called **1-step SARSA**, alternatively, we can also update $Q$ value function every fixed number of steps, which is called **$n$-step SARSA**

To train multiple episodes: 
```python
def td_sarsa_training(env, num_episodes=1000, gamma=0.9, epsilon=0.1, alpha=0.1):
    # Key: position, Value: value for of each action
    Q = defaultdict(lambda: np.random.rand(4) * 0.01)
    progress_data = []
    
    for episode in tqdm.tqdm(range(num_episodes)):
        epsilon = max(0.01, epsilon*0.95)
        Q, policy = sarsa_one_epoch(env, Q, gamma, epsilon, alpha)
        progress_data.append({"Q": Q.copy(), "policy": policy.copy(), "episode": episode})
    return progress_data
```
- Similar to that of Monte-Carlo methods, we used a decaying $\epsilon$ for action search, to encourage more exploration in early episodes and more exploitation in later episodes.

##### 6.1.2.2 Visualization and Result
We train SARSA for 10000 episodes, and visualize the result
```python
sarsa_progress = td_sarsa_training(env, num_episodes=10000, gamma=0.9, epsilon=0.1, alpha=0.2)
```
![SARSA Training Visualization](/images/CliffWalking-04-09-2025_09_47_PM.png)

The learned policy in `epoch=9999` is similar to that learned from MC methods, that it tries to avoid the cliff-adjacent grids, the reasoning is also similar:

- The agent used $\epsilon$-greedy search, so even the agent learned a good policy, its exploration nature may still lead to fall off in cliff-adjacent grids. So the agent learned to walk far away from the cliff, taking the constant cost of extra `-1` reward, to avoid a potential `-100` reward. 

### 6.3 Q-Learning
Let's recap the SARSA algorithm again, it used $\epsilon$-greedy search on $Q$ value functions for two purposes:
1. **Planning**: Decide the action $a_{t+1}$ of the next step
2. **Policy Update**: use the actual action $a_{t+1}$ to update the value function.

So this policy has to incorporate a trade-off between exploration and exploitation. What if we have two policies:
1. One **Behavior Policy** that focuses on exploration, it decides the interaction with the environment.
2. One **Target Policy** that focuses on exploitation, it doesn't do interaction, but focuses on learning from previous interactions.

Then the behavior policy can be more aggressive to keep exploring risky areas, without fearing these risky behaviors affect its value function. On the other hand, its target policy focuses on greedily learning the optimal policy, without being penalized by random exploratory behaviors.

This new method is called Q-Learning, the difference between SARSA and Q-Learning can also formalize as **On-Policy** vs **Off-Policy**:
- **On-Policy** learns the value of the policy it is actually using to make decisions.
- **Off-Policy** Learns the value of a different policy than the one it is currently using to make decisions.

In SARSA, we used the actual value $Q(s_{t+1}, a_{t+1})$ to update the value function:

$$
G(s_t) = r_{t+1} + \gamma\ Q(s_{t+1}, a_{t+1}) \newline
Q(s_{t}, a_{t}) \leftarrow Q(s_{t}, a_{t}) + \alpha \cdot (G(s_t) - Q(s_t, a_t))
$$

In Q-Learning, we used the theoretical optimal next action instead of the actual next action for updates:
$$
G(s_t) = r_{t+1} + \gamma\ \underset{a}{max}\ Q(s_{t+1}) \newline
Q(s_{t}, a_{t}) \leftarrow Q(s_{t}, a_{t}) + \alpha \cdot (G(s_t) - Q(s_t, a_t))
$$

#### 6.3.1 Q-Learning in Cliff Walking
- Step 0, Initialization:
    - Initialize a random value function: $Q(s, a)$
    - Initialize an $\epsilon$-greedy policy

- Step 1, Bootstrap a step:
    - Agent in state $s_t$ and action $a_t$
    - Bootstrap $a_t$, get the reward $r_{t+1}$ and new state $s_{t+t}$
- Step 2, Update value function for the step: $Q(s_t, a_t)$
    - $Q(s_{t}, a_{t}) = Q(s_{t}, a_{t}) + \alpha \cdot (r_{t+1} + \gamma\ max\ Q(s_{t+1}, a) - Q(s_t, a_t))$
- Finish 1 episode by repeated running step 1-2 until the agent reached `goal` state.
- Run above algorithm multiple times until the policy converges.

This looks very similar to SARSA, the only difference is:
- We no longer need to sample $s_{t+1}$ in each step.
- We used $max\ Q(s_{t+1})$ instead of $Q(s_{t+1}, a_{t+1})$ for value update.

#### 6.3.2 Python Implementation of Q-Learning
##### 6.3.2.1 Q-Learning
```python
def q_learning_one_epoch(env, Q, gamma=0.9, epsilon=0.1, alpha=0.1):
    # Get init action
    state = env.reset()
    done = False

    while not done:
        action = theta_greedy_action(Q, state, epsilon)
        next_state, reward, done = env.step_interactive(action)

        # TD-Update current function
        Q[state][action] += alpha*(reward + gamma*max(Q[next_state]) - Q[state][action])
        
        state = next_state

    policy = defaultdict(int)
    for k, v in Q.items():
        best_actions = [a for a, i in enumerate(v) if i == np.max(v)]
        policy[k] = best_actions
    return Q, policy
```
- We no longer computed `next_action` in each step.
- `Q` is updated using `max(Q[next_state])`.

```python 
def q_learning_training(env, num_episodes=1000, gamma=0.9, epsilon=0.1, alpha=0.1):
    # Key: position, Value: value for of each action
    Q = defaultdict(lambda: np.random.rand(4) * 0.01)
    progress_data = []
    
    for episode in tqdm.tqdm(range(num_episodes)):
        epsilon = max(0.01, epsilon*0.95)
        Q, policy = q_learning_one_epoch(env, Q, gamma, epsilon, alpha)
        progress_data.append({"Q": Q.copy(), "policy": policy.copy(), "episode": episode})
    return progress_data
```

##### 6.3.2.2 Visualization and Result
Q-Learning converges faster than SARSA, we only trained 200 episodes.
```python
q_learning_progress = q_learning_training(env, num_episodes=200, gamma=1.0, epsilon=0.1, alpha=0.2)
```
![Q-Learning Training Visualization](/images/CliffWalking-04-09-2025_11_18_PM.png)

While SARSA found a **safe path** under randomness of $\epsilon$-greedy, Q-Learning found the shortest optimal path- It learns to hug off the cliff!

## 7. Summary 

- **Model-Based vs. Model-Free**: Dynamic Programming (DP) is a model-based approach requiring full knowledge of the environment's dynamics. In contrast, Monte Carlo (MC) and Temporal-Difference (TD) methods are model-free, learning directly from interactions.

- **On-Policy vs. Off-Policy**: SARSA is an on-policy method, meaning it evaluates and improves the policy that is used to make decisions. Q-Learning is off-policy, as it evaluates the optimal policy independently of the agent's actions.

- **Exploration vs. Exploitation**: Reinforcement Learning (RL) faces the dilemma of choosing between exploring new actions to discover their effects and exploiting known actions to maximize reward. Techniques like $\epsilon$-greedy balance this trade-off.

- **Bellman Equations**: Central to DP and TD methods, these equations provide the foundation for iteratively improving value functions and policies.

- **Comparison of DP, MC, and TD**:
  - **DP**: Requires a complete model of the environment and is computationally intensive but provides exact solutions.
  - **MC**: Sample-based and learns from complete episodes, making it intuitive but potentially inefficient for long episodes.
  - **TD**: Bootstrap-based, updating estimates at each step, offering a balance between DP's precision and MC's sampling.
  - **Intuitive Comparison**: If Temporal-Difference methods require broader updates, they resemble Dynamic Programming, as DP considers all states for updates. Conversely, if TD methods require deeper updates, they resemble Monte Carlo methods, which rely on complete episodes for learning.


### 7.1. What's More

In the cliff walking problem, the state space is relatively small, consisting of only about 20 grid cells. This allows us to use tabular methods, where we store the value function, either $V(s)$ or $Q(s, a)$, in a table. However, in more complex problems like self-driving cars or Atari games, the state space becomes significantly larger. This complexity necessitates the use of parametrized methods, which employ function approximations, such as neural networks, to estimate value functions or policies.

These advanced methods enable RL algorithms to handle complex, high-dimensional environments, paving the way for techniques like Deep Q-Learning and Policy Gradient methods. Such approaches are often employed in the post-training of large language models (LLMs).

## 8. References
- Sutton, R. S., & Barto, A. G. (2018). [*Reinforcement Learning: An Introduction (2nd ed.)*](http://www.incompleteideas.net/book/the-book-2nd.html). MIT Press.
- Wang, Q., Yang, Y., & Jiang, J. (2022). [*Easy RL: Reinforcement Learning Tutorial*](https://github.com/datawhalechina/easy-rl). Posts & Telecom Press.

