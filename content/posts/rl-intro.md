---
title: "Introduction to Reinforcement Learning"
date: 2025-03-03
draft: true
ShowToc: true
TocOpen: true
tags: ["Reinforcement Learning"]
---

## Intuition
I've been having interests in Reinforcement Learning(RL) for a while, especially after ChatGPT when everyone was talking about `RLHF`, but often felt it's too daunting for me to do some reading and dig into it. Recently I finally found some spare time to go through it, and fortunately the famous Richard Sutton's \<Reinforcement Learning: an Introduction\> book is surprisingly intuitive even for RL beginners. 

However, the mathematical notions used in RL is quite different from those used in supervised learning, and may felt weird or confusing for beginners just like me. As a reference, I will share and summarize my reading notes here.

## Introduction of Reinforcement Learning
So first of all, why is RL needed since we already have supervised/unsupervised learning? And how is RL different from the other methods?

Wikipedia's definition of the term [**Machine Learning**](https://en.wikipedia.org/wiki/Machine_learning) is: *statistical algorithms that can learn from data and generaize to unseen data.* Say we have data $X$, and its training data sample $x_i$, we are trying to learning the distribution about $X$.

- In supervised learning, the data is pair of $<x_i, y_i>$, so we are trying to learn distribution $p(y_ | x_)$. For example, in image classification, given a image $x$, trying to predict its label $y$. If we viewed GPT as a special case of supervised learning, it can be seen as given a text sequence's previous tokens, predict the next token.

- In reinforcement learning, the $x$ is no longer static, or known before training. If in supervised learning the goal is to learn a mapping from $x_i$ to $y_i$, in RL it became to learn the mapping from situations to actions, both can only get through direct interaction with the environment. 


### Components of RL 
We defined reinforcement learning as a method to model the interaction in a game/strategy etc. There are two entities in such RL system:

![Agent Environment Interaction](/images/ae1.png)

- $Agent$ is the learner and decision-maker.
- $Environment$ is everything outside the agent that it interacts with. 

Then there are three types of interactions between agent and environment:
- $Action$ is an action the agent takes to interact with the environment.
- $State$ is a representation of the environment, which is changed by action.
- $Reward$ can be seen as the environment's corresponding response to the agent based on its state.

A sequence of interactions is:
- In $t$, the environment's state is $S_t$, it exhibits reward $R_t$
- Based on $S_t$ and $R_t$, the agent takes action $A_t$
- As a result of $A_t$, the state changed to $S_{t+1}$, and exhibits reward $R_{t+1}$

The goal of an RL system is to learn a strategy so the agent makes the best action in each step to maximize its rewards. We can define three components of such agent:

- $Policy$ is the strategy the agent used to determine its next action.
- $Value\ Function$ is a function the agent used to evaluate its current state
- $Model$ is the way the environment interacts with the agent, it defined how the environment works.

#### 2.1 Policy
Policy is a mapping from current state $s_t$ to action $a_t$. It can be deterministic or stochastic:

In **Stochastic Policy**, the result is a ditribution, in which the agent sample its action from.

$\pi(a_{t} | s_{t}) = p(a = a_t | s=s_t) $

In **Deterministic Policy**, the result is a action, for a given state, the agent will surely execute a specific action.

$a_{t} = argmax\ \pi(a|s_t)$

#### 2.2 Value Function

In an RL system, the end goal is to find a strategy to win the game, like in Chess or Go. At any timestamp t, we wants to have a metric to evaluate the effectiveness of current strategy.

The reward $R_t$ defined above only refers to immediate reward at timestamp $t$, but not the overall winning chance. So we defined $V_t$ as the overall reward at timestamp $t$, which includes the current immediate reward $R_t$, and its expected future reward $V_{t+1}$.

$V_t = R_t + \gamma V_{t+1}$

So the reward at timestamp $t$ is decided by its current state $s_t$, its taken action $a_t$, and its next state $s_{t+1}$.

$r_t = R(s_t, a_t, s_{t+1})$

Assume we take $T$ steps from timestamp $0$ to $T-1$, all these steps form a trajectory $\tau$, the sum reward can be represented as the reward over the trajectory.

$$
\tau = (s_0, a_0, r_0, s_1, a_1, r_1, ..., s_{T-1}, a_{T-1}, r_{T-1}) \

R(\tau) = \sum_{t=0}^{T-1}r_t \
$$

#### 2.3 Model
A $model$ defined how the environment interacts with the agent. A model is comprised of $<S, A, P, R>$

- $S$, the space of all possible states
- $A$, the space of all possible actions
- $P$, transformation function of how state changed, $P(s_{t+1}|s_t, a_t)$
- $R$, how reward is calculated for each state, $R(s_t, a_t)$

If all these 4 elements are known, we can easily model the interaction without actual interaction, this is called *model based learning*.

In reallity, while $S$ and $A$ is often known, the transformation and reward part is either fully unknown or hard to estimate, so agent need to interact with real environment to observe the state and reward, this is called *model-free learning*. 

Comparing the two, *model-free learning* relied on real interaction to get the next state and reward, while *model based learning* modeled these without specific interaction. 

### Optimization of RL
The optimization of a RL task can be divided into two parsts:
- **Value estimation**: given a strategy $\pi$, evaluate the effectiveness of how does the strategy work, this is computing $V_\pi$.
- **Policy optimization**: given the value function $V_\pi$, optimize to get a better policy $\pi$.

In essence this is very similar to k-Means clustering that we have two optimization targets, and we opmitize them iteratively to get the optimial answer. In k-means,

- Given the current cluster assignment of each point, we compute the optimal centroid. Similar to *value estimation*.
- Given the current optimial centroid, we find better cluster assignment for each point. Similar to *policy optimization*.

But are the two steps both necessary in RL optimization? Answer is No. 

### 3.1 Value-based agent
An agent can only learn the value function $V_\pi$, it can maintain a table of mapping from $<S, A>$ to $V$, then in each step, it picked the action that will maximize the ultimate value. In this type of work, the agent doesn't explicitely have a strategy or policy.

$ a_t = argmax\ V(a | s_t)$

### 3.2 Policy-based agent
A policy agent directly learns the policy, and each step it directly outputs the distribution of next action without knowing value function.

$ a_t \sim P(a|s_t)$

### 3.3 Actor-Critic
An agent can learn both $\pi$ and $V_{\pi}$ as we described above. 
- Actor refers to learning of policy $\pi$
- Critic refers to learning of value function $V_\pi$

## 4. Example Problem, Cliff Walking Problem

Now let's work on a problem together to walk through all the different pieces and optimization methods in RL. The Cliff Walking problem is a classic reinforcement learning (RL) environment introduced in Sutton & Barto’s book, “Reinforcement Learning: An Introduction.” 

### 4.1 Problem Statement:
- The world is represented as a 4×12 grid world.
- The start state is at the bottom-left corner $(3, 0)$, and the goal state is at the bottom-right corner $(3, 11)$.
- The bottom row between the start and goal is called the cliff $(3, 1-10)$ — if the agent steps into any of these cliff cells, it falls off, receives a large negative reward (e.g., -100), and is reset to the start.
- Each non-terminal move incurs a reward of -1.

Now let's map this problem statement to different components of RL system.
- $Agent$, the robot that exists in the grid world, the agent needs to find a path from start position to end position to collect the rewards.
- $Environment$, the 4x12 grid world, as well as the transition and reward for each move. This environment is fully observable and deterministic.
- $State$, the state space is all the possible locations of the agent on the grid, there are 48 grids and minus the 10 cliff grids, there are 38 possible grids.
- $Action$, the action space is all the possible moves the agent can take. There are 4 possible actions, `UP`, `DOWN`, `LEFT`, `RIGHT`.
- $Reward$, a scalar signal from the environment for the agent's each move. We can define it as:
    - -1 for each normal move.
    - -100 if the agent fells into the cliff.
    - 0 upon reaching the goal location.
- $Policy$, the strategy the agent should take to reach the goal state. Here it should be a mapping from $state$ to $action$, here it tells what direction should the agent take in each grid cell. For example, `(3, 0) -> MOVE UP`. The end goal for this problem is to find such a policy.

#### 4.1.1 Define the Environemnt
We can define the  environment as following.  
```python
class CliffWalk:
    def __init__(self, height, width, start, end, cliff):
        self.height = height
        self.width = width
        self.start = start
        self.end = end
        self.cliff = set(cliff)
        
        self.actions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # up, down, left, right
        self.reset()

    def reset(self):
        self.agent_pos = self.start
        return self.agent_pos
```
Then we can define the cells for start, end and cliff cells, and initialize a environment.
```python
start = (3, 0)
end = (3, 11)
height = 4
width = 12
cliff_cells = [(3, i) for i in range(1, 11)]

env = CliffWalk(height, width, start, end, cliff_cells)

# Check github for visualization code.
env.render_plot()
```
![Cliff Walking Visualization](/images/CliffWalking-04-08-2025_10_12_AM.png)

#### 4.1.2 Define the Step function
Next, we define rules and rewards for the agent's action. 

- In `self.actions` we defined 4 type of actions, each representing walking 1 step in the direction.
- The `step` function takes in a current location `(i, j)` and `action` index, execute it and returns a tuple representing:
    - The agents new position after the action
    - Reward of current action


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

The moving rules and rewards are:
- Agent received -1 for each normal move.
- Agent received -100 if fell off the cliff.
- Agent received 0 if reaching the goal.
- If the agent moved out of the grid, it stayed still in the same grid and still received -1.


Now we introduced the environment setup of the cliff walking problem, now let's try to solve it with three classes of RL methods, which is dynamic programming, monte-carlo and temporal-difference methods. Comparison between these three methods will be given at the end of the article.
## 5. Dynamic Programming Methods
### 5.1 Introduction
### 5.1.1 Markov Property
A Markov decision process(MDP) is defined by 5 elements:
$$
MDP = <S, A, P, R, \gamma>
$$
- $S$: state space
- $A$: action space
- $P$: **transition probability** from a state and action to its next state, $p(s_{t+1}|st, at)$
- $R$: **reward function** immediate reward after a transition, $r(s_{t+1},st, at)$
- $\gamma$: **discount factor** that weights the importance of future reward.

We define a deicision process has **Markov Property** if its next state and reward only depend on its current state and action, not the full history. We can see the cliff walking problem suffices the **markovian propterty**.

### 5.1.2 Bellman Optimal Function
We define the optimal value function is:
$$
V^{*}(s) = maxV_\pi(s)
$$
Here we searched a policy $\pi$ to maximize the state $V$, the result policy is our optimal policy.
$$
\pi^{*}(s) = argmaxV_\pi(s)
$$
For each state $s$, we searched over its possible actions to maximize the value:
$$
\pi^{*}(a | s) = 1, a = argmaxQ^*(s, a)
$$

### 5.2 Value Iteration

#### 5.2.1 Value Iteration for Cliff Walking
- Initialize each state's value function to 0: $V(s) = 0$
- For each state $s_t$
    - seach over its possible actions $a_t$, each $a_t$ leads to a new state $s_{t+1}$
    - update current state's value function with the action that beares largest reward. $V(s_t) = \underset{a}{max}(r_a + V(s_{t+1}))$
- Repeat the previous steps until convergence

#### 5.2.2 Python Implementation of Value Iteration for Cliff Walking
Below we defined one iteration for value update:
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
- `V` is the value function, its key is a gird location of `(i, j)`, value is initlized to 0
- We iterate over all grid locations that's not a cliff or goal location, for each grid, we iterated over its 4 actions, and pick the action with the largest value to update current value.
- We used `max(abs(V_new[(i, j)] - V[i, j]))` as the difference between value iteartions.
- Value iteration does not explicitly optimize the policy, instead it's learnt implicitily by selecting over an action that maximized its next value state. 

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

- `gamma` is a discounted factor that defined the future reward's current value
- The training iteration stoped until the difference between two value functions are `<tolerance`.
- We returned the `V` and `policy` data during training for evaluation purpose

#### 5.2.3 Result and Visualization
We run training using `gamma=0.9`, it converges in 15 epoches
```python
dp_progress = value_iteration_train(env, gamma=0.9, tolerance=1e-6)
print(f"Trained {len(dp_progress)} epoches")
```
```text
Trained 15 epoches
```
We visulize both the value function and policy in epoch `1, 7, 14`
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
From the visuliaztion:
- In epoch 1, the agent learnt to avoid the cliff
- In epoch 7, the agent learnt the best actions on right side of the grid, which is either take `DOWN` or `RIGHT` action to reach the goal grid.
- In eppch 14, the value function converges, which the optimal path now is to take `UP` from start then always take `RIGHT` until close to the goal.

![Value Iteration Visualization](/images/CliffWalking-04-08-2025_01_17_PM.png)

### 5.3 Policy Iteration
In previous **Value Iteration** method, during iterations we only updated the value function until convergence, the policy is derived implicitely from the value function. So can we optimize the policy directly? This comes into another dynamic programming method in MDPs, **Policy Iteration**.

A policy iteration consists of two parts:
1. **Policy Evaluation**, given a policy $\pi$, compute its state-value function $V^{\pi}(s)$, which is the expected return of following the policy $\pi$.
$$
V(s_t) = \sum P(s_{t+1} | s_t, \pi) * [r(s_t, \pi, s_{t+1}) + \gamma * V(s_{t+1})]
$$

2. **Policy Improvment**, update the agent's policy respect to the current value function.
$$
\pi_{new}(s) = \underset{a}{argmax}\ \sum P(s_{t+1} | s_t, \pi) * [r(s_t, \pi, s_{t+1}) + \gamma * V(s_{t+1})]
$$

#### 5.3.1 Policy Iteration for Cliff Walking
- Initialization:
    - Intialize each state's value function to 0: $V(s) = 0$
    - Initialize policy to take all 4 actions in all states.
- Step 1, **policy evaluation**, for each state $s_t$
    - Search over its policy's actions $a_t$  each $a_t$ leads to a new state $s_{t+1}$
    - Update current state's value function with the mean reward of policy actions. $V(s_t) = \underset{a}{mean}(r_a + \gamma * V(s_{t+1}))$
    - Repeat until the value function convergent.
- Step 2, **policy improvement**, for each state $s_t$
    - Seach over current policy's actions $a_t$ at each $s_t$, compute its value function.
    - Update the policy $\pi(s_t)$ by only keeping actions with the largest value function.
- Repeat step 1 and 2 until the policy doens't change.

#### 5.3.2 Python Implementation of Policy Iteration for Cliff Walking
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
- This function is very similar to the `value_iterate` function in value interation, except one major difference:
    - In `value_iterate`, we compute value functions among all actions and used `np.max(values)` to pick the best action, which means we are implicitely changing the policy using `argmax`.
    - In `policy_eval`, we only iterate actions in existing policy `policy[(i, j)]`, and used `np.mean` to calculate the expected value function, which means we are only doing evaluation instead of policy optimization here.
- The function returned a new value function `V_new` after convergence.

Then let's implement the **policy improvement** step:
```python
def policy_improve(env, V, policy, gamma=0.9):
    policy_new = defaultdict(list)

    policy_stable = True
    for i in range(env.height):
        for j in range(env.width):
            if (i, j) not in env.cliff and (i, j) != env.end:
                values = []
                for a in range(4):
                    (i_new, j_new), reward = env.step(i, j, a)
                    values.append(V[(i_new, j_new)] * gamma + reward)
                
                max_val = np.max(values)
                best_actions = [a for a, v in enumerate(values) if v == max_val]
                if set(best_actions) != set(policy[(i, j)]):
                    policy_stable = False
                policy_new[(i, j)] = best_actions
    return policy_new, policy_stable
```
- The `policy_improve` is a one step optimization, it takes in the current value function `V`, picked the `argmax` action to update the policy, it also takes in current policy `policy` to compare whether there is any changes between the two policy. It returns both the updated policy `policy_new` and a boolean indicated whether the policy changed during optimization.

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
- For all states, value function `V` is default to 0, policy is default to all 4 actions.
- The training iteration stoped until the the policy no longer changed.

#### 5.3.3 Results and Evaluation
We run training using gamma=0.9, it converges in 6 epoches
```python
dp_progress = policy_iteration_train(env, gamma=0.9, tolerance=1e-6)
print(f"Trained {len(dp_progress)} epoches")
```
```text
Trained 6 epoches
```
We also visualize the value function and policy in epoch `1, 3, 5`:
- In epoch 1, because the initialized policy includes all actions, this leads to grid in the `i=2` row has a low value function as it has 25% of falling into the cliff and incur `-100` reward, so the learnt policy for most grids is to move upward and avoid the cliff.
- In later epoches, since the policy no longer includes actions that leads to fall off the cliff, the value function improved for all grids, also it learnt the optimal path towards the goal grid.

![Policy Iteration Visualization](/images/CliffWalking-04-08-2025_02_49_PM.png)


## 6. Monte-Carlo Methods
It's nice that we solved the cliff walking problem with DP methods, and what's more? Remember in DP we assumed full knowledge of the environment - specifically:
- The transition probability: $P(s_{t+1} | s_t, a_t)$
- The reward function : $r(s_{t+1}, a_t, s_t)$

What if the agent is in another environment that itself doesn't know any of such information ahead? Assume the agent was placed in the `start` location, with no knowledge about:
- where is the `goal` grid, and how to reach it.
- Which grid it will go to if taking an action and what reward it will get.

Then the agent need to interact with the environment to generate **episodes** (sequence of states, actions, rewards) until it reached the `goal` grid, and learn these information and otpimize the policy during the interaction.

Compare the two methods, DP is like a **planner** who knows the full map and compute the best path. Monte-Carlo is like an **explorer** that tries different routes and keep optimizing the policy.

### 6.1 Interaction Environment
We first need to chang the `CliffWalk` environment to mimic an interaction environment.

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

        # Move, get -1 reward
        if 0 <= ni < self.height and 0 <= nj < self.width:
            self.agent_pos = ni, nj
        else:
            # Move out of grid, get -1 reward
            self.agent_pos = i, j
        done, reward = False, -1
        if self.agent_pos == self.end:
            done, reward = True, 0
        return self.agent_pos, reward, done
```

Compare the new implementation of `step` function with previous one:
- The new implementation only takes an `action` index, it tracks the agent's state using `self.agent_pos`
- We are forbidden to compute the state and reward for any $<state, action>$ now. The agent has to reach to a specific $s_t$ and take an $a_t$, call `step` to finally get the $s_{t+1}, r_t$ from interaction.
- The `step` function returns a boolean varaible `done` indicating whether the agent reached the `goal` grid

### 6.2 Monte-Carlo Simulation
Monte Carlo (MC) methods learn from complete episodes of interaction with the environment. The core idea is to estimate the value of a $<state, action>$ pair by averaging the total return oberseved after visiting a state across multiple episodes.

### 6.2.1 $\epsilon$-search algorithm
In RL system, it's very common to face the exploration vs exploitation dillema:
- **Exploitation**: Pick the best known action so far (greedy)
- **Exploration**: Try other actions to discover potentially better ones

If the agent always acts greedily, it may get stuck in suboptimal path, without getting oppourtunity to discover potential better paths. The $\epsilon$-search try to balance this by introducing a random $\epsilon$, in each step:
- **Exploration**: With pobability $\epsilon$, choose a random action.
- **Exploitation**: With probability $1 - \epsilon$, choose action with highest value: $a = argmax\ Q(s, a)$




### 6.2.2 Monte-Carlo method in Cliff Walking
- Step 0, Initilization:
    - Initialize a random value function: $Q(s, a)$
    - Initialize an $\epsilon$-greedy policy

- Step 1, Generate episodes:
    - From the `start` state, follow current policy to generate full episode until the agent reached `goal` grid, we will get a sequence of $<s_t, a_t, r_t>$
- Step 2, Update value function: $Q(s, a)$
    - For each $<s_t, a_t>$ pair in episode trace, compute its return by $G_t = r\ + \gamma*G_{t+1}$
    - Update $Q(s, a)$ by averaging returns across multiple episodes.
- Step 3, Improve policy:
    - The new policy is the $\epsilon$-greedy policy with updated value function $Q(s, a)$.
- Repeat step 1-3 until the policy converges.

### 6.3 Python Implementation of MC in Cliff Walking
#### 6.3.1 $\epsilon$-greedy search

This function implements the $\epsilon$-search to pick the action, 

```python
def epsilon_greedy(action_values, epsilon):
    if np.random.rand() < epsilon:
        # Random action
        action = np.random.randint(0, 4)
    else:
        # Optimzed action
        max_val = np.max(action_values)
        best_actions = [i for i in range(4) if action_values[i] == max_val]
        # Random pick among best actions
        action = np.random.choice(best_actions)
    return action
```

#### 6.3.2 MC-Simulation
This function simulates 1 episode of MC simulation.
- At the beginning, `env.reset()` set the agent to `start` state.
- `Q` is the Value table, with key is the current location `(i, j)`, value is a list of size 4, the value at index `k` represents value for action `k`.
- The simulation stop after it reached the `goal` state.
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

#### 6.3.3 Value Function Update
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
- `returns` is a dictionary, with key being a `<state, action>` combination, value being a list that stored its expected reward in each episode.
- For each episode, we traversed backwards, iteratively computing each state's value using function: $G_t = r_t + \gamma * G_{t+1}$.

#### 6.3.4 Monte-Carlo training
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
- `Q` is initilized by giving equal weights to each action.
- We set $\epsilon$ to decay over episodes, `epsilon = max(0.01, epsilon*0.99)`. In earlier epoches, the agent has no prior knowledge, so we enough more exploration, then in later epoches focus more on exploitation.

#### 6.3.5 Results and Visualizations
We run MC sampling for 5000 episodes:
```python
progress_data = monte_carlo_training(env, num_episodes=5000, gamma=0.9, epsilon=0.3)
```
![Monte-Carlo Visualization](/images/CliffWalking-04-09-2025_04_41_PM.png)

We can see the learnt policy is not ideally the optimal shortest path, and the agent is trying to avoid the grid next to the cliff in its first several steps, why? This is a explainable:
- The $Q$ value fuction is averaged over episodes, an early cliff fall trace will drag the **average return** for those cliff-adjacent grids.
- Also we used $\epsilon$-greedy policy, so even in later episodes when the agent learnt a good policy, they will still randomly explore and occasionally fall off the cliff in cliff-adjacent grids.

## 7. Temporal-Difference Methods
In previous illustration of Monte Carlo methods, it estimate the value function using complete episodes. While this is intuitively simple and unbiased, it's very sample-inefficient. The value function updates only happen at the end of episodes, learning can be slow—especially in environments with long or variable episode lengths. 

Temporal-Difference (TD) methods address these limitations by updating value estimates after each time step using bootstrapped predictions, leading to faster and more stable learning.

I found an intuitive way to understand the difference between TD and MC methods are compare this to **Gradient Descent** and **SGD** in neural netwrok optimization, but in the temporal axis, view one step in RL as one batch in supervised model training.
- Gradient descent computes the gradient using the full dataset, while *SGD* compute using only data points in current batch, update the parameters, then move to the next batch.
- Monte-Carlo methods generates a full episode, backpropogated along the episode to update value function. While TD methods run one step, used its TD difference to update value function, then move to the next step.

Then how is TD-difference computed, remember we want to estimate value function using:
$$
V(s_t)\ = r_{t+1} + \gamma\ V(s_{t+1})
$$

So we can bootstrap at $s_t$, execute one more step and compute the value estimates and used it to update the value function:
$$
G(s_t) = r_{t+1} + \gamma\ V(s_{t+1}) 
\newline
\text{TD Error} = G(s_t) - V(s_t) 
\newline
V(s_t) \leftarrow V(s_t) + \alpha \cdot (G(s_t) - V(s_t))
$$
- $\gamma$ is the discount factor
- $\alpha$ is the single step learning rate

### 7.1. SARSA
SARSA is one of the most straightforward awy in TD-methods. The idea is intuitive, using next step's $Q(s_{t+1}, a_{t+1})$ to subtract current step's $Q(s_{t}, a_{t})$ as the TD error, and update value function.
$$
G(s_t) = r_{t+1} + \gamma\ Q(s_{t+1}, a_{t+1}) \newline
Q(s_{t}, a_{t}) \leftarrow Q(s_{t}, a_{t}) + \alpha \cdot (G(s_t) - Q(s_t, a_t))
$$
In every step, we need to get its current state $s_t$, action $a_t$, bootstrap one step forward, get the reward $r_{t+1}$, the new state $s_{t+1}$ and action $a_{t+1}$. In each step, we need the sequence of $<s_t, a_t, r_{t+1}, s_{t+1}, a_{t+1}>$, and this is why this method called *SARSA*.

### 7.1.1 SARSA method in Cliff Walking
- Step 0, Initilization:
    - Initialize a random value function: $Q(s, a)$
    - Initialize an $\epsilon$-greedy policy

- Step 1, Bootstrap a step:
    - Agent in state $s_t$ and action $a_t$
    - Bootstrap $a_t$, get the reward $r_{t+1}$ and new state $s_{t+t}$
    - Use the same policy to get the new action $a_{t+1}$
- Step 2, Update value function for the step: $Q(s_t, a_t)$
    - $Q(s_{t}, a_{t}) = Q(s_{t}, a_{t}) + \alpha \cdot (r_{t+1} + \gamma\ Q(s_{t+1}, a_{t+1}) - Q(s_t, a_t))$
- Finish 1 episode by repeated running step 1-2 until the agent reached `goal` state.
- Run above algorithm multiple times until the policy converge.

### 7.1.2 Python Implementation of SARSA
#### 7.1.2.1 SARSA
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

#### 7.1.2.2 Visualization and Result
We train SARSA for 10000 episodes, and visualize the result
```python
sarsa_progress = td_sarsa_training(env, num_episodes=10000, gamma=0.9, epsilon=0.1, alpha=0.2)
```
![SARSA Training Visualization](/images/CliffWalking-04-09-2025_09_47_PM.png)

The learnt policy in `epoch=9999` is similar to that learnt from MC methods, that it tries to avoid the cliff-adjacent grids, the reasoning is also similar:

- The agent used $\epsilon$-greedy search, so even the agent learnt a good policy, its exploration nature may still lead to fall off in cliff-adjacent grids. So the agent learnt to walk far away from the cliff, taking the constant cost of extra `-1` reward, to avoid a potential `-100` reward. 

### 7.2 Q-Learning
Let's recap the SARSA algorithm again, it used $\epsilon$-greedy search on $Q$ value functions for two purposes:
1. **Planning**: Decide the action $a_{t+1}$ of next step
2. **Policy Update**: use the actual action $a_{t+1}$ to update value function.

So this policy have to encorporate a trade-off between exploration and exploitation. What if we have two policies:
1. One **Behavior Policy** that focused on exploration, it decides the interaction with the environment.
2. One **Target Policy** that focused on exploitation, it doesn't do interaction, but focused on learning from previous interactions.

Then the behavior policy can be more aggressive to keep exploring risky areas, without fearing these risky behaviors affect its value function. On the other hand, its target policy focused on greedily learning the optimal policy, without being penalized by random exploratary behaviors.

This new method is called Q-Learning, the difference between SARSA and Q-Learning can also formalize as **On-Policy** vs **Off-Policy**:
- **On-Policy** learns the value of the policy it is actually using to make decisions.
- **Off-Policy** Learns the value of a different policy than the one it is currently using to make decisions.

In SARSA, we used the actual value $Q(s_{t+1}, a_{t+1})$ to update the value function:

$$
G(s_t) = r_{t+1} + \gamma\ Q(s_{t+1}, a_{t+1}) \newline
Q(s_{t}, a_{t}) \leftarrow Q(s_{t}, a_{t}) + \alpha \cdot (G(s_t) - Q(s_t, a_t))
$$

In Q-Learning, we used the theoretical optimal next action instead of actual next action for updates:
$$
G(s_t) = r_{t+1} + \gamma\ \underset{a}{max}\ Q(s_{t+1}) \newline
Q(s_{t}, a_{t}) \leftarrow Q(s_{t}, a_{t}) + \alpha \cdot (G(s_t) - Q(s_t, a_t))
$$

### 7.2.1 Q-Learning in Cliff Walking
- Step 0, Initilization:
    - Initialize a random value function: $Q(s, a)$
    - Initialize an $\epsilon$-greedy policy

- Step 1, Bootstrap a step:
    - Agent in state $s_t$ and action $a_t$
    - Bootstrap $a_t$, get the reward $r_{t+1}$ and new state $s_{t+t}$
- Step 2, Update value function for the step: $Q(s_t, a_t)$
    - $Q(s_{t}, a_{t}) = Q(s_{t}, a_{t}) + \alpha \cdot (r_{t+1} + \gamma\ max\ Q(s_{t+1}, a) - Q(s_t, a_t))$
- Finish 1 episode by repeated running step 1-2 until the agent reached `goal` state.
- Run above algorithm multiple times until the policy converge.

This looks very similar to SARSA, the only difference is:
- We no longer need to sample $s_{t+1}$ in each step.
- We used $max\ Q(s_{t+1})$ instead of $Q(s_{t+1}, a_{t+1})$ for value update.

### 7.2.2 Python Implementation of Q-Learning
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

#### 7.2.3 Visualization and Result
Q-Learning converges faster than SARSA, we only trained 200 episodes.
```python
q_learning_progress = q_learning_training(env, num_episodes=200, gamma=1.0, epsilon=0.1, alpha=0.2)
```
![Q-Learning Training Visualization](/images/CliffWalking-04-09-2025_11_18_PM.png)

While SARSA found a **safe path** under randomness of $\epsilon$-greedy, Q-Learning found the shortest optimal path- It learns to hug off the cliff!

## 8. Summary



