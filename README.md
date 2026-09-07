# Implementation-of-SARSA-Control-Algorithm-using-Gymnasium
## NAME: VELLACHI TILAK
## REG NO: 212223240172
## Aim

To implement the **SARSA control algorithm** using the Gymnasium `FrozenLake-v1` environment and learn an action-value function that helps the agent select better actions for reaching the goal state while avoiding holes.


## Problem Statement
# SARSA Control Algorithm

The problem is to implement the SARSA control algorithm using the Gymnasium `FrozenLake-v1` environment. The agent must learn an optimal action-value function by interacting with the environment and selecting actions using an epsilon-greedy policy. The objective is to reach the goal state safely while avoiding the hole states.

## Software Requirements

* Python 3.x
  
* Jupyter Notebook/colab
  
* Gymnasium
  
* NumPy
  
* Matplotlib



## Environment Description

The `FrozenLake-v1` environment is a grid-based reinforcement learning problem. The agent moves through frozen states to reach the goal while avoiding holes.

* **S** – Starting state
* **F** – Frozen and safe state
* **H** – Hole state
* **G** – Goal state

The agent can perform four actions: **Left, Down, Right, and Up**. A reward is received when the agent successfully reaches the goal state.


## Theory

SARSA stands for:

$$
S_t, A_t, R_{t+1}, S_{t+1}, A_{t+1}
$$

It updates the Q-value using the action actually selected in the next state.

The SARSA update rule is:

$$
Q(S_t,A_t) \leftarrow Q(S_t,A_t) + \alpha
\left[
R_{t+1} + \gamma Q(S_{t+1},A_{t+1}) - Q(S_t,A_t)
\right]
$$

Where:

| Symbol | Meaning |
|---|---|
| $S_t$ | Current state |
| $A_t$ | Current action |
| $R_{t+1}$ | Reward received after taking action $A_t$ |
| $S_{t+1}$ | Next state |
| $A_{t+1}$ | Next action selected using the current policy |
| $\alpha$ | Learning rate |
| $\gamma$ | Discount factor |
| $Q(s,a)$ | Action-value function |

---

## Epsilon-Greedy Policy

SARSA uses an epsilon-greedy policy for action selection.

With probability $\epsilon$, the agent explores by selecting a random action.

With probability $1-\epsilon$, the agent exploits by selecting the action with the highest Q-value.

$$
a =
\begin{cases}
\text{random action}, & \text{with probability } \epsilon \\
\arg\max_a Q(s,a), & \text{with probability } 1-\epsilon
\end{cases}
$$

---


## Algorithm
1. Initialize the Q-table with zeros.
2. Set the learning rate, discount factor, epsilon, and number of episodes.
3. Reset the environment and obtain the initial state.
4. Select an action using the epsilon-greedy policy.
5. Perform the selected action and observe the reward and next state.
6. Select the next action using the epsilon-greedy policy.
7. Update the Q-value using the SARSA update rule.
8. Replace the current state and action with the next state and action.
9. Repeat until the episode terminates.
10. Repeat the process for all episodes and obtain the learned policy.

## Python Program

```python

# -------------------------------------------------
# SARSA Training
# -------------------------------------------------
# Write your code here

# Initialize Q-table with zeros
Q = np.zeros((env.observation_space.n, env.action_space.n))

episode_rewards = []

for episode in range(num_episodes):

    # Reset the environment
    state, info = env.reset()

    # Choose initial action using epsilon-greedy policy
    action = epsilon_greedy_action(state, epsilon)

    total_reward = 0

    for step in range(max_steps_per_episode):

        # Take the selected action
        next_state, reward, terminated, truncated, info = env.step(action)

        # Check whether the episode is finished
        done = terminated or truncated

        # Choose next action using epsilon-greedy policy
        next_action = epsilon_greedy_action(next_state, epsilon)

        # SARSA update rule
        Q[state, action] = Q[state, action] + alpha * (
            reward + gamma * Q[next_state, next_action] - Q[state, action]
        )

        # Move to the next state and action
        state = next_state
        action = next_action

        # Add reward
        total_reward += reward

        # Stop if the episode is finished
        if done:
            break

    # Store the total reward for this episode
    episode_rewards.append(total_reward)

    # Reduce epsilon
    epsilon = max(epsilon_min, epsilon * epsilon_decay)


# -------------------------------------------------
# Extract State Values and Policy
# -------------------------------------------------

state_values = np.max(Q, axis=1)

policy = np.argmax(Q, axis=1)




```
---

## Output

Final Q-table:

<img width="1007" height="272" alt="image" src="https://github.com/user-attachments/assets/088bc795-62a7-472d-8c36-ba933f1cf6ce" />





Estimated State-Value Function:

<img width="770" height="93" alt="image" src="https://github.com/user-attachments/assets/ff417eb3-d938-42e6-b5ec-459109eaa89b" />




Learned Policy:

<img width="695" height="115" alt="image" src="https://github.com/user-attachments/assets/92ebc667-8e04-4c19-ba5d-8a69b2078a1b" />


<img width="576" height="393" alt="image" src="https://github.com/user-attachments/assets/75cf4b29-22b3-4878-b1d3-1ae1fc8ffaa6" />



## Result


The SARSA algorithm was successfully implemented in the Gymnasium FrozenLake environment. The agent updated its Q-values based on the actions it actually selected and gradually learned a policy for reaching the goal while reducing the chances of falling into holes.


## Inference

The update target in SARSA relies on $Q(S_{t+1}, A_{t+1})$, which causes value estimation updates to fluctuate when exploration is high. Gradually decaying $\epsilon$ reduces policy variance over time, stabilizing the TD error and allowing the action-value function to smoothly converge to an optimal policy in stochastic grid environments.
