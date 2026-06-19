# Reinforcement Learning Traffic Light Signal Control
# Q-Learning with Real-World Traffic Volume Data


import pandas as pd
import numpy as np
import random
from tabulate import tabulate


# Load the real-world dataset
data = pd.read_csv("Metro_Interstate_Traffic_Volume.csv")

print("Dataset loaded successfully.")
print(data.head())


# Only take some part of the dataset randomly
data = data.sample(200, random_state=1).reset_index(drop=True)


# Discretise traffic volume into states
def discretize_traffic(volume):
    if volume <= 2000:
        return 0   # Low traffic
    elif volume <= 4000:
        return 1   # Medium traffic
    else:
        return 2   # High traffic

data["state"] = data["traffic_volume"].apply(discretize_traffic)


# States and actions
states = [0, 1, 2]       
actions = [0, 1]         


# Initialise Q-table 3x3
Q = np.zeros((len(states), len(actions)))


# Q-learning parameters
alpha = 0.1      # Learning rate
gamma = 0.9      # Discount factor
epsilon = 0.2    # Exploration rate


# Q-learning agent
for episode in range(200):

    # Randomly pick a traffic situation
    row = data.sample(1).iloc[0]
    state = row["state"]

    # ε-greedy action selection
    if random.uniform(0, 1) < epsilon:
        action = random.choice(actions)
    else:
        action = np.argmax(Q[state])

    # Negative Reward: penalise high traffic volume
    reward = -row["traffic_volume"]

    # Simplified environment (state will remains the same)
    next_state = state

    # Q-learning update
    Q[state, action] = Q[state, action] + alpha * (
        reward + gamma * np.max(Q[next_state]) - Q[state, action]
    )


# Round up values
Q_rounded = np.round(Q).astype(int)


# Label states and actions
state_labels = {
    0: "Low Traffic",
    1: "Medium Traffic",
    2: "High Traffic"
}

action_labels = {
    0: "Keep Signal",
    1: "Switch Signal"
}


# Create labelled Q-table
Q_df = pd.DataFrame(
    Q_rounded,
    index=[state_labels[s] for s in states],
    columns=[action_labels[a] for a in actions]
)

# Display Q-table with borders 
print("\nLabelled Q-Table (Rounded Integers):")
print(tabulate(Q_df, headers="keys", tablefmt="grid"))


# Show optimal action per traffic state
print("\nOptimal Action per Traffic State:")
for state in states:
    best_action = np.argmax(Q[state])
    print(f"{state_labels[state]} → {action_labels[best_action]}")
