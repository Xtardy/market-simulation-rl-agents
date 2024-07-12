import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import torch
import torch.nn as nn
import torch.optim as optim

# Parameters
N_AGENTS = 100
INITIAL_PRICE = 100
INITIAL_BALANCE = 1000
LEARNING_RATE = 0.001
GAMMA = 0.99
UPDATE_INTERVAL = 10
PRICE_HISTORY_LENGTH = 5
INACTIVITY_LIMIT = 100  # Number of periods before resetting the agent

class Agent(nn.Module):
    def __init__(self):
        super(Agent, self).__init__()
        self.fc1 = nn.Linear(PRICE_HISTORY_LENGTH + 2, 16)
        self.fc2 = nn.Linear(16, 10)  # 10 outputs to represent 1% to 10%
        self.reset()

    def reset(self):
        self.balance = INITIAL_BALANCE
        self.position = 0  # 0: neutral, positive: long, negative: short
        self.optimizer = optim.Adam(self.parameters(), lr=LEARNING_RATE)
        self.inactivity_counter = 0  # Inactivity counter

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return torch.softmax(self.fc2(x), dim=0)  # Softmax to get probabilities

# Initialization
agents = [Agent() for _ in range(N_AGENTS)]
price_history = [INITIAL_PRICE]

# Visualization setup
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 8))
line, = ax1.plot([], [])
bars = ax2.bar(range(N_AGENTS), [agent.balance for agent in agents])

ax1.set_title("Market Price")
ax1.set_xlabel("Time")
ax1.set_ylabel("Price")

ax2.set_title("Agent Balances")
ax2.set_xlabel("Agents")
ax2.set_ylabel("Balance")

ax3.set_title("Balance Distribution")
ax3.set_xlabel("Balance")
ax3.set_ylabel("Number of Agents")

def update(frame):
    global price_history

    # Brownian motion
    price_change = np.random.normal(0, 0.1)
    new_price = max(0, price_history[-1] + price_change)

    # Ensure we have at least PRICE_HISTORY_LENGTH prices in the history
    if len(price_history) < PRICE_HISTORY_LENGTH:
        price_history = [new_price] * PRICE_HISTORY_LENGTH

    total_investment = 0
    # Update agents
    for i, agent in enumerate(agents):
        # Create the state with the last 5 prices, balance, and position
        state = torch.tensor(
            price_history[-PRICE_HISTORY_LENGTH:] + [agent.balance, agent.position],
            dtype=torch.float32
        )
        action_probs = agent(state)
        action = torch.argmax(action_probs).item()

        # Check inactivity and reset if necessary
        if action == 9:  # If the agent chooses to do nothing (9 is now the non-investment action)
            agent.inactivity_counter += 1
            if agent.inactivity_counter >= INACTIVITY_LIMIT:
                agent.reset()
                print(f"Agent {i} reset due to inactivity")
        else:
            agent.inactivity_counter = 0  # Reset the counter if the agent acts

        invest_percentage = (action + 1) / 100.0  

        # Apply position and update balance based on percentage
        if agent.position == 0:
            if np.random.rand() < 0.5:
                agent.position = invest_percentage  # Long (Call)
            else:
                agent.position = -invest_percentage  # Short (Put)
        else:
            # Update balance based on current position
            profit = agent.position * (new_price - price_history[-2]) * agent.balance
            agent.balance += profit
            agent.position = 0  # Close the position after updating

        # Reinforcement learning
        if frame % UPDATE_INTERVAL == 0:
            reward = agent.balance - INITIAL_BALANCE
            loss = -torch.log(action_probs[action]) * reward
            agent.optimizer.zero_grad()
            loss.backward(retain_graph=True)
            agent.optimizer.step()
            
        total_investment += agent.position * agent.balance
        
    market_impact = total_investment / (N_AGENTS * INITIAL_BALANCE)
    new_price += market_impact
    price_history.append(new_price)
    
    # Update the plots
    line.set_data(range(len(price_history)), price_history)
    ax1.relim()
    ax1.autoscale_view()

    for i, bar in enumerate(bars):
        bar.set_height(agents[i].balance)
    ax2.relim()
    ax2.autoscale_view()

    # Update the histogram
    ax3.clear()
    ax3.set_title("Balance Distribution")
    ax3.set_xlabel("Balance")
    ax3.set_ylabel("Number of Agents")
    balances = [agent.balance for agent in agents]
    ax3.hist(balances, bins=20, edgecolor='black')
    ax3.set_xlim(min(balances), max(balances))

    return line, bars

ani = FuncAnimation(fig, update, frames=1000, interval=50, blit=False)
plt.tight_layout()
plt.show()
