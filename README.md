
# market-simulation-rl-agents

This project simulates a financial market with reinforcement learning agents making investment decisions. Agents use neural networks to make these decisions. Their actions, combined with Brownian motion, influence market prices. The simulation visualizes market price changes, agent balances, and balance distribution in real-time.

## Features
- **Reinforcement Learning Agents**: Agents use neural networks to make decisions based on recent price history, balance, and position.
- **Market Impact Simulation**: Agents' actions dynamically affect market prices.
- **Visualization**: Real-time plots of market price, agent balances, and balance distribution.

## Requirements
- Python 3.x
- NumPy
- Matplotlib
- PyTorch

## Installation

1. Install Python 3.x from the [official website](https://www.python.org/).
2. Install the required packages using pip:
   ```bash
   pip install numpy matplotlib torch
   ```

## Usage

To run the simulation, execute the Python script:
   ```bash
   python market_simulation.py
   ```

This will open a window displaying three plots:
1. **Market Price**: Shows the evolution of the market price over time.
2. **Agent Balances**: Displays the balances of all agents.
3. **Balance Distribution**: Illustrates the distribution of agent balances.

## Customization

You can customize the simulation parameters by modifying the following variables in the script:
- `N_AGENTS`: Number of agents.
- `INITIAL_PRICE`: Starting market price.
- `INITIAL_BALANCE`: Starting balance for each agent.
- `LEARNING_RATE`: Learning rate for the agents' optimizer.
- `GAMMA`: Discount factor for reinforcement learning.
- `UPDATE_INTERVAL`: Interval for agents' learning updates.
- `PRICE_HISTORY_LENGTH`: Number of past prices considered.
- `INACTIVITY_LIMIT`: Inactivity limit before resetting an agent.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
