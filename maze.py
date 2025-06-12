import numpy as np
import matplotlib.pyplot as plt
import random
from matplotlib.colors import ListedColormap

class MazeEnvironment:
    def __init__(self, size=10, obstacles_prob=0.2):
        self.size = size
        # 0: empty, 1: obstacle, 2: start, 3: goal, 4: food/resource
        self.maze = np.zeros((size, size))
        
        # Add obstacles
        for i in range(size):
            for j in range(size):
                if random.random() < obstacles_prob:
                    self.maze[i, j] = 1
        
        # Ensure start and goal are not obstacles
        self.start_pos = (0, 0)
        self.goal_pos = (size-1, size-1)
        self.maze[self.start_pos] = 2
        self.maze[self.goal_pos] = 3
        
        # Add food/resources for maintaining homeostasis
        food_count = size // 3
        for _ in range(food_count):
            x, y = random.randint(0, size-1), random.randint(0, size-1)
            if self.maze[x, y] == 0:  # Only place on empty cells
                self.maze[x, y] = 4
    
    def render(self):
        cmap = ListedColormap(['white', 'black', 'green', 'red', 'blue'])
        plt.figure(figsize=(7, 7))
        plt.imshow(self.maze, cmap=cmap)
        plt.title("Maze with Homeostasis Resources")
        plt.grid(True)
        plt.show()

class HomeostasisAgent:
    def __init__(self, maze, learning_rate=0.1, discount_factor=0.9, exploration_rate=0.3):
        self.maze = maze
        self.position = maze.start_pos
        self.size = maze.size
        
        # Internal states (homeostasis variables)
        self.energy = 100  # Max energy
        self.energy_depletion_rate = 1  # Energy lost per action
        self.energy_critical = 20  # Critical energy level
        
        # Learning parameters
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        
        # Q-table: [x, y, energy_state, action] -> value
        # Energy states: 0 (critical), 1 (normal)
        self.q_table = {}
        
        # Actions: 0: up, 1: right, 2: down, 3: left
        self.actions = [(0, -1), (1, 0), (0, 1), (-1, 0)]
    
    def get_state(self):
        # Get current state including position and energy state
        energy_state = 0 if self.energy <= self.energy_critical else 1
        return (self.position[0], self.position[1], energy_state)
    
    def get_q_value(self, state, action):
        if (state, action) not in self.q_table:
            self.q_table[(state, action)] = 0
        return self.q_table[(state, action)]
    
    def choose_action(self):
        if random.random() < self.exploration_rate:
            return random.randint(0, 3)
        
        state = self.get_state()
        q_values = [self.get_q_value(state, a) for a in range(4)]
        return np.argmax(q_values)
    
    def take_action(self, action):
        # Get the movement vector
        dx, dy = self.actions[action]
        new_x, new_y = self.position[0] + dx, self.position[1] + dy
        
        # Check if the move is valid
        if 0 <= new_x < self.size and 0 <= new_y < self.size and self.maze.maze[new_x, new_y] != 1:
            self.position = (new_x, new_y)
        
        # Update internal states
        self.energy -= self.energy_depletion_rate
        
        # Check for food/resources
        if self.maze.maze[self.position] == 4:
            self.energy = min(100, self.energy + 30)  # Restore energy
            self.maze.maze[self.position] = 0  # Consume the resource
        
        # Calculate reward
        reward = 0
        
        # Goal reached reward
        if self.position == self.maze.goal_pos:
            reward += 100
            return reward, True
        
        # Homeostasis reward: penalize if energy is critical
        if self.energy <= self.energy_critical:
            reward -= 5
        
        # Death condition
        if self.energy <= 0:
            reward -= 100
            return reward, True
        
        return reward, False
    
    def update_q_table(self, state, action, reward, next_state, done):
        old_value = self.get_q_value(state, action)
        
        if done:
            future_rewards = 0
        else:
            next_q_values = [self.get_q_value(next_state, a) for a in range(4)]
            future_rewards = self.discount_factor * max(next_q_values)
        
        # Q-learning update
        new_value = old_value + self.learning_rate * (reward + future_rewards - old_value)
        self.q_table[(state, action)] = new_value

def train_agent(env, agent, episodes=1000):
    rewards_history = []
    steps_history = []
    
    for episode in range(episodes):
        # Reset environment and agent
        env = MazeEnvironment(size=env.size, obstacles_prob=0.3)
        agent.maze = env
        agent.position = env.start_pos
        agent.energy = 100
        
        total_reward = 0
        steps = 0
        done = False
        
        while not done and steps < 1000:  # Limit steps to avoid infinite loops
            state = agent.get_state()
            action = agent.choose_action()
            reward, done = agent.take_action(action)
            next_state = agent.get_state()
            
            agent.update_q_table(state, action, reward, next_state, done)
            
            total_reward += reward
            steps += 1
        
        rewards_history.append(total_reward)
        steps_history.append(steps)
        
        if episode % 100 == 0:
            print(f"Episode {episode}, Avg Reward: {np.mean(rewards_history[-100:]):.2f}, Avg Steps: {np.mean(steps_history[-100:]):.2f}")
    
    return rewards_history, steps_history

def main():
    env = MazeEnvironment(size=10)
    agent = HomeostasisAgent(env)
    
    env.render()  # Show initial maze
    
    rewards, steps = train_agent(env, agent, episodes=1000)
    
    # Plot results
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(rewards)
    plt.title('Rewards per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    
    plt.subplot(1, 2, 2)
    plt.plot(steps)
    plt.title('Steps per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Steps Taken')
    
    plt.tight_layout()
    plt.show()
    
    # Test and visualize a trained agent
    test_env = MazeEnvironment(size=10)
    agent.maze = test_env
    agent.position = test_env.start_pos
    agent.energy = 100
    agent.exploration_rate = 0  # No exploration during testing
    
    # Record agent's path
    path = [agent.position]
    done = False
    steps = 0
    
    while not done and steps < 100:
        action = agent.choose_action()
        _, done = agent.take_action(action)
        path.append(agent.position)
        steps += 1
    
    # Render final path
    cmap = ListedColormap(['white', 'black', 'green', 'red', 'blue', 'yellow'])
    plt.figure(figsize=(7, 7))
    maze_with_path = test_env.maze.copy()
    for x, y in path:
        if maze_with_path[x, y] == 0:  # Only mark path on empty cells
            maze_with_path[x, y] = 5  # Path color
    plt.imshow(maze_with_path, cmap=cmap)
    plt.title("Agent's Path with Homeostasis Management")
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()