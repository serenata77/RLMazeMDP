import numpy as np

class MazeEnv:
    def __init__(self, maze_map, start_pos, goal_pos):
        self.maze_map = np.array(maze_map) # e.g., 0 for path, 1 for wall, 2 for goal
        self.rows, self.cols = self.maze_map.shape
        self.start_pos = start_pos
        self.goal_pos = goal_pos
        self.agent_pos = start_pos # (row, col)

        # Map (row, col) to a unique integer state ID
        self.states = {}
        state_id = 0
        for r in range(self.rows):
            for c in range(self.cols):
                self.states[(r, c)] = state_id
                state_id += 1
        self.number_states = state_id # Total number of valid positions

        self.actions = {
            0: (0, -1),  # Left
            1: (0, 1),   # Right
            2: (-1, 0),  # Up
            3: (1, 0)    # Down
        }
        self.number_actions = len(self.actions)

        self.agent_state = self.states[self.agent_pos] # Initial integer state ID

    def get_state_id(self, pos):
        return self.states[pos]

    def get_pos_from_state_id(self, state_id):
        # Inverse mapping (you'd need to store this or calculate it)
        for pos, s_id in self.states.items():
            if s_id == state_id:
                return pos
        return None

    def make_step(self, action_id):
        current_row, current_col = self.agent_pos
        dr, dc = self.actions[action_id]
        new_row, new_col = current_row + dr, current_col + dc

        reward_env = -0.01 # Small penalty for each step

        # Check for valid move
        if 0 <= new_row < self.rows and \
           0 <= new_col < self.cols and \
           self.maze_map[new_row, new_col] != 1: # Assuming 1 is a wall
            self.agent_pos = (new_row, new_col)
        else:
            # Hit a wall, stay in the same position
            reward_env = -0.1 # Penalty for hitting a wall
            # self.agent_pos remains unchanged

        # Check if goal is reached
        if self.agent_pos == self.goal_pos:
            reward_env = 1 # Positive reward for reaching the goal

        new_state_id = self.get_state_id(self.agent_pos)
        return reward_env, new_state_id

    def new_episode(self):
        self.agent_pos = self.start_pos
        self.agent_state = self.get_state_id(self.agent_pos)