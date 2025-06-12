import numpy as np

class MultiArmedBandit:
    def __init__(self, reward_rates=[]):
        self.number_states = 1
        self.number_actions = len(reward_rates) + 1
        self.states = np.arange(self.number_states)
        self.actions = np.arange(self.number_actions)
        self.step = 0
        self.agent_state = 0
        self.probas = [0.] + reward_rates

    def make_step(self, action):
        self.step += 1
        proba = self.probas[action]
        if np.random.random() < proba:
            return 1, self.agent_state
        else:
            return 0, self.agent_state

    def new_episode(self):
        pass


class MazeEnv:
    def __init__(self, maze_map=None, start_pos=(0,0), goal_pos=None):
        if maze_map is None:
            # Default simple maze if none provided
            self.maze_map = np.array([
                [0, 0, 0, 0, 0],
                [0, 1, 0, 1, 0],
                [0, 0, 0, 0, 0],
                [0, 1, 0, 1, 0],
                [0, 0, 0, 0, 0]
            ])
        else:
            self.maze_map = np.array(maze_map)

        self.rows, self.cols = self.maze_map.shape
        self.start_pos = tuple(start_pos) # Ensure tuple
        self.goal_pos = tuple(goal_pos) if goal_pos is not None else (self.rows - 1, self.cols - 1) # Default goal is bottom-right

        # Define actions: (row_change, col_change)
        self.actions_movements = {
            0: (0, -1),  # Left
            1: (0, 1),   # Right
            2: (-1, 0),  # Up
            3: (1, 0)    # Down
        }
        self.number_actions = len(self.actions_movements)
        self.actions = np.arange(self.number_actions)

        # Mapping for plotting Q-values (human-readable names)
        self.actions_map = {
            0: "Left",
            1: "Right",
            2: "Up",
            3: "Down"
        }

        # Create a mapping from (row, col) position to a unique state ID
        self.states = {}
        self.pos_to_state_id = {} # New: for quick lookup
        self.state_id_to_pos = {} # New: for quick inverse lookup
        state_id_counter = 0
        for r in range(self.rows):
            for c in range(self.cols):
                if self.maze_map[r, c] != 1:  # Not a wall
                    pos = (r, c)
                    self.states[pos] = state_id_counter
                    self.pos_to_state_id[pos] = state_id_counter
                    self.state_id_to_pos[state_id_counter] = pos
                    state_id_counter += 1
        self.number_states = state_id_counter

        self.agent_pos = None
        self.agent_state = None # Current state ID of the agent
        self.new_episode() # Initialize agent position

    def new_episode(self):
        self.agent_pos = self.start_pos
        self.agent_state = self.get_state_id(self.agent_pos)
        return self.agent_state

    def get_state_id(self, pos):
        """Converts (row, col) position to a unique integer state ID."""
        return self.pos_to_state_id.get(pos)

    def get_pos_from_state_id(self, state_id):
        """Converts a unique integer state ID back to (row, col) position."""
        return self.state_id_to_pos.get(state_id)

    def is_valid_pos(self, pos):
        r, c = pos
        return (0 <= r < self.rows and 0 <= c < self.cols and
                self.maze_map[r, c] != 1) # Not a wall

    def make_step(self, action):
        dr, dc = self.actions_movements[action]
        new_r, new_c = self.agent_pos[0] + dr, self.agent_pos[1] + dc
        new_pos = (new_r, new_c)

        reward = 0
        done = False

        if self.is_valid_pos(new_pos):
            self.agent_pos = new_pos
            self.agent_state = self.get_state_id(self.agent_pos)
            if self.agent_pos == self.goal_pos:
                reward = 1 # Reward for reaching the goal
                done = True
        else:
            # If invalid move (hit wall or out of bounds), agent stays in current position
            # No reward, no change in state
            pass 
        
        return reward, self.agent_state