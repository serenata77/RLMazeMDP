from plots import  plot_maze_results  
<<<<<<< HEAD
from agents_maze import KonidarisMF
=======
from agents_corrected import KonidarisMF
>>>>>>> 97ef2e1cbabb8b1b734022f2efcb88eced59b414
# from envs import MultiArmedBandit # Remove or comment out
from envs import MazeEnv # Import your new MazeEnv
from play import play_with_parameters
import numpy as np

# Definitions
agent_name_to_class = {"KonidarisMF": KonidarisMF}

# env_name_to_class = {"Bandit": MultiArmedBandit} # Remove or comment out
env_name_to_class = {"Maze": MazeEnv} # Add your MazeEnv

agent_to_test = "KonidarisMF"

env_to_test = "Maze" # Change to Maze

###########
# Change the parameters of the environment here
# Example Maze setup
maze_map_example = [
    [0, 0, 0, 0, 0],
    [0, 1, 0, 1, 0],
    [0, 0, 0, 0, 0],
    [0, 1, 0, 1, 0],
    [0, 0, 0, 0, 0]
] # 0: path, 1: wall
start_pos_example = (0, 0)
goal_pos_example = (4, 4)

env_params = {"Maze": {"maze_map": maze_map_example,
                       "start_pos": start_pos_example,
                       "goal_pos": goal_pos_example}}
###########

###########
# Change the parameters of the agent here
params_KonidarisMF = {"alpha": 0.2,
                      "beta": 0.5, # Adjust beta for exploration in maze
                      "changing_priorities": False, # Start with False for simpler behavior
                      "differential_evolution": None, # or [1,1,1,1] if 4 actions
                      "increase_decrease_ratio": 5}

agent_params = {"KonidarisMF": params_KonidarisMF}
###########

###########
# Choose the number of trials, the number of tests and the random seed here
trials = 500 # Number of episodes (maze runs)
nb_tests = 5 # Number of times to run the whole experiment
np.random.seed(1)
###########


# Launching the experiment and plotting results
logs, env, agent = play_with_parameters(nb_tests,
                                        env_name_to_class,
                                        env_to_test,
                                        env_params,
                                        agent_name_to_class,
                                        agent_to_test,
                                        agent_params,
                                        trials)

# You'll likely need a new plotting function or adapt plot_bandit to visualize maze performance
# For example, plot average steps to goal, or visualize learned Q-values on the maze.
plot_maze_results(logs, env, agent)
