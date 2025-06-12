import matplotlib.pyplot as plt
import numpy as np
import matplotlib.cm as cm

# Define some colors for plotting the maze path
MAZE_COLORS = {
    'path': 0.8,      # Light gray for path
    'wall': 0.2,      # Dark gray for wall
    'start': 0.9,     # Yellow for start
    'goal': 0.1,      # Dark blue for goal
    'agent_path': 'red' # Red for agent's path
}

def plot_maze_path(maze_map, start_pos, goal_pos, agent_path=None, ax=None, title="Maze Path"):
    """
    Plots the maze grid and the agent's path.

    Args:
        maze_map (np.array): 2D array representing the maze.
                             0: path, 1: wall, 2: start, 3: goal (or use start_pos/goal_pos explicitly)
        start_pos (tuple): (row, col) of the starting position.
        goal_pos (tuple): (row, col) of the goal position.
        agent_path (list): List of (row, col) tuples representing the agent's trajectory.
        ax (matplotlib.axes.Axes): Axis to plot on. If None, a new figure and axis are created.
        title (str): Title of the plot.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))

    # Create a visual maze array based on MAZE_COLORS
    display_maze = np.copy(maze_map).astype(float)
    display_maze[maze_map == 0] = MAZE_COLORS['path'] # Paths
    display_maze[maze_map == 1] = MAZE_COLORS['wall'] # Walls

    # Mark start and goal
    display_maze[start_pos] = MAZE_COLORS['start']
    display_maze[goal_pos] = MAZE_COLORS['goal']

    ax.imshow(display_maze, cmap='Greys', origin='upper', extent=[ -0.5, display_maze.shape[1] - 0.5, display_maze.shape[0] - 0.5, -0.5])
    ax.set_xticks(np.arange(maze_map.shape[1]))
    ax.set_yticks(np.arange(maze_map.shape[0]))
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.grid(which='major', color='black', linestyle='-', linewidth=2)

    # Plot agent's path if provided
    if agent_path:
        path_rows, path_cols = zip(*agent_path)
        ax.plot(path_cols, path_rows, color=MAZE_COLORS['agent_path'], linewidth=2, marker='o', markersize=4)
        ax.plot(path_cols[0], path_rows[0], 'go', markersize=8, label='Start') # Mark start of path
        ax.plot(path_cols[-1], path_rows[-1], 'rx', markersize=8, label='End of Path') # Mark end of path

    ax.set_title(title)

def plot_q_values(Q_table, maze_env, action_to_plot_idx=0, ax=None, title="Q-values for Action (e.g., 0: Left)"):
    """
    Plots the Q-values for a specific action across the maze states.
    Assumes Q_table is (num_states, num_actions)
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))

    # Get maze dimensions
    rows, cols = maze_env.rows, maze_env.cols

    # Create a 2D array for Q-values to display on the maze
    q_display_grid = np.full((rows, cols), np.nan) # Use NaN for walls or unvisited
    
    # Map Q-values from 1D state ID to 2D grid
    for r in range(rows):
        for c in range(cols):
            state_id = maze_env.get_state_id((r, c))
            if maze_env.maze_map[r, c] != 1: # Only for non-wall cells
                q_display_grid[r, c] = Q_table[state_id, action_to_plot_idx]

    # Mask NaN values for proper color mapping
    masked_q_grid = np.ma.masked_where(np.isnan(q_display_grid), q_display_grid)

    cmap = cm.viridis # A good colormap for continuous data
    cax = ax.imshow(masked_q_grid, cmap=cmap, origin='upper', extent=[-0.5, cols - 0.5, rows - 0.5, -0.5])
    
    # Add Q-value text to each cell
    for r in range(rows):
        for c in range(cols):
            if maze_env.maze_map[r, c] != 1: # Only for non-wall cells
                val = q_display_grid[r, c]
                if not np.isnan(val):
                    ax.text(c, r, f'{val:.2f}', ha='center', va='center', color='black', fontsize=8)

    ax.set_xticks(np.arange(cols))
    ax.set_yticks(np.arange(rows))
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.grid(which='major', color='black', linestyle='-', linewidth=2)
    ax.set_title(title)
    plt.colorbar(cax, ax=ax, label=f"Q-value for Action {action_to_plot_idx} ({maze_env.actions_map.get(action_to_plot_idx, 'Unknown Action')})")

def plot_performance_metrics(logs, ax_steps=None, ax_rewards=None, title_prefix="Maze Performance"):
    """
    Plots the average steps to goal and total environmental rewards over trials.

    Args:
        logs (dict): Dictionary containing simulation logs, specifically 'all_steps_to_goal'
                     and 'all_env_rewards'.
        ax_steps (matplotlib.axes.Axes): Axis for steps plot.
        ax_rewards (matplotlib.axes.Axes): Axis for rewards plot.
        title_prefix (str): Prefix for the plot titles.
    """
    all_steps_to_goal = np.array(logs['all_steps_to_goal'])
    all_env_rewards = np.array(logs['all_env_rewards'])

    if ax_steps is None or ax_rewards is None:
        fig, (ax_steps, ax_rewards) = plt.subplots(1, 2, figsize=(14, 5))

    # Plot Steps to Goal
    mean_steps, ci_lower_steps, ci_upper_steps = oneD_CI(all_steps_to_goal, axis=0)
    x_axis = np.arange(len(mean_steps))
    ax_steps.plot(x_axis, mean_steps, label="Avg Steps to Goal", color='blue')
    ax_steps.fill_between(x_axis, ci_lower_steps, ci_upper_steps, color='blue', alpha=0.15)
    ax_steps.set_xlabel("Trial")
    ax_steps.set_ylabel("Steps to Goal")
    ax_steps.set_title(f"{title_prefix} - Steps to Goal")
    ax_steps.grid(True)

    # Plot Total Environmental Rewards
    mean_rewards, ci_lower_rewards, ci_upper_rewards = oneD_CI(all_env_rewards, axis=0)
    ax_rewards.plot(x_axis, mean_rewards, label="Avg Total Env Reward", color='green')
    ax_rewards.fill_between(x_axis, ci_lower_rewards, ci_upper_rewards, color='green', alpha=0.15)
    ax_rewards.set_xlabel("Trial")
    ax_rewards.set_ylabel("Total Environmental Reward")
    ax_rewards.set_title(f"{title_prefix} - Total Environmental Reward")
    ax_rewards.grid(True)


def plot_satiations_and_priorities(logs, ax_satiations=None, ax_priorities=None, title_prefix="Internal States"):
    """
    Plots the evolution of satiations and priorities.

    Args:
        logs (dict): Dictionary containing simulation logs, specifically 'all_satiations_per_step'
                     and 'all_priorities_per_step'.
        ax_satiations (matplotlib.axes.Axes): Axis for satiations plot.
        ax_priorities (matplotlib.axes.Axes): Axis for priorities plot.
        title_prefix (str): Prefix for the plot titles.
    """
    # Fix for ValueError: setting an array element with a sequence (inhomogeneous shape)
    # This section ensures that all test runs' data has the same length before converting to numpy array.
    
    # Find the minimum length among all test runs' per-step logs
    min_len = float('inf')
    if logs["all_satiations_per_step"]: # Ensure list is not empty
        for test_data in logs["all_satiations_per_step"]:
            min_len = min(min_len, len(test_data))
    else:
        print("No satiation/priority data available to plot.")
        return

    all_satiations_per_step_processed = []
    all_priorities_per_step_processed = []

    for test_idx in range(len(logs["all_satiations_per_step"])):
        # Truncate each test run's data to min_len
        test_satiations = np.array(logs["all_satiations_per_step"][test_idx][:min_len])
        test_priorities = np.array(logs["all_priorities_per_step"][test_idx][:min_len])
        all_satiations_per_step_processed.append(test_satiations)
        all_priorities_per_step_processed.append(test_priorities)

    # Now, convert the list of arrays (which now have consistent shapes) to a single NumPy array
    all_satiations_per_step_processed = np.array(all_satiations_per_step_processed)
    all_priorities_per_step_processed = np.array(all_priorities_per_step_processed)

    # The rest of the function (averaging and plotting) remains the same as before
    # as it now receives consistently shaped arrays.

    if len(all_satiations_per_step_processed.shape) == 3: # (nb_tests, total_steps_in_one_test_run, num_actions)
        mean_satiations, lower_satiations, upper_satiations = oneD_CI(all_satiations_per_step_processed, axis=0)
        mean_priorities, lower_priorities, upper_priorities = oneD_CI(all_priorities_per_step_processed, axis=0)

        if ax_satiations is None or ax_priorities is None:
            fig, (ax_satiations, ax_priorities) = plt.subplots(1, 2, figsize=(14, 5))

        # Action names (ensure consistency with MazeEnv.actions_map)
        action_names_list = ["Left", "Right", "Up", "Down"] # Assuming 4 actions. Adjust if different.
        colors = ['blue', 'orange', 'green', 'red'] # Corresponding colors for actions

        # Plot Satiations
        num_actions = mean_satiations.shape[1]
        for action_idx in range(num_actions):
            ax_satiations.plot(mean_satiations[:, action_idx], label=action_names_list[action_idx], color=colors[action_idx % len(colors)])
            ax_satiations.fill_between(np.arange(len(mean_satiations)), lower_satiations[:, action_idx], upper_satiations[:, action_idx], alpha=0.15, color=colors[action_idx % len(colors)])
        ax_satiations.set_xlabel("Cumulative Step in Simulation (Truncated to Min Length)")
        ax_satiations.set_ylabel("Satiation Level")
        ax_satiations.set_title(f"{title_prefix} - Evolution of Satiations")
        ax_satiations.legend()
        ax_satiations.grid(True)

        # Plot Priorities
        for action_idx in range(num_actions):
            ax_priorities.plot(mean_priorities[:, action_idx], label=action_names_list[action_idx], color=colors[action_idx % len(colors)])
            ax_priorities.fill_between(np.arange(len(mean_priorities)), lower_priorities[:, action_idx], upper_priorities[:, action_idx], alpha=0.15, color=colors[action_idx % len(colors)])
        ax_priorities.set_xlabel("Cumulative Step in Simulation (Truncated to Min Length)")
        ax_priorities.set_ylabel("Priority Level")
        ax_priorities.set_title(f"{title_prefix} - Evolution of Priorities")
        ax_priorities.legend()
        ax_priorities.grid(True)
    else:
        if ax_satiations is not None:
            ax_satiations.set_title("Satiation data not in expected 3D format for averaging (nb_tests, steps, actions)")
        if ax_priorities is not None:
            ax_priorities.set_title("Priority data not in expected 3D format for averaging (nb_tests, steps, actions)")


def oneD_CI(array_of_array, axis=0):
    """
    Calculates mean and 95% confidence interval for an array of arrays.
    Handles inputs from 1D to 3D.
    """
    if array_of_array.ndim == 1:
        mean = np.mean(array_of_array)
        std = np.std(array_of_array)
        n = len(array_of_array)
        if n > 1:
            ci = 1.96 * std / np.sqrt(n)
        else:
            ci = 0
        return mean, mean - ci, mean + ci
    elif array_of_array.ndim == 2:
        mean = np.mean(array_of_array, axis=axis)
        std = np.std(array_of_array, axis=axis)
        n = array_of_array.shape[axis] if axis == 0 else array_of_array.shape[0]
        if n > 1:
            ci = 1.96 * std / np.sqrt(n)
        else:
            ci = np.zeros_like(mean)
        return mean, mean - ci, mean + ci
    elif array_of_array.ndim == 3:
        # Expected: (nb_tests, num_data_points_per_test, num_dimensions)
        mean = np.mean(array_of_array, axis=axis) # Averages across nb_tests
        std = np.std(array_of_array, axis=axis)
        n = array_of_array.shape[axis]
        if n > 1:
            ci = 1.96 * std / np.sqrt(n)
        else:
            ci = np.zeros_like(mean)
        return mean, mean - ci, mean + ci
    else:
        raise ValueError("Unsupported array dimension for oneD_CI. Max 3D supported.")


def plot_maze_results(logs, env, agent):
    """
    Main plotting function for maze results.

    Args:
        logs (dict): Dictionary containing simulation logs.
        env (object): The environment object (MazeEnv instance) from the last test.
        agent (object): The agent object (KonidarisMF instance) from the last test.
    """
    print("Generating performance plots...")
    fig_perf, (ax_steps, ax_rewards) = plt.subplots(1, 2, figsize=(14, 5))
    plot_performance_metrics(logs, ax_steps=ax_steps, ax_rewards=ax_rewards)
    plt.suptitle("Maze Performance over Trials (Averaged Across Tests)")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

    print("Generating internal states plots...")
    fig_internal, (ax_satiations, ax_priorities) = plt.subplots(1, 2, figsize=(14, 5))
    plot_satiations_and_priorities(logs, ax_satiations=ax_satiations, ax_priorities=ax_priorities)
    plt.suptitle("Internal States (Satiations and Priorities) Evolution (Averaged Across Tests)")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

    print("Generating maze path plot...")
    # Plot the path of the *last* trial from the *last* test run
    # logs['all_paths_per_trial'] is (nb_tests, trials, path_length_variable, 2_coords)
    # We want the last path from the last test run
    if logs['all_paths_per_trial'] and logs['all_paths_per_trial'][-1]:
        # Get the path from the last trial of the last test run
        example_path = logs['all_paths_per_trial'][-1][-1]
        fig_maze_path, ax_maze_path = plt.subplots(figsize=(6, 6))
        plot_maze_path(env.maze_map, env.start_pos, env.goal_pos, agent_path=example_path,
                       ax=ax_maze_path, title=f"Agent Path in Last Trial of Last Test")
        plt.tight_layout()
        plt.show()
    else:
        print("No agent path data available to plot. Ensure 'all_paths_per_trial' is correctly logged in play.py.")
        fig_maze_path, ax_maze_path = plt.subplots(figsize=(6, 6))
        plot_maze_path(env.maze_map, env.start_pos, env.goal_pos, ax=ax_maze_path,
                       title="Maze Layout (No Path Data)")
        plt.tight_layout()
        plt.show()

    print("Generating Q-value plots...")
    # Plot Q-values for each action
    # Assuming `agent.Q` is accessible and has shape (num_states, num_actions)
    if hasattr(agent, 'Q') and agent.Q is not None and agent.Q.shape[0] > 0:
        num_actions = agent.size_actions
        
        # Determine grid size for subplots
        grid_rows = int(np.ceil(np.sqrt(num_actions)))
        grid_cols = int(np.ceil(num_actions / grid_rows))
        
        fig_q_values, axes_q_values = plt.subplots(grid_rows, grid_cols, figsize=(6 * grid_cols, 6 * grid_rows))
        axes_q_values = axes_q_values.flatten() if num_actions > 1 else [axes_q_values]

        for action_idx in range(num_actions):
            if action_idx < len(axes_q_values):
                action_name = env.actions_map.get(action_idx, f"Action {action_idx}") # Use env.actions_map
                plot_q_values(agent.Q, env, action_to_plot_idx=action_idx,
                              ax=axes_q_values[action_idx], title=f"Q-values for {action_name}")
            else:
                # Hide unused subplots if any
                axes_q_values[action_idx].set_visible(False)

        plt.suptitle("Learned Q-Values for Each Action")
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()
    else:
        print("Agent Q-table not available for plotting or is empty. Ensure agent.Q is properly initialized and accessible.")