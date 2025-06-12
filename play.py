import numpy as np

def play_function(environment,
                  agent,
                  trials=100,
                  max_steps_per_trial=1000):
    """
Run the :agent: on the :environment: for a maze task.
    """
    log = {}
    all_rewards = [] # This will be internal rewards sum per trial
    all_env_rewards = [] # To log actual environment rewards sum per trial
    all_choices_per_step = [] # Log of actions chosen at each step
    all_probas_per_step = [] # Log of action probabilities at each step
    all_satiations_per_step = [] # Log of satiations at each step
    all_priorities_per_step = [] # Log of priorities at each step
    all_steps_to_goal = []
    all_paths_per_trial = [] # NEW: To store the agent's path for each trial

    for trial_num in range(trials):
        environment.new_episode()
        current_state_id = environment.agent_state # This is the integer ID
        current_pos = environment.get_pos_from_state_id(current_state_id) # Get (row, col)
        
        done = False
        steps_this_trial = 0
        trial_internal_rewards = []
        trial_env_rewards = []
        path_this_trial = [current_pos] # Start path with initial position

        while not done and steps_this_trial < max_steps_per_trial:
            action = agent.choose_action(current_state_id)
            reward_env, new_state_id = environment.make_step(action)
            internal_reward = agent.learn(current_state_id,
                                          reward_env,
                                          new_state_id,
                                          action)

            trial_internal_rewards.append(internal_reward)
            trial_env_rewards.append(reward_env)
            all_choices_per_step.append(action) # Log actions for *every step*
            all_probas_per_step.append(list(agent.Q_probas[current_state_id].flatten())) # Log probas for *current state*
            all_satiations_per_step.append(list(agent.satiations.flatten())) # Log satiations for *every step*
            all_priorities_per_step.append(list(agent.priorities.flatten())) # Log priorities for *every step*

            current_state_id = new_state_id
            current_pos = environment.get_pos_from_state_id(current_state_id) # Update (row, col)
            path_this_trial.append(current_pos) # Add new position to path

            steps_this_trial += 1

            if current_pos == environment.goal_pos: # Check if goal is reached using (row, col)
                done = True

        all_rewards.append(np.sum(trial_internal_rewards))
        all_env_rewards.append(np.sum(trial_env_rewards))
        all_steps_to_goal.append(steps_this_trial)
        all_paths_per_trial.append(path_this_trial) # Store the complete path for this trial

    log['all_rewards'] = all_rewards # Sum of internal rewards per trial
    log['all_env_rewards'] = all_env_rewards # Sum of env rewards per trial
    log['all_choices_per_step'] = all_choices_per_step
    log['all_probas_per_step'] = all_probas_per_step
    log['all_satiations_per_step'] = all_satiations_per_step
    log['all_priorities_per_step'] = all_priorities_per_step
    log['all_steps_to_goal'] = all_steps_to_goal
    log['all_paths_per_trial'] = all_paths_per_trial # NEW: Maze paths

    return log


def play_with_parameters(nb_tests,
                         env_name_to_class,
                         env_to_test,
                         env_params,
                         agent_name_to_class,
                         agent_to_test,
                         agent_params,
                         trials,
                         max_steps_per_trial=1000):
    all_logs = {}
    last_env = None # To store the last environment instance
    last_agent = None # To store the last agent instance

    for i in range(nb_tests):
        env_class = env_name_to_class[env_to_test]
        env = env_class(**env_params[env_to_test])
        agent_class = agent_name_to_class[agent_to_test]
        agent = agent_class(env, **agent_params[agent_to_test])
        logs = play_function(environment=env,
                             agent=agent,
                             trials=trials,
                             max_steps_per_trial=max_steps_per_trial)
        if i == 0:
            for key in logs.keys():
                all_logs[key] = []
        for key, value in logs.items():
            all_logs[key].append(value)
        last_env = env # Keep track of the last environment
        last_agent = agent # Keep track of the last agent

    return all_logs, last_env, last_agent # Return the last env and agent for plotting