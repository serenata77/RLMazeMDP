import numpy as np
from constants import konidaris_function_sym # Make sure constants.py defines this if you use it

class KonidarisMF:

    def __init__(
            self,
            environment,
            alpha=0.3,
            beta=1.,
            changing_priorities = True,
            increase_decrease_ratio = 10,
            differential_evolution = None):

        self.size_environment = len(environment.states)
        self.size_actions = environment.number_actions # Use environment.number_actions
        self.shape_SA = (self.size_environment, self.size_actions)

        # Initialize Q-table (make sure it's accessible as self.Q)
        self.Q = np.zeros(self.shape_SA) # Ensure this line exists and initializes Q

        # Intializing satiations and priorities to
        self.satiations = 1/2 * np.ones(self.size_actions)
        self.priorities = 1/2 * np.ones(self.size_actions)

        # Exploration parameter
        self.beta = beta

        # Learning parameter
        self.alpha = alpha

        # Boolean to decide if the agent should prioritize some resources or not
        self.changing_priorities = changing_priorities

        # Ratio between the decrease after 1 step and the increase when getting
        # a reward
        self.increase_decrease_ratio = increase_decrease_ratio


        # Defining the basic decrease step
        self.basic_step = 0.01

        # Modify this to change the differential evolution of the internal
        # variables. For example, if the rat can only get water or food,
        # [2, 1] indicates that water decreases twice faster than food. If None,
        # equivalent to [1,1,...] with the size of satieties.
        if differential_evolution is None :
            self.differential_evolution = np.ones(self.size_actions)
        else :
            self.differential_evolution = np.array(differential_evolution)

        # Make sure Q_probas is initialized
        self.Q_probas = np.zeros(self.shape_SA) # Ensure this is initialized

    def choose_action(self, state):
        # Q_probas are computed here, so this method implicitly makes them available
        # You might want to store Q_probas as an instance variable if you want to plot them directly
        # For now, it's computed internally.
        
        # Computing action probabilities (softmax)
        exp_Q = np.exp(self.Q[state]*self.beta)
        self.Q_probas[state] = exp_Q / np.sum(exp_Q) # Store for potential plotting later if needed

        # Epsilon-greedy exploration or full softmax
        # (The original code doesn't show explicit epsilon-greedy,
        # relies on beta for softmax exploration)
        return np.random.choice(self.size_actions, p=self.Q_probas[state])

    def get_internal_reward(self, action, reward_env):
        # ... (rest of the get_internal_reward function remains the same)
        # Ensure that `new_satiation` is properly calculated and used.
        # This function is crucial for how the agent learns from environmental rewards.
        # It's already there in your original code.
        new_satiation = self.satiations.copy()
        
        # Decrease all satiations
        new_satiation -= self.differential_evolution*self.basic_step

        # Increase satiation for the chosen action if reward_env is 1
        if reward_env == 1:
            new_satiation[action] += self.increase_decrease_ratio*self.basic_step

        # Capping satiations to be between 0 and 2
        satiation_too_high = new_satiation > 2
        satiation_too_low = new_satiation < 0
        bad_satiation = np.logical_or(satiation_too_high,
                                              satiation_too_low)
        if np.any(bad_satiation):
            new_satiation[new_satiation>2]=2
            new_satiation[new_satiation<0]=0
            internal_reward = -1
            return internal_reward

        # Else, returns the weighted sum of the satiation variations
        satiation_variation = new_satiation - self.satiations
        all_priorities = self.priorities.copy() # Ensure this is used
        internal_reward = np.sum(all_priorities*satiation_variation)*10
        self.satiations = new_satiation # Update satiations here!
        return internal_reward


    def update_priority_level(self, action, reward_env):
        learning_rate_priority = 0.1
        self.priorities[action] *= (1-learning_rate_priority)
        self.priorities[action] += (1-reward_env)*learning_rate_priority
        self.priorities[self.priorities<0]=0.

        # The original code had self.priorities[0]=0. - be careful if this is intentional or a bug.
        # If action 0 always has 0 priority, it might impact learning.
        # For a maze, all actions (movements) usually have some priority.
        # If your maze doesn't have a "no reward" action corresponding to original index 0, consider removing or adapting this.
        # self.priorities[0]=0. # Consider commenting this out unless action 0 is truly special

        # No change
        if not self.changing_priorities :
            self.priorities = 1/2 * np.ones(self.size_actions)


    def model_free_update(self, state, reward, new_state, action):
        
        # Update the old value multiplying it with (1-alpha)
        self.Q[state][action] *= (1 - self.alpha)

        # Add the new value with the learning rate alpha
        self.Q[state][action] += self.alpha*reward

    def learn(self, state, reward, new_state, action):
        internal_reward = self.get_internal_reward(action, reward)
        self.update_priority_level(action, reward) # Pass original reward_env
        self.model_free_update(state, internal_reward, new_state, action) # Use internal_reward for Q-update
        return internal_reward