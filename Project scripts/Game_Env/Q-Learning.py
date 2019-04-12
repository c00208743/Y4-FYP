import random
from pygame.constants import K_LEFT, K_RIGHT
import cartpole
from pygame_player import PyGamePlayer
import matplotlib.pyplot as plt
import numpy as np
import os

class QCartPolePlayer(PyGamePlayer):
    def __init__(self):
        """
        Plays CartPole by choosing moves randomly
        """
        super(QCartPolePlayer, self).__init__(run_real_time=True)
        self.last_score = 0
        self.max_states = 10**5 # max amount of states in q-table
        self.gamma = 0.9   # discount rate
        self.alpha = 0.01 # learning rate
        self.epsilon = 1.0 # Epsilon Greedy Strategy
        self.initial_games = 10000
        # The number of possible actions (left, right)
        self.num_actions = 2
        self.Q = self.initialize_Q()
        self.length = []
        self.reward = []

    def get_state(self):

        self.observation = cartpole.get_state()
        self.state = self.get_state_as_string(self.assign_bins(self.observation, bins)) # set state to string to use as key in dict

        return self.state

    def get_keys_pressed(self, screen_array, feedback, terminal):

        ##choose action randomly or use EGS
        if np.random.uniform() < self.epsilon:
            self.action_index = random.randrange(0, self.num_actions) # explore
        else:
            self.action_index = self.max_dict(self.Q[self.state])[0] # exploit

        if self.action_index == 0:
            action = [K_LEFT]
        elif self.action_index == 1:
            action = [K_RIGHT]

        return action

    def get_new_state(self):
        # observe the env after a descion has been made
        self.new_observation = cartpole.get_state()
        self.new_state = self.get_state_as_string(self.assign_bins(self.new_observation, bins))
        #print(self.new_state)


        return self.new_state

    def q_learn(self):
        #get reward
        self.reward = cartpole.get_score()
        #if game ends and reward < 200 then reward = -300
        #if (cartpole.get_end() == True and self.reward < 100):
            #self.reward = -300
        #print(self.reward)


        a1, max_q_s1a1 = self.max_dict(self.Q[self.new_state])

        self.Q[self.state][self.action_index] += self.alpha*(self.reward + self.gamma*max_q_s1a1 - self.Q[self.state][self.action_index])
        return self.new_state


    def get_feedback(self):
        # Get the difference in scores between this and the last
        # frame.
        score_change = cartpole.get_score() - self.last_score
        self.last_score = cartpole.get_score()
        #print(cartpole.get_score())

        return float(score_change), score_change == -1

    def start(self):
        super(QCartPolePlayer, self).start()

        #run(screen_width=640, screen_height=480)
        for n in range(self.initial_games):
            self.epsilon = 1.0 / np.sqrt(n+1)
            cartpole.run()

    ############# Q-Learning #######################

    def max_dict(self, d): # returns max q value with key
    	max_v = float('-inf')
    	for key, val in d.items():
    		if val > max_v:
    			max_v = val
    			max_key = key
    	return max_key, max_v

    def create_bins(self):
    	# obs[0] -> cart position --- -4.8 - 4.8
    	# obs[1] -> cart velocity --- -inf - inf
    	# obs[2] -> pole angle    --- -41.8 - 41.8
    	# obs[3] -> pole velocity --- -inf - inf

    	bins = np.zeros((4,10))
    	bins[0] = np.linspace(-4.8, 4.8, 10)
    	bins[1] = np.linspace(-5, 5, 10)
    	bins[2] = np.linspace(-.418, .418, 10)
    	bins[3] = np.linspace(-5, 5, 10)
        #print(bins)
    	return bins

    def assign_bins(self, observation, bins):
    	state = np.zeros(4)
    	for i in range(4):
    		state[i] = np.digitize(observation[i], bins[i])
    	return state

    def get_state_as_string(self, state):
    	string_state = ''.join(str(int(e)) for e in state)
    	return string_state

    def get_all_states_as_string(self):
    	states = []
        #print(self.max_states)
    	for i in range(self.max_states):
            states.append(str(i).zfill(4))
    	return states

    def initialize_Q(self):
        Q = {}

        all_states = self.get_all_states_as_string()
        for state in all_states:
            Q[state] = {}
            for action in range(self.num_actions): # make q-table with max states and action
                #print(self.num_actions)
                Q[state][action] = 0
        return Q


    ####################################


if __name__ == '__main__':
    player = QCartPolePlayer()
    bins = player.create_bins()
    player.start()
