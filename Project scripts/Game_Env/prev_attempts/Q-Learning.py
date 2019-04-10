import random

import pygame
from pygame.constants import K_LEFT, K_RIGHT, QUIT, KEYDOWN, KEYUP
import matplotlib.pyplot as plt
import cartpole

import numpy as np
import os


#################################################################


def function_intercept(intercepted_func, intercepting_func):
    """Intercepts a method call and calls the supplied intercepting_func
    with the result of it's call and it's arguments.  Stolen wholesale
    from PyGamePlayer.
    - param intercepted_func: The function we are going to intercept
    :param intercepting_func: The function that will get called after
    the intercepted func. It is supplied the return value of the
    intercepted_func as the first argument and it's args and kwargs.
    :return: a function that combines the intercepting and intercepted
    function, should normally be set to the intercepted_functions
    location
    """

    def wrap():
        # call the function we are intercepting and get it's result
        real_results = intercepted_func()

         # call our own function a
        intercepted_results = intercepting_func(real_results)
        return intercepted_results

    return wrap



##################################################################


class QPlayer(object):

    def __init__(self):

        # The number of possible actions (left, right, no move)
        self.num_actions = 2

        # Variables for holding information about the previous
        # timestep.
        self.last_score = 0
        self.last_state = np.array([0.0, 0.0, 0.0, 0.0])
        self.last_action = np.array([1.0, 0.0, 0.0])

        # Variables for dealing with pressed keys.
        self.keys_pressed = []
        self.last_keys_pressed = []

        # Size of the observations collection.
        self.observations = []

        #amount of games
        self.initial_games = 100

        #Q-Learning
        self.max_states = 10**6
        self.gamma = 0.9
        self.alpha = 0.01
        self.epsilon = 1


    ####################################

    def play_game(self):

        # play set amount of games
        for n in range(self.initial_games):
            self.Q = self.initialize_Q()
            self.length = []
            self.reward = []
            self.epsilon = 1.0 / np.sqrt(n+1)
            cartpole.run()

    ####################################

    def max_dict(self, d):
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
            for action in range(self.num_actions):
                Q[state][action] = 0
            return Q


    ####################################
    def choose_next_action(self):

        # This code chooses the next action.  This means either
        # choosing a random action, based on the current probability
        # value, or use the NN to generate the next action.

        new_action = np.zeros([self.num_actions])

        # Check to see if we are going to use a random action or not.
        if np.random.uniform() < self.epsilon:
            action_index = random.randrange(0, self.num_actions)
        else:
            action_index = self.max_dict(self.Q[self.state])[0]


        new_action[action_index] = 1

        return action_index


    ####################################

    def get_keys_pressed(self, reward):

        # Here is where the actual work gets done.

        self.current_state = cartpole.get_state()
        self.state = self.get_state_as_string(self.assign_bins(self.current_state, bins))


        # get the next action.
        action_index = self.choose_next_action()





        self.last_state = self.current_state
        # Set the move to take, based on the action.
        if action_index == 0:
            action = [K_LEFT]
        elif action_index == 1:
            action = [K_RIGHT]
        print(action)
        return action


    ####################################


    def get_reward(self):

        # Get the difference in scores between this and the last
        # frame.
        score_change = cartpole.get_score() - self.last_score
        self.last_score = cartpole.get_score()
        #print(cartpole.get_score())

        return float(score_change)


    ####################################


    def start(self):

        # This code intercepts the regular pygame commands for
        # updating the screen, and getting keyboard inputs, and
        # redirects them to commands in this file.

        pygame.display.flip = function_intercept(pygame.display.flip,
                                                 self.on_screen_update)
        pygame.event.get = function_intercept(pygame.event.get,
                                              self.on_event_get)

        # Run the game.
        #cartpole.run()
        self.play_game()


    ####################################


    def on_event_get(self, _):

        # This code is the custom modification of the pygame.event.get
        # command.  It merely processes the current collection of
        # moves.

        key_up_events = []

        if len(self.last_keys_pressed) > 0:
            diff_list = list(set(self.last_keys_pressed) -
                             set(self.keys_pressed))
            key_up_events = [pygame.event.Event(KEYUP, {"key": x})
                             for x in diff_list]

        key_down_events = [pygame.event.Event(KEYDOWN, {"key": x})
                           for x in self.keys_pressed]

        result = key_down_events + key_up_events

        for e in _:
            if e.type == QUIT:
                result.append(e)

        return result


    ####################################


    def on_screen_update(self, _):

        # This function handles the latest information from the
        # 'player'.
        reward  = self.get_reward()

        keys = self.get_keys_pressed(reward)
        self.last_keys_pressed = self.keys_pressed
        self.keys_pressed = keys




##############################################################

# Run it!
if __name__ == '__main__':
    player = QPlayer()
    bins = player.create_bins()
    player.start()
