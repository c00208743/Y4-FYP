import random

import pygame
from pygame.constants import K_LEFT, K_RIGHT, QUIT, KEYDOWN, KEYUP

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


class RandomPlayer(object):

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


    ####################################

    #def play_games(self):

        # play set amount of games
        #for n in range(self.initial_games):
            #cartpole.run()


    ####################################
    def choose_next_action(self):

        # This code chooses the next action.  This means either
        # choosing a random action, based on the current probability
        # value, or use the NN to generate the next action.

        new_action = np.zeros([self.num_actions])

        # Check to see if we are going to use a random action or not.
        action_index = random.randrange(0, self.num_actions)



        new_action[action_index] = 1

        return new_action, action_index


    ####################################

    def get_keys_pressed(self, reward):

        # This is the real work horse of the code.  Here is where the
        # actual work gets done.

        # Get the current state of the game.
        current_state = cartpole.get_state()

        # Append the latest observation to the collection of
        # observations.
        self.last_state = cartpole.get_state()
        self.observations.append([self.last_state, self.last_action,
                                  reward, current_state])


        # Reset the last state, and get the next action.
        self.last_state = current_state
        self.last_action, action_index = self.choose_next_action()


        # Set the move to take, based on the action.
        if action_index == 0:
            action = [K_LEFT]
        elif action_index == 1:
            action = [K_RIGHT]

        return action


    ####################################


    def get_reward(self):

        # Get the difference in scores between this and the last
        # frame.
        score_change = cartpole.get_score() - self.last_score
        self.last_score = cartpole.get_score()

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
        for n  in range(self.initial_games):
            cartpole.run()


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
    player = RandomPlayer()
    player.start()
