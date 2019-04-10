import random
from pygame.constants import K_LEFT, K_RIGHT
import cartpole
from pygame_player_three import PyGamePlayer
import matplotlib.pyplot as plt
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

class DQNCartPolePlayer(PyGamePlayer):
    def __init__(self):
        """
        Plays CartPole by choosing moves randomly
        """
        super(DQNCartPolePlayer, self).__init__(run_real_time=True)
        self.last_score = 0
        self.gamma = 0.95
        self.alpha = 0.001
        self.memory_size = 10**5
        self.memory = deque(maxlen=self.memory_size)
        self.batch_size = 20
        self.epsilon = 1.0
        self.epsilon_max = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.initial_games = 1000000
        self.num_actions = 2
        self.observation_space = 4

        ##lazy jamie
        self.scores = []

        ##Create Model
        self.model = Sequential()
        ##input size = 4 ... cart pos, pole pos, cart and pole vel
        self.model.add(Dense(24, input_shape=(self.observation_space, ), activation="relu"))
        self.model.add(Dense(24, activation="linear"))
        self.model.add(Dense(self.num_actions, activation="linear"))
        self.model.compile(loss="mse", optimizer=Adam(lr=self.alpha))

    def get_state(self):
        ##should only take state if game has just started
        self.score = cartpole.get_score()
        if self.score == 0:
            #print("Begining of game")
            self.state = cartpole.get_state()
            self.state = np.reshape(self.state, [1, self.observation_space])

        if self.score == 1:
            #print("Begining of game")
            self.state = cartpole.get_state()
            self.state = np.reshape(self.state, [1, self.observation_space])

        #print(self.score)


        ##retrun state
        return 0

    def get_keys_pressed(self, screen_array, feedback, terminal):

        ##choose action randomly
        if np.random.rand() < self.epsilon:
            self.action_index = random.randrange(0, self.num_actions)
        else:
            q_values = self.model.predict(self.state)
            self.action_index = np.argmax(q_values[0])
            #print("Smart Decision")


        if self.action_index == 0:
            action = [K_LEFT]
        elif self.action_index == 1:
            action = [K_RIGHT]

        return action

    def get_new_state(self):
        ##should only take state of game after an action

        ##check score
        self.reward = cartpole.get_score()
        ##check if game ended
        terminal = cartpole.get_end()

        ##check new state
        self.new_state = cartpole.get_state()

        self.reward = self.reward if not terminal else -self.reward
        self.new_state = np.reshape(self.new_state, [1, self.observation_space])

        self.remember(self.state, self.action_index, self.reward, self.new_state, terminal)

        self.state = self.new_state

        self.experience_replay()

        ##if game end record scores
        if terminal == True:
            self.scores.append(self.reward)

        return self.state


    def get_feedback(self):
        # Get the difference in scores between this and the last
        # frame.
        score_change = cartpole.get_score() - self.last_score
        self.last_score = cartpole.get_score()
        #print(cartpole.get_score())

        return float(score_change), score_change == -1

    def start(self):
        super(DQNCartPolePlayer, self).start()

        for n in range(self.initial_games):
            self.score = 0
            cartpole.run()

        print(self.scores)

    ############# DQN #######################
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def experience_replay(self):
        if len(self.memory) < self.batch_size:
            return
        batch = random.sample(self.memory, self.batch_size)
        for state, action, reward, state_next, terminal in batch:
            q_update = reward
            if not terminal:
                q_update = (reward + self.gamma * np.amax(self.model.predict(state_next)[0]))
            q_values = self.model.predict(state)
            q_values[0][action] = q_update
            self.model.fit(state, q_values, verbose=0)
        self.epsilon *= self.epsilon_decay
        self.epsilon = max(self.epsilon_min, self.epsilon)


    ####################################


if __name__ == '__main__':
    player = DQNCartPolePlayer()
    player.start()
