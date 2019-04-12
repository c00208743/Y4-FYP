import random
from pygame.constants import K_LEFT, K_RIGHT
import cartpole
from pygame_player_two import PyGamePlayer
import matplotlib.pyplot as plt
import numpy as np
import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
from statistics import mean, median
from collections import Counter


class RandomCartPolePlayer(PyGamePlayer):
    def __init__(self):
        """
        Plays CartPole by choosing the next move with the Networks assumed best probability
        """
        super(RandomCartPolePlayer, self).__init__(run_real_time=True)
        self.last_score = 0
        self.training_games = 100 #amount of training games / samples
        self.num_actions = 2 #agent can choose to move left or right
        self.alpha = 1e-3 #learning rate
        self.goal_steps = 500 # amount of frames / choices for the agent to make
        self.score_requirement = 30 # score requirement to be used as training data
        self.play_games = 10 # how many games to be played
        self.training = True
        self.playing = False
        self.training_data = []
        self.scores = [] # array of scores at the end of the game
        self.accepted_scores = []
        self.play_scores = []
        self.choices = []
        self.game_memory =[]


    def get_keys_pressed(self, screen_array, feedback, terminal):

        ##choose action randomly for training data
        if self.training ==True:
            self.action_index = random.randrange(0, self.num_actions)
        else:
            ##else use NN to predict action
            if len(self.prev_obseration) == 0:
                self.action_index = random.randrange(0, self.num_actions)
            else:
                self.action_index = np.argmax(self.model.predict(self.prev_obseration.reshape(-1, len(self.prev_obseration), 1))[0])
            self.choices.append(self.action_index)

        if self.action_index == 0:
            action = [K_LEFT]
        elif self.action_index == 1:
            action = [K_RIGHT]

        return action

    def get_observation(self):
        self.observation = cartpole.get_state()
        
        ## remember env and action choosen 
        if self.training ==True:
            if len(self.prev_obseration) > 0:
                self.game_memory.append([self.prev_obseration, self.action_index])
            #print(self.game_memory)
            self.prev_obseration = self.observation
        else:
            self.prev_obseration = self.observation
            self.game_memory.append([self.observation, self.action_index])


        self.reward = cartpole.get_score()
        self.score += self.reward

        return self.score

    def get_score_at_the_end(self):
        ##check if game is over
        if self.training ==True:
            if cartpole.get_end() == True:
                ## if score is more than requirement use as training data
                if self.score >= self.score_requirement:
                    print("Decent Game")
                    self.accepted_scores.append(self.score)
                    #print(self.game_memory)
                    for data in self.game_memory:
                        if data[1] == 1:
                            output = [0,1]
                        elif data[1] == 0:
                            output = [1,0]

                        self.training_data.append([data[0], output])
                        #print(self.training_data)


                self.scores.append(self.score)
                self.game_memory = []
        else:
            if cartpole.get_end() == True:
                self.scores.append(self.score)

        return self.scores

    def get_feedback(self):
        # Get the difference in scores between this and the last
        # frame.
        score_change = cartpole.get_score() - self.last_score
        self.last_score = cartpole.get_score()
        #print(cartpole.get_score())

        return float(score_change), score_change == -1

    def start(self):
        super(RandomCartPolePlayer, self).start()

        #run random games for training data
        for n in range(self.training_games):
            #print("training")
            self.score = 0
            self.prev_obseration = []
            cartpole.run()

        #train model with training data
        print("End of training")
        self.training = False
        #print(self.training_data)
        self.model = self.train_model(self.training_data)

        for n in range(self.play_games):
            #print("play")
            self.score = 0
            self.prev_obseration = []
            cartpole.run()

        #print(self.training_data)
        print(self.scores)
        print(self.choices)

    ############# Neural Network #######################

    def neural_network_model(self, input_size):
        network = input_data(shape = [None, input_size, 1], name='input')

        network = fully_connected(network, 3, activation='relu')
        network = dropout(network, 0.8)


        network = fully_connected(network, 2, activation='softmax')
        network = regression(network, optimizer='adam', learning_rate= self.alpha,
                            loss='categorical_crossentropy', name='targets')
        model = tflearn.DNN(network, tensorboard_dir='log')

        print("Finished Model")
        return model

    def train_model(self, training_data, model=False):
        #print(training_data)
        X = np.array([i[0] for i in training_data]).reshape(-1, len(training_data[0][0]), 1)
        Y = [i[1] for i in training_data]

        if not model:
            model = self.neural_network_model(input_size = len(X[0]))

        model.fit(X, Y, n_epoch=3, snapshot_step=500, show_metric=True,
                    run_id='openaistuff')

        return model


    ####################################


if __name__ == '__main__':
    player = RandomCartPolePlayer()
    player.start()
