import gym
import random
import numpy as np
import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
from statistics import mean, median
from collections import Counter

import time

LR = 1e-3 # learning rate
env = gym.make('CartPole-v0')
env.reset()
goal_steps = 500 # limit on frames and game choices
score_requirement = 50 # score limit for training data
initial_games = 10000 # amount of games to train

def some_random_games_first(): # play game with random movements
    for episode in range(5):
        env.reset()
        for t in range(goal_steps):
            env.render()
            action = env.action_space.sample()
            observation, reward, done, info = env.step(action)
            if done:
                break
##should render but doesnt
##some_random_games_first()

def initial_population():
    training_data = [] # training data sample
    scores = [] # all scores
    accepted_scores = [] # scores of successful random choice games
    for _ in range(initial_games):
        score = 0
        game_memory = []
        prev_obseration = []
        for _ in range(goal_steps):
            action = random.randrange(0, 2) # decide action
            observation, reward, done, info = env.step(action) # move forward a frame with action

            if len(prev_obseration) > 0:
                game_memory.append([prev_obseration, action]) # remember action choosen in previous state

            prev_obseration = observation
            score +=reward

            if done:
                break

        if score >= score_requirement:
            accepted_scores.append(score)
            for data in game_memory:
                if data[1] == 1:
                    output = [0,1]
                elif data[1] == 0:
                    output = [1,0]

                training_data.append([data[0], output]) # add to traing data
                #print(training_data)

        env.reset()
        scores.append(score)

    training_data_save = np.array(training_data)
    np.save('save.npy', training_data_save)


    print('Average Accepted Score: ', mean(accepted_scores))
    print('Median Accepted Score: ', median(accepted_scores))
    print(Counter(accepted_scores))

    return training_data

def neural_network_model(input_size):
    network = input_data(shape = [None, input_size, 1], name='input') #input layer

    network = fully_connected(network, 3, activation='relu') # hidden layer
    network = dropout(network, 0.8)


    network = fully_connected(network, 2, activation='softmax') # output layer
    network = regression(network, optimizer='adam', learning_rate=LR,
                        loss='categorical_crossentropy', name='targets')
    model = tflearn.DNN(network, tensorboard_dir='log')

    return model

def train_model(training_data, model=False):
    #print(training_data)
    X = np.array([i[0] for i in training_data]).reshape(-1, len(training_data[0][0]), 1)
    Y = [i[1] for i in training_data]

    if not model:
        model = neural_network_model(input_size = len(X[0])) # train model with training sample

    model.fit(X, Y, n_epoch=3, snapshot_step=500, show_metric=True,
                run_id='openaistuff')

    return model

training_data = initial_population()
model = train_model(training_data)

scores = []
choices = []

for each_game in range(10):
    score = 0
    game_memory = []
    prev_obs = []
    env.reset() # reset environment
    for _ in range(goal_steps):
        env.render()
        time.sleep(0.1) # render slower --> demo purposes
        if len(prev_obs) == 0:
            action = random.randrange(0, 2) # choose random action
            #action = env.action_space.sample()
        else:
            action = np.argmax(model.predict(prev_obs.reshape(-1, len(prev_obs), 1))[0]) # use nn to predict an action
        choices.append(action)

        new_observation, reward, done, info = env.step(action)
        prev_obs = new_observation
        game_memory.append([new_observation, action]) # record how action influenced the env
        score += reward
        if done:
            break
    scores.append(score)

print('Average Score', sum(scores)/len(scores))
print('Choice 1: {}, Choice 0 : {}'.format(choices. count(1)/len(choices),
        choices.count(0)/len(choices)))
