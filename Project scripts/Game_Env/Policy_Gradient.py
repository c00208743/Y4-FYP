import random
from pygame.constants import K_LEFT, K_RIGHT
import cartpole
from pygame_player_four import PyGamePlayer
import matplotlib.pyplot as plt
import numpy as np


class PGCartPolePlayer(PyGamePlayer):
    def __init__(self):
        """
        Plays CartPole by choosing moves with the best policy
        """
        super(PGCartPolePlayer, self).__init__(run_real_time=True)
        self.last_score = 0

        self.gamma = 0.99
        self.alpha = 1e-2
        self.training_games = 1000
        self.playing_games = 10
        self.playing_reward = 0
        self.num_actions = 2 # The number of possible actions (left, right)
        self.hidden_neurons = 10 # number of hidden layer neurons
        self.batch_size = 5 # how many episodes do we param update
        self.decay_rate = 0.99
        self.input_neuron = 4 # how many neurons are in the input player
        self.training = True

        self.episode_number = 0

        ##Create the model
        self.model = {}

        self.model['W1'] = np.random.randn(self.hidden_neurons,self.input_neuron) / np.sqrt(self.input_neuron)
        self.model['W2'] = np.random.randn(self.hidden_neurons) / np.sqrt(self.hidden_neurons)

        self.grad_buffer = { k : np.zeros_like(v) for k,v in self.model.iteritems() } # update buffers that add up gradients over a batch
        self.rmsprop_cache = { k : np.zeros_like(v) for k,v in self.model.iteritems() } # rmsprop memory

        self.xs, self.hs, self.dlogps, self.drs = [],[],[],[]

        self.reward_sum = 0
        self.prev_score = 0

        ##capture play_scores
        self.play_scores = []

    def get_state(self):
        ##observe the current state of the game
        self.observation = cartpole.get_state()

        return self.observation

    def get_keys_pressed(self, screen_array, feedback, terminal):
        if self.training ==True:
            aprob, h = self.policy_forward(self.observation)
            self.action_index = 1 if np.random.uniform() < aprob else 0

            #print("State : " , self.observation)
            #print("Hidden : " , h)

            self.xs.append(self.observation) #observation
            self.hs.append(h) #hidden statek
            y = 1 if self.action_index == 1 else 0 # a "fake label"

            self.dlogps.append(y - aprob) # grad that encourages the action that was taken to be taken
        else:
            aprob, _ = self.policy_forward(self.observation)
            self.action_index = 1 if np.random.uniform() < aprob else 0
            print("probability", aprob)
            print(self.action_index)

        #print("probability", aprob)
        #print(self.action_index)

        if self.action_index == 0:
            action = [K_LEFT]
        elif self.action_index == 1:
            action = [K_RIGHT]

        return action

    def get_new_state(self):
        ##observe the current state of the game
        self.new_observation = cartpole.get_state()
        ##check score
        self.reward = cartpole.get_score() - self.prev_score
        self.prev_score = cartpole.get_score()

        ##check if game ended
        self.done = cartpole.get_end()
        self.reward_sum += self.reward

        if self.training ==True:
            self.drs.append(self.reward) # record reward (has to be done after we call step() to get reward for previous action)

            if self.done: # an episode finished
            # stack together all inputs, hidden states, action gradients, and rewards for this episode
                self.epx = np.vstack(self.xs)
                eph = np.vstack(self.hs)
                epdlogp = np.vstack(self.dlogps)
                epr = np.vstack(self.drs)

                # reset array memory
                self.xs, self.hs, self.dlogps, self.drs = [],[],[],[]

                # compute the discounted reward backwards through time
                discounted_epr = self.discount_rewards(epr)
                # standardize the rewards to be unit normal (helps control the gradient estimator variance)
                discounted_epr = discounted_epr - np.mean(discounted_epr)
                discounted_epr /= np.std(discounted_epr)

                epdlogp *= discounted_epr # modulate the gradient with advantage (PG magic happens right here.)

                grad = self.policy_backward(eph, epdlogp)
                for k in self.model: self.grad_buffer[k] += grad[k] # accumulate grad over batch

                # perform rmsprop parameter update every batch_size episodes
                if self.episode_number % self.batch_size == 0:
                    for k,v in self.model.iteritems():
                        g = self.grad_buffer[k] # gradient
                        self.rmsprop_cache[k] = self.decay_rate * self.rmsprop_cache[k] + (1 - self.decay_rate) * g**2
                        self.model[k] = self.alpha * g / (np.sqrt(self.rmsprop_cache[k]) + 1e-5)
                        self.grad_buffer[k] = np.zeros_like(v) # reset batch gradient buffer

                self.reward_sum = 0
                self.episode_number += 1
                self.prev_score = 0
        else:
            if self.done or self.reward_sum >= 200:
                self.play_scores.append(self.reward_sum)
                self.reward_sum = 0
                self.prev_score = 0

        return self.observation


    def get_feedback(self):
        # Get the difference in scores between this and the last
        # frame.
        if self.last_score > cartpole.get_score():
            self.last_score = 0

        self.score_change = cartpole.get_score() - self.last_score
        self.last_score = cartpole.get_score()
        #print(cartpole.get_score())

        return float(self.score_change), self.score_change == -1

    def start(self):
        super(PGCartPolePlayer, self).start()

        for n in range(self.training_games):
            cartpole.run()

        self.training = False

        for n in range(self.playing_games):
            cartpole.run()

        print(self.play_scores)

    ############# Policy Gradient #######################
    def sigmoid(self, x):
    	return 1.0 / (1.0 + np.exp(-x)) # sigmoid "squashing" function to interval [0,1]

    def discount_rewards(self, r):
    	#""" take 1D float array of rewards and compute discounted reward """
    	discounted_r = np.zeros_like(r)
    	running_add = 0
    	for t in reversed(xrange(0, r.size)):
    		running_add = running_add * self.gamma + r[t]
    		discounted_r[t] = running_add
    	return discounted_r

    def policy_forward(self, x):
        #print("State : ", x)
        #print("First Set of Weights : ", self.model['W1'])
        #if all(v ==0 for v in self.model['W1']):
            #self.model['W1'] =
    	h = np.dot(self.model['W1'], x)
        #print("Hidden States : ", h)
    	h[h<0] = 0 # ReLU nonlinearity
    	logp = np.dot(self.model['W2'], h)
    	p = self.sigmoid(logp)
    	return p, h # return probability of taking action 1, and hidden state

    def policy_backward(self, eph, epdlogp):
    	#""" backward pass. (eph is array of intermediate hidden states) """
    	dW2 = np.dot(eph.T, epdlogp).ravel()
    	dh = np.outer(epdlogp, self.model['W2'])
    	dh[eph <= 0] = 0 # backpro prelu
    	dW1 = np.dot(dh.T, self.epx)
    	return {'W1':dW1, 'W2':dW2}
    ####################################


if __name__ == '__main__':
    player = PGCartPolePlayer()
    player.start()
