import random

import numpy as np


class SimpleQLearningAgent:

    def __init__(self, env, alpha = 0.1, gamma = 0.6, epsilon = 0.1):
        self._env = env
        self._alpha = alpha
        self._gamma = gamma
        self._epsilon = epsilon
        self._q_table = None


    def learn(self, total_timesteps, log_step=None):
        self._q_table = np.zeros([self._env.observation_space.n, self._env.action_space.n])
        x = []
        y = []
        eps_step = self._epsilon / total_timesteps
        for i in range(total_timesteps):
            if log_step is not None and i % log_step == 0:
                print("step " + str(i))

            state = self._env.reset()
            reward = 0
            done = False
            counter = 0
            ep_rewords = []

            while not done:
                counter += 1
                if random.uniform(0, 1) < self._epsilon:
                    action = self._env.action_space.sample() # Explore action space
                else:
                    action = np.argmax(self._q_table[state]) # Exploit learned values

                next_state, reward, done, info = self._env.step(action) 
                ep_rewords.append(reward)

                old_value = self._q_table[state, action]
                next_max = np.max(self._q_table[next_state])
                
                new_value = (1 - self._alpha) * old_value + self._alpha * (reward + self._gamma * next_max)
                self._q_table[state, action] = new_value

                state = next_state
                #self._epsilon -= eps_step

            x.append(i)
            y.append(np.mean(ep_rewords))

        return x, y

    
    def predict(self, state):
        return np.argmax(self._q_table[state])