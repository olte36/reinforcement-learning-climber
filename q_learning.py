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
        #self._q_table = np.zeros([self._env.observation_space.n, self._env.action_space.n])
        self._q_table = np.empty((0, ), dtype=[('state', 'i4'), ('action', 'i4'), ('value', 'f4')])

        for i in range(total_timesteps):
            if log_step is not None and i % log_step == 0:
                print("step " + str(i))

            state = self._env.reset()
            reward = 0
            done = False
            counter = 0

            while not done and counter < 50:
                counter += 1
                if self._q_table[self._q_table["state"] == state].size == 0:
                    for ac in range(self._env.action_space.start, self._env.action_space.start + self._env.action_space.n):
                        self._q_table = np.append(self._q_table, np.array([(state, ac, 0)], dtype=self._q_table.dtype))

                if random.uniform(0, 1) < self._epsilon:
                    action = self._env.action_space.sample() # Explore action space
                else:
                    #action = np.argmax(self._q_table[state]) # Exploit learned values
                    action = self._q_table[self._q_table[self._q_table["state"] == state]["value"].argmax()]["action"]
                
                try:
                    next_state, reward, done, info = self._env.step(action)
                except Exception as ex:
                    raise ex

                if self._q_table[self._q_table["state"] == next_state].size == 0:
                    for ac in range(self._env.action_space.start, self._env.action_space.start + self._env.action_space.n):
                        self._q_table = np.append(self._q_table, np.array([(next_state, ac, 0)], dtype=self._q_table.dtype)) 
                
                #old_value = self._q_table[state, action]
                old_value = self._q_table[np.logical_and(self._q_table["state"] == state, self._q_table["action"] == action)]["value"][0]

                #next_max = np.max(self._q_table[next_state])
                next_max = self._q_table[self._q_table["state"] == next_state]["value"].max()
                
                new_value = (1 - self._alpha) * old_value + self._alpha * (reward + self._gamma * next_max)
                self._q_table[["value"]][np.logical_and(self._q_table["state"] == state, self._q_table["action"] == action)] = new_value
                #self._q_table[state, action] = new_value

                state = next_state

        return self

    
    def predict(self, state):
        return self._q_table[self._q_table[self._q_table["state"] == state]["value"].argmax()]["action"]