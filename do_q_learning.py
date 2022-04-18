import numpy as np
import gym
import random
import time
import sys
import q_learning
import climber_model
import climber_env
import pygame



if __name__ == '__main__':

    #env = gym.make("Taxi-v3")
    climber = climber_model.Climber()
    env = climber_env.ClimberEnv(climber)


    # Hyperparameters
    # alpha = 0.1
    # gamma = 0.6
    # epsilon = 0.1
    # print(str(env.observation_space) + " " + str(env.action_space))
    # q_table = np.zeros([env.observation_space.n, env.action_space.n])

    # # learn agent
    # for i in range(10000):
    #     state = env.reset()

    #     reward = 0
    #     done = False
        
    #     while not done:
    #         if random.uniform(0, 1) < epsilon:
    #             action = env.action_space.sample() # Explore action space
    #         else:
    #             action = np.argmax(q_table[state]) # Exploit learned values

    #         next_state, reward, done, info = env.step(action) 
            
    #         old_value = q_table[state, action]
    #         next_max = np.max(q_table[next_state])
            
    #         new_value = (1 - alpha) * old_value + alpha * (reward + gamma * next_max)
    #         q_table[state, action] = new_value

    #         state = next_state

    print("start learning")
    model = q_learning.SimpleQLearningModel(env=env)
    model.learn(total_timesteps=20000, log_step=1000)

    # play
    print("the new")
    state = env.reset()
    while True:
        for i in pygame.event.get():
            if i.type == pygame.constants.QUIT:
                env.close()
                sys.exit()
        #action = np.argmax(q_table[state])
        action = model.predict(state)
        state, reward, done, info = env.step(action)
        env.render()
        time.sleep(1)
        if done:
            print("the new")
            obs = env.reset()
