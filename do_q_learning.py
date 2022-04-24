import time
import sys

import gym
import pygame

import climber_env
import q_learning
import routes



if __name__ == '__main__':

    #env = gym.make("Taxi-v3")
    #env = gym.envs.toy_text.TaxiEnv()
    route = routes.generate_random_route(250, 500, 50)
    #route = routes.generate_simple_route(250, 500, step=70)
    env = climber_env.ClimberEnv(route=route, climb_direction="bt")

    env.reset()
    env.render()
    #time.sleep(20)
    #sys.exit()

    print("start learning")
    model = q_learning.SimpleQLearningAgent(env=env)
    model.learn(total_timesteps=20000, log_step=1000)

    # play
    print("the new")
    state = env.reset()
    while True:
        #action = np.argmax(q_table[state])
        action = model.predict(state)
        state, reward, done, info = env.step(action)
        env.render(mode="human")
        time.sleep(1)
        if done:
            print("the new")
            obs = env.reset()

        for i in pygame.event.get():
            if i.type == pygame.constants.QUIT:
                env.close()
                sys.exit()







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
