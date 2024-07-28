import time
import sys

import numpy as np
import gym
import pygame
import matplotlib.pyplot as plt

import climber_env
import q_learning
import routes



if __name__ == '__main__':

    #env = gym.make("Taxi-v3")
    #env = gym.envs.toy_text.TaxiEnv()
    route = routes.generate_random_route(200, 500, 25)
    #route = routes.generate_simple_route(250, 500, step=70)
    belay_points = np.array([[40, 120, 0], [70, 200, 0], [120, 250, 0], [100, 290, 0], [30, 350, 0]])
    #belay_points = np.array([[40, 120, 0]])
    env = climber_env.ClimberEnv(route=route, belay_points=belay_points, max_transitions=70, climb_direction="bt")

    env.reset()
    #env._clipped = env._clipped + True
    env.render()
    #time.sleep(20)
    #input()
    #sys.exit()

    print("start learning")
    model = q_learning.SimpleQLearningAgent(env=env, gamma=0.95)
    x_l, y_l = model.learn(total_timesteps=100000, log_step=10000)

    # play
    x = []
    y = []
    for i in range(1):
        ep_rewords = []
        done = False
        #print("new episode")
        state = env.reset()
        c = 0
        while not done and c < 100:
            c += 1
            action = model.predict(state)
            state, reward, done, info = env.step(action, v=True)
            ep_rewords.append(reward)
            route_dist=info["route_dist"]
            transitions_done=info["transitions_done"]
            #print(f"reward={reward} route_dist={route_dist} transitions_done={transitions_done}")
            env.render()
            time.sleep(1)

        x.append(i)
        y.append(np.mean(ep_rewords))

    plt.scatter(x_l, y_l, s=1, c="#000000")
    plt.xlabel('Episodes')
    plt.ylabel('Mean rewards')
    #plt.title('My first graph!')
    plt.show()
    
    env.close()
