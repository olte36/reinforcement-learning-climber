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
    route = routes.generate_random_route(200, 400, 20)
    #route = routes.generate_simple_route(250, 400, step=70)
    env = climber_env.ClimberEnv(route=route, max_transitions=30, climb_direction="bt")

    env.reset()
    env.render()
    #time.sleep(20)
    #sys.exit()

    print("start learning")
    model = q_learning.SimpleQLearningAgent(env=env, gamma=0.9)
    model.learn(total_timesteps=50000, log_step=10000)

    # play
    for _ in range(3):
        done = False
        print("new episode")
        state = env.reset()
        while not done:
            action = model.predict(state)
            state, reward, done, info = env.step(action)
            route_dist=info["route_dist"]
            transitions_done=info["transitions_done"]
            print(f"reward={reward} route_dist={route_dist} transitions_done={transitions_done}")
            env.render()
            time.sleep(1)

        #for i in pygame.event.get():
        #    if i.type == pygame.constants.QUIT:
        #        env.close()
        #        sys.exit()
    
    env.close()
