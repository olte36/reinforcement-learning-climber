import gym
import time

from stable_baselines3 import DQN
from stable_baselines3 import PPO

import routes
import climber_env

if __name__ == '__main__':
    env = gym.make("MountainCar-v0")
    #route = routes.generate_simple_route(250, 500, step=70)
    #env = climber_env.ClimberEnv(route=route, climb_direction="bt")

    model = DQN(
        policy="MlpPolicy", 
        env=env, 
        device="cpu", 
        learning_rate=4e-3,
        batch_size=128,
        buffer_size=10000,
        learning_starts=10,
        gamma=0.98,
        target_update_interval=10,
        train_freq=16,
        gradient_steps=5,
        exploration_fraction=0.1,
        exploration_final_eps=0.1,
        policy_kwargs={"net_arch": [256, 256]}
    )
    #model = PPO(policy = "MlpPolicy",env =  env)
    model.learn(total_timesteps=120000)

    obs = env.reset()
    print("new episode")
    while True:
        action, states = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        env.render()
        #time.sleep(1)
        if done:
            print("new episode")
            obs = env.reset()
            
            