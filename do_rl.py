import gym
import time

from stable_baselines3 import DQN
from stable_baselines3 import PPO


if __name__ == '__main__':
    env = gym.make("MountainCar-v0")

    #model = DQN(policy="MlpPolicy", env=env, learning_rate=1e-3)
    model = PPO(policy = "MlpPolicy",env =  env)
    model.learn(total_timesteps=250000)

    obs = env.reset()
    while True:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        env.render()
        #time.sleep(0.5)
        if done:
            obs = env.reset()
            
            