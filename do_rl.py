import gym
import time
import numpy as np

from stable_baselines3 import DQN
from stable_baselines3 import PPO

import routes
import climber_env

if __name__ == '__main__':
    #env = gym.make("MountainCar-v0")
    route = routes.generate_simple_route(250, 500, step=70)
    #belay_points = np.array([[40, 120, 0], [70, 200, 0], [120, 250, 0], [100, 290, 0], [30, 350, 0]])
    belay_points = np.array([[40, 120, 0]])
    env = climber_env.ClimberEnv(route=route, belay_points=belay_points, max_transitions=30, climb_direction="bt")

    # model = DQN(
    #     policy="MlpPolicy", 
    #     env=env, 
    #     device="cpu", 
    #     learning_rate=4e-3,
    #     batch_size=128,
    #     buffer_size=10000,
    #     learning_starts=10,
    #     gamma=0.98,
    #     target_update_interval=10,
    #     train_freq=16,
    #     gradient_steps=1,
    #     exploration_fraction=0.2,
    #     exploration_final_eps=0.01,
    #     policy_kwargs={"net_arch": [256, 256]},
    #     verbose=1
    # )
    model = PPO(
        policy="MlpPolicy",
        env= env,
        gamma=0.98,
        verbose=1,
        device="cpu"
    )
    model.learn(total_timesteps=100000, log_interval=10000)
    all_rewards = []
    num_episodes = 2
    for _ in range(num_episodes):
        obs = env.reset()
        print("new episode")
        done = False
        ep_rewords = []
        while not done:
            action, states = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            ep_rewords.append(reward)
            env.render()
            time.sleep(1)
            if done:
                print("new episode")
                obs = env.reset()

        all_rewards.append(sum(ep_rewords))
    
    print("Sum rewards per episode:", all_rewards)
    mean_episode_reward = np.mean(all_rewards)
    print("Mean reward:", mean_episode_reward, "Num episodes:", num_episodes)
            
            