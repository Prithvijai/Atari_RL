# import libraries 
import gymnasium as gym
# for train in stack of env instead of single env
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_util import make_atari_env
import os

from ale_py import ALEInterface
#import ale_py
ale = ALEInterface()

#gym.register_envs(ale_py)

import time
episodes = 2
env = gym.make('ALE/Breakout-v5', render_mode="human")

try:
    for episode in range(1, episodes + 1):
        obs, _ = env.reset()
        done = False
        score = 0

        while not done:
            action = env.action_space.sample()
            obs, reward, done, truncated, info = env.step(action)
            score += reward  # Keep accumulating the score
            time.sleep(0.02)
        print(f'Episode: {episode}, Score: {score}')
finally:
    env.close()
