# import libraries 
import gymnasium as gym
# for train in stack of env instead of single env
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3 import A2C
import os

from ale_py import ALEInterface
ale = ALEInterface()

env = gym.make('ALE/Breakout-v5', render_mode="human")

log_path = os.path.join('./logs')
model = A2C('CnnPolicy', env, verbose=1, tensorboard_log=log_path)

# training 
model.learn(total_timesteps=10000)

a2c_path = os.path.join('./models','A2C_10000_model')
model.save(a2c_path)


