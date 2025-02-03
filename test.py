import gymnasium as gym
# for train in stack of env instead of single env
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack, VecVideoRecorder
import os

from stable_baselines3 import A2C

from ale_py import ALEInterface
import ale_py
ale = ALEInterface()

#env = make_atari_env('ALE/Breakout-v5', n_envs=1, seed=0)
#env = VecFrameStack(env, n_stack=4)

video_folder = './test-videos'
os.makedirs(video_folder, exist_ok=True)
# import the trained model 
#model = A2C.load('./models/A2C_20000_model', env)

# env = VecVideoRecorder(env, video_folder, record_video_trigger=lambda x: x == 0, video_length=1000)
#print(evaluate_policy(model, env, n_eval_episodes=10, render=True))

import time
episodes = 5
# env = make_atari_env('ALE/Breakout-v5', n_envs=1, seed=0)
# env = VecFrameStack(env, n_stack=4)
env = make_atari_env('ALE/Breakout-v5', n_envs=1, seed=0)
env = VecFrameStack(env, n_stack=4)

model = A2C.load('./models/A2C_200000_4win_model', env, device='cpu')
# Monitor(test_env)
#print(evaluate_policy(model, env, n_eval_episodes=10, render=True))
env = VecVideoRecorder(env, video_folder, record_video_trigger=lambda x: x == 0, video_length=1000)

try:
    # Evaluate the policy and record the video
    mean_reward, std_reward = evaluate_policy(model, env, render=False, return_episode_rewards=True)
    print(f"Evaluation results - Mean reward: {mean_reward}, Std reward: {std_reward}")
    
finally:
    # Properly close the environment to save the video and release resources
    env.close()
# try:
#     for episode in range(1, episodes + 1):
#         obs, _ = env.reset()
#         done = False
#         score = 0

#         while not done:
#             env.render()
#             #action = env.action_space.sample()
#             action,_ = model.predict(obs)
#             #obs, reward, done, truncated, info = env.step(action)
#             obs, reward, done, tun, info = env.step(action)
#             score += reward  # Keep accumulating the score
#             #time.sleep(0.02)
#         print(f'Episode: {episode}, Score: {score}')
# finally:
#     env.close()