import os
import gymnasium as gym
from gymnasium.wrappers import RecordVideo  # Import the video recorder wrapper
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3 import A2C
from stable_baselines3.common.vec_env import VecFrameStack, VecVideoRecorder
from ale_py import ALEInterface
ale = ALEInterface()
# Create a directory for the videos if it doesn't exist
video_folder = './videos'
os.makedirs(video_folder, exist_ok=True)

# Create the environment with a render mode that produces images (not human)
# env = gym.make('ALE/Breakout-v5', render_mode="rgb_array")
env = make_atari_env('ALE/Breakout-v5', n_envs=4, seed=0)
env = VecFrameStack(env, n_stack=4)

# Wrap the environment with VecVideoRecorder
env = VecVideoRecorder(env, video_folder, record_video_trigger=lambda x: x == 0, video_length=1000)

# Set up logging path for tensorboard logs if needed
log_path = os.path.join('./logs')

# Initialize the model with the wrapped environment
model = A2C('CnnPolicy', env, verbose=1, tensorboard_log=log_path)

# Train the model
model.learn(total_timesteps=200000)

# Save the model
a2c_path = os.path.join('./models', 'A2C_200000_4win_model')
model.save(a2c_path)
