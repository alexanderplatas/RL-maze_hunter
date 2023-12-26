import gym
import numpy as np
from stable_baselines3.common.env_checker import check_env
import cv2
from stable_baselines3 import PPO
from MazeEnv import MazeEnv
from TrainingEnv import TrainingEnv


class Wrapper(gym.Wrapper):

    def __init__(self, maze_env):
        super().__init__(maze_env)
        self.env = maze_env

    def step(self, actions):

        observation, reward, done, truncated, info = self.env.step(str(actions[0]) + str(actions[1]))

        hunter_reward, prey_reward = 0, 0

        if truncated:

            if info['loser'] == 'Hunter':
                hunter_reward = -10
            elif info['loser'] == 'Prey':
                prey_reward = -10

        elif done:

            if info['winner'] == 'Hunter':
                hunter_reward = 10
            elif info['winner'] == 'Prey':
                prey_reward = 10

        return [observation[:6], observation[6:]], [prey_reward, hunter_reward], done, truncated, info

    def reset(self):

        observation, information = self.env.reset()
        return [observation[:6], observation[6:]], information


# env = Wrapper(MazeEnv(render=True))
# obs, _ = env.reset()
# prey_obs = obs[0]
# hunter_obs = obs[1]

# while not done:
#     action, _states = prey_model.predict(prey_obs)
#     obs, reward, done, truncated, info = env.step([action, -1])
#
#     if truncated:
#         done = False
#         obs, _ = env.reset()
#         prey_obs = obs[0]
#         hunter_obs = obs[1]

############ Train Model #############

ppo_model = PPO("MlpPolicy", TrainingEnv(render=True), verbose=0)
# ppo_model = PPO.load("prey_model.model", env=maze_env)

ppo_model.learn(total_timesteps=200)
