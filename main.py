import sys

import cv2
import gym
from stable_baselines3 import PPO

from envs.HunterTrainingEnv import HunterTrainingEnv
from envs.MazeEnv import MazeEnv
from envs.PreyTrainingEnv import PreyTrainingEnv


##################### PARAMS #########################

TOTAL_TIMESTEPS = 10_000
NUM_EVALUATIONS = 10
NUM_EPISODES_PER_EVAL = 5

################## MAZE WRAPPER ######################

class MazeWrapper(gym.Wrapper):

    def __init__(self, maze_env):
        super().__init__(maze_env)
        self.env = maze_env

    def step(self, actions):
        observation, reward, done, truncated, information = self.env.step(actions)

        prey_observation = [observation[:25][i:i + 5] for i in range(0, len(observation[:25]), 5)]
        hunter_observation = observation[25:]

        return [prey_observation, hunter_observation], reward, done, truncated, information

    def reset(self):
        observation, information = self.env.reset()

        prey_observation = [observation[:25][i:i + 5] for i in range(0, len(observation[:25]), 5)]
        hunter_observation = observation[25:]

        return [prey_observation, hunter_observation], information


################ EVALUATE POLICY #####################

def evaluate_policy(model, environment, n_eval_episodes: int = 100):
    total_steps = 0
    total_reward = 0

    # Poner render a True para ver la evaluación
    environment.unwrapped.set_render(True)

    for n in range(n_eval_episodes):

        episode_steps = 0
        episode_reward = 0
        done = False

        obs, info = environment.reset()

        while not done:
            action, _states = model.predict(obs)
            obs, reward, done, truncated, info = environment.step(action)
            episode_reward += reward
            episode_steps += 1

        print(f"Epidose {n + 1} -->  Reward: {episode_reward} - Steps: {episode_steps}")
        total_reward += episode_reward
        total_steps += episode_steps

    print(f"\n  - Average reward per episode: {total_reward / n_eval_episodes}")
    print(f"  - Average steps per episode: {total_steps / n_eval_episodes}")

    # Volver a poner el render a False
    environment.unwrapped.set_render(False)

    cv2.destroyAllWindows()

################## TRAIN AGENT #######################

def train_agent(model):

    timesteps_per_batch = int(TOTAL_TIMESTEPS / NUM_EVALUATIONS)

    for i in range(0, NUM_EVALUATIONS):
        model.learn(total_timesteps=timesteps_per_batch)
        print(f"\nEVALUATION: {i + 1}/{NUM_EVALUATIONS}\t\tTRAINING STEPS: {'{0:,}'.format(timesteps_per_batch * i)}\n")
        evaluate_policy(model, env, n_eval_episodes=NUM_EPISODES_PER_EVAL)

    return model





################### TRAIN AGENT #######################

if sys.argv[1] == 'train':

    if sys.argv[2] == 'prey':

        env = PreyTrainingEnv(render=False)

        # New agents
        # model = DQN("MlpPolicy", env, verbose=0)
        # model = PPO("MlpPolicy", env, verbose=0)
        # model = A2C("MlpPolicy", env, verbose=0)

        # Trained agent
        prey_model = PPO.load("models/prey.model", env=env)

        trained_model = train_agent(prey_model)

        trained_model.save("models/new_prey.model")

    elif sys.argv[2] == 'hunter':

        env = HunterTrainingEnv(render=False)

        # New agents
        # model = DQN("MlpPolicy", env, verbose=0)
        # model = PPO("MlpPolicy", env, verbose=0)
        # model = A2C("MlpPolicy", env, verbose=0)

        # Trained agent
        hunter_model = PPO.load("models/hunter.model", env=env)

        trained_model = train_agent(hunter_model)

        trained_model.save("models/new_hunter.model")

    else:
        print(" Incorrect argument")


################### EVAL AGENT #######################

elif sys.argv[1] == 'eval':

    env = MazeWrapper(MazeEnv(render=True))

    # Load prey
    prey_agent = PPO.load("models/prey.model")

    # Load hunter
    hunter_agent = PPO.load("models/hunter.model")

    for i in range(NUM_EVALUATIONS):

        observations, _ = env.reset()
        n_steps = 0
        done = False
        info = {"winner": 'Nobody'}
        rewards = [0, 0]

        while not done:
            prey_action, _ = prey_agent.predict(observations[0])
            hunter_action, _ = hunter_agent.predict(observations[1])
            observations, rewards, done, truncated, info = env.step([prey_action, hunter_action])
            n_steps += 1

        if done:
            print(f'\nEpisode {i + 1}: {info["winner"]} wins')
            print(f' - Total steps: {n_steps}')
            print(f' - Prey Reward: {rewards[0]}')
            print(f' - Hunter Reward: {rewards[1]}')

else:
    print(" Incorrect argument")
