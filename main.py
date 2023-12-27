import gym
from stable_baselines3.common.env_checker import check_env

from MazeEnv import MazeEnv
from TrainingEnv import TrainingEnv
from gymnasium.wrappers import TransformObservation
from stable_baselines3 import PPO, DQN, A2C
from common import evaluate_policy


class MazeWrapper(gym.Wrapper):

    def __init__(self, maze_env):
        super().__init__(maze_env)
        self.env = maze_env

    def step(self, actions):

        observation, reward, done, truncated, info = self.env.step(actions)

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


###################### 1 agent ########################

env = TrainingEnv(render=False)

# model = DQN("MlpPolicy", env, verbose=0)
# model = PPO("MlpPolicy", env, verbose=0)
# model = A2C("MlpPolicy", env, verbose=0)

model = PPO.load("models/ppo_prey_prueba1.model", env=env)

TOTAL_TIMESTEPS = 100_000
NUM_EVALUATIONS = 2
NUM_EPISODES_PER_EVAL = 5

timesteps_per_batch = int(TOTAL_TIMESTEPS / NUM_EVALUATIONS)

for i in range(0, NUM_EVALUATIONS):
    model.learn(total_timesteps=timesteps_per_batch)
    print(f"\nEVALUATION: {i + 1}/{NUM_EVALUATIONS}\t\tTRAINING STEPS: {'{0:,}'.format(timesteps_per_batch * i)}\n")
    evaluate_policy(model, env, n_eval_episodes=NUM_EPISODES_PER_EVAL)

model.save("models/ppo_prey_prueba1.model")

###################### 2 agents #######################

# env = MazeWrapper(MazeEnv(render=True))
#
# NUM_EPISODES = 10
#
# # Load prey
# prey_agent = PPO.load("models/ppo_prey_prueba1.model")
#
# # Load hunter
# hunter_agent = PPO.load("models/ppo_prey_prueba1.model")
#
# for i in range(NUM_EPISODES):
#
#     observations, _ = env.reset()
#     n_steps = 0
#     done = False
#     info = {"winner": 'Nobody'}
#     rewards = [0, 0]
#
#     while not done:
#
#         prey_action, _ = prey_agent.predict(observations[0])
#         hunter_action, _ = hunter_agent.predict(observations[1])
#         observations, rewards, done, truncated, info = env.step([prey_action, hunter_action])
#         n_steps += 1
#
#     if done:
#         print(f'Episode {i+1}: {info["winner"]} wins')
#         print(f' - Total steps: {n_steps}')
#         print(f' - Prey Reward: {rewards[0]}')
#         print(f' - Hunter Reward: {rewards[1]}')
