import json
from time import sleep

import cv2
import gymnasium
import numpy as np
from gym import spaces
from gymnasium import spaces

MAX_STEPS_LIMIT = 100
COLLISION_REWARD = -10
WIN_REWARD = 10
FPS = 15
PREY_COLOR = (255, 144, 30)
HUNTER_COLOR = (30, 144, 255)
OBSTACLES_COLOR = (100, 100, 100)
GOAL_COLOR = (0, 150, 0)
MAP = 'borders'


class MazeEnv(gymnasium.Env):

    def __init__(self, render=False):

        """ Inicialización del entorno """

        super(MazeEnv, self).__init__()

        # Action and observation spaces
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(low=-594, high=594, shape=(12,), dtype=np.float64)

        # Load environment distribution
        with open('maps.json', 'r', encoding='utf8') as f:
            map = json.load(f)[MAP]

        # Initial state
        self.prey_state = map['prey']
        self.hunter_state = map['hunter']

        # Final state
        self.goal = map['goal']

        # Obstacles
        self.obstacles = map['obstacles']

        # Visualize game
        self.render = render
        self.letter_color = (255, 144, 30)

        # Game Over
        self.truncated = False
        self.done = False

        # Reward and number of steps
        self.reward, self.num_steps = 0, 0

        # Additional information
        self.info = {
            "winner": 'Nobody',
            "loser": 'Nobody'
        }

        if self.render:
            # Generate board
            self.img = np.zeros((620, 594, 3), dtype=np.uint8)
            self._generate_board()

    """ Realizar un paso sobre el entorno, action = [prey_action, hunter_action] """

    def step(self, action):

        self.num_steps += 1

        ########## Move hunter ############
        if action[1] == 0:  # UP
            self.hunter_state[1] -= 1
        if action[1] == 1:  # DOWN
            self.hunter_state[1] += 1
        if action[1] == 2:  # RIGHT
            self.hunter_state[0] += 1
        if action[1] == 3:  # LEFT
            self.hunter_state[0] -= 1

        ########### Move prey #############
        if action[0] == 0:  # UP
            self.prey_state[1] -= 1
        if action[0] == 1:  # DOWN
            self.prey_state[1] += 1
        if action[0] == 2:  # RIGHT
            self.prey_state[0] += 1
        if action[0] == 3:  # LEFT
            self.prey_state[0] -= 1

        self.num_steps += 1

        ######### if hunter dies ##########

        if self._is_dead(self.hunter_state) or self.num_steps >= MAX_STEPS_LIMIT:

            self.info['loser'] = 'Hunter'
            self.letter_color = (0, 0, 255)
            self.truncated = True
            self.done = True

        ########## if prey dies ###########

        elif self._is_dead(self.prey_state):

            self.info['loser'] = 'Prey'
            self.letter_color = (0, 0, 255)
            self.truncated = True
            self.done = True

        ######### if hunter wins ##########

        elif self.hunter_state == self.prey_state:
            self.done = True
            self.info['winner'] = 'Hunter'
            self.letter_color = HUNTER_COLOR

        ########## if prey wins ###########

        elif self.goal == self.prey_state:
            self.done = True
            self.info['winner'] = 'Prey'
            self.letter_color = PREY_COLOR

        ############# Rewards #############

        else:
            self.reward = 0

        ########## Visualization ##########

        if self.render:

            board = self.img.copy()
            cv2.circle(board, (self.prey_state[0] * 22 + 11, self.prey_state[1] * 22 + 11),
                       8, PREY_COLOR, -1)
            cv2.circle(board, (self.hunter_state[0] * 22 + 11, self.hunter_state[1] * 22 + 11),
                       8, HUNTER_COLOR, -1)
            cv2.putText(board, f"Steps: {self.num_steps}", (5, 612),
                        cv2.FONT_HERSHEY_PLAIN, 1, self.letter_color, 1, cv2.LINE_AA)

            if self.done:
                cv2.putText(board, f'{self.info["winner"]} wins', (480, 612), cv2.FONT_HERSHEY_PLAIN,
                            1, self.letter_color, 2, cv2.LINE_AA)
                sleep_time = 1
            else:
                sleep_time = 1 / FPS
            cv2.imshow('MAZE HUNTER', board)
            cv2.waitKey(1)
            sleep(sleep_time)

        ############# Return ##############

        observation = self._get_obs()

        return observation, self.reward, self.done, self.truncated, self.info

    def reset(self, seed=None):

        """ Restaurar el entorno para empezar un nuevo episodio """

        # Reload environment distribution
        with open('maps.json', 'r', encoding='utf8') as f:
            map = json.load(f)[MAP]

        # Initial state
        self.prey_state = map['prey']
        self.hunter_state = map['hunter']

        # Final state
        self.goal = map['goal']

        # Obstacles
        self.obstacles = map['obstacles']

        # Game Over
        self.truncated = False
        self.done = False

        # Reward and number of steps
        self.reward, self.num_steps = 0, 0

        # Additional information
        self.info = {
            "winner": 'Nobody',
            "loser": 'Nobody'
        }

        if self.render:
            # Generate board
            self.letter_color = (255, 144, 30)
            self.img = np.zeros((620, 594, 3), dtype=np.uint8)
            self._generate_board()

        ############# Return ##############

        observation = self._get_obs()
        return observation, self.info

    def _is_dead(self, state):

        """ Dado un estado devuelve True si este es un estado terminal (muerte) """

        # returns true if current state is a terminal state
        return state in self.obstacles or -1 in state or 27 in state

    def _get_obs(self):

        """ Genera la observación del estado actual """

        ######## Prey observation #########

        prey_pos_x = self.prey_state[0]
        prey_pos_y = self.prey_state[1]

        # Direction to the goal
        prey_final_x = self.goal[0] - prey_pos_x > 0
        prey_final_y = self.goal[1] - prey_pos_y > 0

        # If there are elements around
        prey_up = self._is_dead([prey_pos_x, prey_pos_y - 1]) or [prey_pos_x, prey_pos_y - 1] == self.hunter_state
        prey_down = self._is_dead([prey_pos_x, prey_pos_y + 1]) or [prey_pos_x, prey_pos_y + 1] == self.hunter_state
        prey_right = self._is_dead([prey_pos_x + 1, prey_pos_y]) or [prey_pos_x + 1, prey_pos_y] == self.hunter_state
        prey_left = self._is_dead([prey_pos_x - 1, prey_pos_y]) or [prey_pos_x - 1, prey_pos_y] == self.hunter_state

        ####### Hunter observation ########

        hunter_pos_x = self.prey_state[0]
        hunter_pos_y = self.prey_state[1]

        # Direction to the prey
        hunter_final_x = self.prey_state[0] - hunter_pos_x > 0
        hunter_final_y = self.prey_state[1] - hunter_pos_y > 0

        # If there are elements around
        hunter_up = self._is_dead([hunter_pos_x, hunter_pos_y - 1])
        hunter_down = self._is_dead([hunter_pos_x, hunter_pos_y + 1])
        hunter_right = self._is_dead([hunter_pos_x + 1, hunter_pos_y])
        hunter_left = self._is_dead([hunter_pos_x - 1, hunter_pos_y])

        # returns the observation of current state
        return np.array([
            prey_final_x, prey_final_y, prey_up, prey_down, prey_right, prey_left,
            hunter_final_x, hunter_final_y, hunter_up, hunter_down, hunter_right, hunter_left
        ])

    def _generate_board(self):

        """ Genera el tablero para la visualización colocando obstáculos y destino """

        # Draw obstacles (red)
        for obstacle in self.obstacles:
            cv2.rectangle(self.img, (obstacle[0] * 22 + 1, obstacle[1] * 22 + 1),
                          (obstacle[0] * 22 + 22 - 1, obstacle[1] * 22 + 22 - 1), OBSTACLES_COLOR, -1)

        # Draw goal (green)
        cv2.rectangle(self.img, (self.goal[0] * 22 + 1, self.goal[1] * 22 + 1),
                      (self.goal[0] * 22 + 22 - 1, self.goal[1] * 22 + 22 - 1), GOAL_COLOR, -1)

    def close(self):
        pass
