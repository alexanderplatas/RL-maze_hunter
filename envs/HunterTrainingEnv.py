import json
import random
from time import sleep

import cv2
import gymnasium
import numpy as np
from gym import spaces
from gymnasium import spaces

MAX_STEPS_LIMIT = 300

COLLISION_REWARD = -10
WIN_REWARD = 10

FPS = 15
MAP = 'original'

OBSTACLES_COLOR = (100, 100, 100)
GOAL_COLOR = (0, 150, 0)
PREY_COLOR = (255, 144, 30)
HUNTER_COLOR = (30, 144, 255)


class HunterTrainingEnv(gymnasium.Env):

    def __init__(self, render=False):

        """ Inicialización del entorno """

        super(HunterTrainingEnv, self).__init__()

        # Action and observation spaces
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(low=-594, high=594, shape=(11,), dtype=np.float64)

        # Load environment distribution
        with open('maps.json', 'r', encoding='utf8') as f:
            selected_map = json.load(f)[MAP]

        # Hunter state
        self.hunter = selected_map['hunter']

        # Prey state
        self.prey = selected_map['prey']

        # Obstacles
        self.obstacles = selected_map['obstacles']

        # Visualize game
        self.render = render
        self.letter_color = (255, 144, 30)

        # Game Over
        self.truncated = False
        self.done = False

        # Reward and number of steps and manhattan dist to goal
        self.init_manhattan = np.sum(np.abs(np.array(self.hunter) - np.array(self.prey)))
        self.reward, self.num_steps = 0, 0

        # Additional information
        self.info = {}

        # Generate board
        if self.render:
            self.img = np.zeros((620, 594, 3), dtype=np.uint8)
            self._generate_board()


    def step(self, action):
        """
        Take given action in environment
        :param action: action to take
            Up      --> 0
            Down    --> 1
            Right   --> 2
            Left    --> 3
        :return: new observation, reward, truncated, done and info
        """

        self.num_steps += 1

        ########## Move hunter ############
        
        if action == 0:  # UP
            self.hunter[1] -= 1
        if action == 1:  # DOWN
            self.hunter[1] += 1
        if action == 2:  # RIGHT
            self.hunter[0] += 1
        if action == 3:  # LEFT
            self.hunter[0] -= 1

        ####### Move goal randomly ########

        if np.sum(np.abs(np.array(self.hunter) - np.array(self.prey))) > 2:
            possible_actions = self._get_possible_actions(self.prey)
            if len(possible_actions) > 0:
                selected_action = random.choice(possible_actions)
                if selected_action == 0:  # UP
                    self.prey[1] -= 1
                if selected_action == 1:  # DOWN
                    self.prey[1] += 1
                if selected_action == 2:  # RIGHT
                    self.prey[0] += 1
                if selected_action == 3:  # LEFT
                    self.prey[0] -= 1

        ############# Rewards #############

        manhattan_dist_to_goal = np.sum(np.abs(np.array(self.hunter) - np.array(self.prey)))
        if self.num_steps < manhattan_dist_to_goal:
            self.reward = 1 / (manhattan_dist_to_goal + 0.00001)
        else:
            self.reward = -0.001

        ############# if died #############

        if self._is_dead(self.hunter) or self.num_steps >= MAX_STEPS_LIMIT:

            self.letter_color = (0, 0, 180)
            self.reward = -10
            self.truncated = True
            self.done = True

        ############ if wins ##############

        elif self.prey == self.hunter:
            self.done = True
            self.reward = 10
            self.letter_color = (0, 120, 0)

        ########## Visualization ##########

        if self.render:

            board = self._draw_step(self.img.copy())

            if self.done:
                cv2.putText(board, 'GAME OVER', (480, 612), cv2.FONT_HERSHEY_PLAIN,
                            1, self.letter_color, 1, cv2.LINE_AA)
                sleep_time = 1
            else:
                sleep_time = 1 / FPS
            cv2.imshow('Maze Hunter', board)

            cv2.waitKey(1)
            sleep(sleep_time)

        ############# Return ##############

        observation = self._get_observation()
        return observation, self.reward, self.done, self.truncated, self.info

    def reset(self, seed=None):

        """ Restaurar el entorno para empezar un nuevo episodio """

        # Load environment distribution
        with open('maps.json', 'r', encoding='utf8') as f:
            selected_map = json.load(f)[MAP]

        # Hunter state
        self.hunter = selected_map['hunter']

        # Prey state
        self.prey = selected_map['prey']

        # Obstacles
        self.obstacles = selected_map['obstacles']        

        # Game Over
        self.truncated = False
        self.done = False

        # Reward and number of steps and manhattan dist to goal
        self.init_manhattan = np.sum(np.abs(np.array(self.hunter) - np.array(self.prey)))
        self.reward, self.num_steps = 0, 0

        # Additional information
        self.info = {}

        # Generate board
        if self.render:
            self.letter_color = (255, 144, 30)
            self.img = np.zeros((620, 594, 3), dtype=np.uint8)
            self._generate_board()

        ############# Return ##############

        observation = self._get_observation()
        return observation, self.info

    def set_render(self, b):
        self.render = b

    def _is_dead(self, state):

        """ Dado un estado devuelve True si este es un estado terminal (muerte) """

        # returns true if current state is a terminal state
        return state in self.obstacles or -1 in state or 27 in state

    def _get_observation(self):

        """ Genera la observación del estado actual """

        # 12 observaciones, 9 casillas mas cercanas, dirección de la presa, y si la casilla actual ha sido visitada
        # hunter -> -2
        # obstacle -> -1
        # nothing -> 0
        # goal -> 1
        # prey -> 2

        ######## Agent observation ########

        pos_x = self.hunter[0]
        pos_y = self.hunter[1]

        # Direction to the prey
        final_x = self.prey[0] - pos_x > 0
        final_y = self.prey[1] - pos_y > 0

        alrededor = []
        for new_x in range(pos_x - 1, pos_x + 2):
            for new_y in range(pos_y - 1, pos_y + 2):

                if [new_x, new_y] in self.obstacles:
                    alrededor.append(-1)
                elif [new_x, new_y] == self.hunter:
                    alrededor.append(-2)
                elif [new_x, new_y] == self.prey:
                    alrededor.append(2)
                # elif [new_x,new_y] == self.goal:
                #     alrededor.append(1)
                else:
                    alrededor.append(0)

        # returns the observation of current state
        return np.array([final_x, final_y] + alrededor)

    def _get_possible_actions(self, state):

        possible_actions = list()
        if not self._is_dead([state[0], state[1] - 1]):
            possible_actions.append(0)
        if not self._is_dead([state[0], state[1] + 1]):
            possible_actions.append(1)
        if not self._is_dead([state[0] + 1, state[1]]):
            possible_actions.append(2)
        if not self._is_dead([state[0] - 1, state[1]]):
            possible_actions.append(3)

        return possible_actions

    def _generate_board(self):

        """ Genera el tablero para la visualización colocando obstáculos y destino """

        # Draw obstacles
        for obstacle in self.obstacles:
            cv2.rectangle(self.img, (obstacle[0] * 22 + 1, obstacle[1] * 22 + 1),
                          (obstacle[0] * 22 + 22 - 1, obstacle[1] * 22 + 22 - 1), OBSTACLES_COLOR, -1)
            
    def _draw_step(self, board):
        """
        Draw moving elements on board (agents, view, steps and reward)
        :param board: Board with fixed elements (frame)
        :return: Board with all elements drawn (frame)
        """
        overlay = board.copy()

        # Draw prey vision
        i = (self.prey[0] - 2) * 22
        j = (self.prey[1] - 2) * 22
        cv2.rectangle(overlay, (i + 1, j + 1), (i + 109, j + 109), (150, 80, 80), -1)
        board = cv2.addWeighted(overlay, 0.3, board, 1 - 0.3, 0)

        # Draw hunter vision
        i = (self.hunter[0] - 1) * 22
        j = (self.hunter[1] - 1) * 22
        cv2.rectangle(overlay, (i + 1, j + 1), (i + 65, j + 65), (80, 80, 150), -1)
        board = cv2.addWeighted(overlay, 0.3, board, 1 - 0.3, 0)

        # Draw prey
        cv2.circle(board, (self.prey[0] * 22 + 11, self.prey[1] * 22 + 11), 8, PREY_COLOR, -1)

        # Draw hunter
        cv2.circle(board, (self.hunter[0] * 22 + 11, self.hunter[1] * 22 + 11), 8, HUNTER_COLOR, -1)

        # Draw steps
        cv2.putText(board, f"Steps: {self.num_steps}", (5, 612),
                    cv2.FONT_HERSHEY_PLAIN, 1, self.letter_color, 1, cv2.LINE_AA)

        # Draw current reward
        cv2.putText(board, f"Reward: {round(self.reward, 4)}", (120, 612),
                    cv2.FONT_HERSHEY_PLAIN, 1, self.letter_color, 1, cv2.LINE_AA)

        return board

    def close(self):
        pass
