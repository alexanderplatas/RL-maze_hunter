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
OBSTACLES_COLOR = (100, 100, 100)
GOAL_COLOR = (0, 150, 0)
AGENT_COLOR = (0, 150, 150)
MAP = 'original'


class HunterTrainingEnv(gymnasium.Env):

    def __init__(self, render=False):

        """ Inicialización del entorno """

        super(HunterTrainingEnv, self).__init__()

        # Action and observation spaces
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(low=-594, high=594, shape=(11,), dtype=np.float64)

        # Load environment distribution
        with open('maps.json', 'r', encoding='utf8') as f:
            map = json.load(f)[MAP]

        # Initial state
        # self.agent_state = map['hunter']
        self.agent_state = [random.randint(1,25),random.randint(1,25)]
        
        # Final state
        # self.goal = map['goal']

        # self.goal = self.agent_state
        # while self.goal == self.agent_state:

        # Obstacles
        self.obstacles = map['obstacles']

        ########### Set up goal ############

        # Miniborders
        # self.goal = random.choice([[9,9],[9,17],[17,9],[17,17]])

        # Bordes
        # self.goal = random.choice([[1,1],[1,25],[25,1],[25,25]])
        # self.goal = random.choice([[1,random.randint(1,25)],[random.randint(1,25),1],[25,random.randint(1,25)],[random.randint(1,25),25]])

        # Random 
        self.goal = [random.randint(1,25),random.randint(1,25)]
        while self.goal not in self.obstacles and np.sum(np.abs(np.array(self.agent_state) - np.array(self.goal))) < 10:
            self.goal = [random.randint(1,25),random.randint(1,25)]

        # Onewall
        # self.goal = [13, 20]

        # Original        
        # self.goal = random.choice([[1, 1], [1, 25], [25, 1], [25, 25], [13, 23]])

        ####################################

        # Visualize game
        self.render = render
        self.letter_color = (255, 144, 30)

        # Game Over
        self.truncated = False
        self.done = False

        # Reward and number of steps and manhattan dist to goal
        self.init_manhattan = np.sum(np.abs(np.array(self.agent_state) - np.array(self.goal)))
        self.reward, self.num_steps = 0, 0

        # Additional information
        self.info = {}

        if self.render:
            # Generate board
            self.img = np.zeros((620, 594, 3), dtype=np.uint8)
            self._generate_board()

    """ Realizar un paso sobre el entorno, action = [prey_action, hunter_action] """

    def step(self, action):

        self.num_steps += 1

        ########### Move agent ############
        if action == 0:  # UP
            self.agent_state[1] -= 1
        if action == 1:  # DOWN
            self.agent_state[1] += 1
        if action == 2:  # RIGHT
            self.agent_state[0] += 1
        if action == 3:  # LEFT
            self.agent_state[0] -= 1

        ####### Move goal randomly ########

        if np.sum(np.abs(np.array(self.agent_state) - np.array(self.goal))) > 2:
            possible_actions = self._get_possible_actions(self.goal)
            if len(possible_actions) > 0:
                selected_action = random.choice(possible_actions)
                if selected_action == 0:  # UP
                    self.goal[1] -= 1
                if selected_action == 1:  # DOWN
                    self.goal[1] += 1
                if selected_action == 2:  # RIGHT
                    self.goal[0] += 1
                if selected_action == 3:  # LEFT
                    self.goal[0] -= 1


        ############# Rewards #############
        
        manhattan_dist_to_goal = np.sum(np.abs(np.array(self.agent_state) - np.array(self.goal)))
        if self.num_steps < manhattan_dist_to_goal:
            self.reward = 1 / (manhattan_dist_to_goal + 0.00001)
        else:
            self.reward = -0.001
            

        ############# if died #############

        if self._is_dead(self.agent_state) or self.num_steps >= MAX_STEPS_LIMIT:

            self.letter_color = (0, 0, 180)
            self.reward = -10
            self.truncated = True
            self.done = True

        ############ if wins ##############

        elif self.goal == self.agent_state:
            self.done = True
            self.reward = 10
            self.letter_color = (0, 120, 0)

        ########## Visualization ##########

        if self.render:

            board = self.img.copy()
            i = (self.agent_state[0] - 1) * 22
            j = (self.agent_state[1] - 1) * 22
            overlay = board.copy()
            cv2.rectangle(overlay, (i + 1, j + 1), (i + 65, j + 65), (80, 80, 150), -1)
            board = cv2.addWeighted(overlay, 0.5, board, 1 - 0.5, 0)
            cv2.rectangle(board, (self.goal[0] * 22 + 1, self.goal[1] * 22 + 1),
                          (self.goal[0] * 22 + 22 - 1, self.goal[1] * 22 + 22 - 1), GOAL_COLOR, -1)
            cv2.circle(board, (self.agent_state[0] * 22 + 11, self.agent_state[1] * 22 + 11), 8, AGENT_COLOR, -1)
            cv2.putText(board, f"Steps: {self.num_steps}", (5, 612),
                        cv2.FONT_HERSHEY_PLAIN, 1, self.letter_color, 1, cv2.LINE_AA)
            if self.done:
                cv2.putText(board, 'GAME OVER', (480, 612), cv2.FONT_HERSHEY_PLAIN,
                            1, self.letter_color, 1, cv2.LINE_AA)
                sleep_time = 1
            else:

                sleep_time = 1 / FPS
            cv2.imshow('DUMB MAZE RUNNER', board)
            cv2.waitKey(1)
            sleep(sleep_time)

        ############# Return ##############

        observation = self._get_observation()
        return observation, self.reward, self.done, self.truncated, self.info

    def reset(self, seed=None):

        """ Restaurar el entorno para empezar un nuevo episodio """

        # Reload environment distribution
        with open('maps.json', 'r', encoding='utf8') as f:
            map = json.load(f)[MAP]

        # Initial state
        # self.agent_state = map['hunter']
        self.agent_state = [random.randint(1,25),random.randint(1,25)]

        # Final state
        # self.goal = map['goal']

        ########### Set up goal ############

        # Miniborders
        # self.goal = random.choice([[9,9],[9,17],[17,9],[17,17]])

        # Bordes
        # self.goal = random.choice([[1,1],[1,25],[25,1],[25,25]])
        # self.goal = random.choice([[1,random.randint(1,25)],[random.randint(1,25),1],[25,random.randint(1,25)],[random.randint(1,25),25]])

        # Random 
        self.goal = [random.randint(1,25),random.randint(1,25)]
        while self.goal not in map['obstacles'] and np.sum(np.abs(np.array(self.agent_state) - np.array(self.goal))) < 10:
            self.goal = [random.randint(1,25),random.randint(1,25)]

        # Onewall
        # self.goal = [13, 20]

        # Original        
        # self.goal = random.choice([[1, 1], [1, 25], [25, 1], [25, 25], [13, 23]])

        ####################################

        # Obstacles
        self.obstacles = map['obstacles']

        # Game Over
        self.truncated = False
        self.done = False

        # Reward and number of steps and manhattan dist to goal
        self.init_manhattan = np.sum(np.abs(np.array(self.agent_state) - np.array(self.goal)))
        self.reward, self.num_steps = 0, 0


        # Additional information
        self.info = {}

        if self.render:
            # Generate board
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

        pos_x = self.agent_state[0]
        pos_y = self.agent_state[1]

        # Direction to the prey
        final_x = self.goal[0] - pos_x > 0
        final_y = self.goal[1] - pos_y > 0

        alrededor = [] 
        for new_x in range (pos_x - 1,pos_x + 2):
            for new_y in range(pos_y-1, pos_y + 2):

                if [new_x,new_y] in self.obstacles:
                    alrededor.append(-1)
                elif [new_x,new_y] == self.agent_state:
                    alrededor.append(-2)
                elif [new_x,new_y] == self.goal:
                    alrededor.append(2)
                # elif [new_x,new_y] == self.goal:
                #     alrededor.append(2)
                else:
                    alrededor.append(0)


        # returns the observation of current state
        return np.array([final_x, final_y] + alrededor )

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

    def close(self):
        pass
