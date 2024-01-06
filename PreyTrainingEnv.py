import json
import random
from collections import deque
from time import sleep

import cv2
import gymnasium
import numpy as np
from gym import spaces
from gymnasium import spaces

MAX_STEPS_LIMIT = 1000
COLLISION_REWARD = -1
BE_EATEN_REWARD = -10
WIN_REWARD = 10
FPS = 30
OBSTACLES_COLOR = (100, 100, 100)
GOAL_COLOR = (0, 150, 0)
PREY_COLOR = (255, 144, 30)
HUNTER_COLOR = (30, 144, 255)
MAP = 'minirandom'


class TrainingEnv(gymnasium.Env):

    def __init__(self, render=False):

        """ Inicialización del entorno """

        super(TrainingEnv, self).__init__()

        # Action and observation spaces
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(low=-594, high=594, shape=(7, 7), dtype=np.float64)

        # Load environment distribution
        with open('maps.json', 'r', encoding='utf8') as f:
            selected_map = json.load(f)[MAP]

        # Obstacles
        self.obstacles = selected_map['obstacles']

        # Hunter
        self.hunter = [random.randint(1, 25), random.randint(1, 25)]
        while self.hunter in self.obstacles:
            self.hunter = [random.randint(1, 25), random.randint(1, 25)]

        manh_dist_hunter_prey = 0  # Minimal initial dist prey-hunter: 2

        # Initial state
        self.prey = self.hunter
        while self.prey in self.obstacles or manh_dist_hunter_prey < 2:
            self.prey = [random.randint(1, 25), random.randint(1, 25)]
            manh_dist_hunter_prey = np.sum(np.abs(np.array(self.prey) - np.array(self.hunter)))

        manh_dist_prey_goal = 0  # Minimal initial dist prey-goal: 2

        # Final state
        self.goal = self.prey
        while self.goal in self.obstacles or manh_dist_prey_goal < 2:
            self.goal = [random.randint(1, 25), random.randint(1, 25)]
            manh_dist_prey_goal = np.sum(np.abs(np.array(self.prey) - np.array(self.goal)))

        # Visualize game
        self.render = render
        self.letter_color = (255, 144, 30)

        # Game Over
        self.truncated = False
        self.done = False

        # State log
        self.state_log = list()

        # Reward and number of steps
        self.reward, self.num_steps = 0, 0

        # Initial distance to goal
        self.dist_to_goal = np.sum(np.abs(np.array(self.prey) - np.array(self.goal)))
        self.min_dist_reached = False

        # Additional information
        self.info = {"END": ""}

        if self.render:
            # Generate board
            self.img = np.zeros((620, 594, 3), dtype=np.uint8)
            self._generate_board()

    """ Realizar un paso sobre el entorno, action = [prey_action, hunter_action] """

    def step(self, action):

        self.num_steps += 1

        # ###### Move hunter randomly #######
        #
        # possible_actions = self._get_possible_actions(self.hunter)
        # selected_action = random.choice(possible_actions)
        # if selected_action == 0:  # UP
        #     self.hunter[1] -= 1
        # if selected_action == 1:  # DOWN
        #     self.hunter[1] += 1
        # if selected_action == 2:  # RIGHT
        #     self.hunter[0] += 1
        # if selected_action == 3:  # LEFT
        #     self.hunter[0] -= 1
        #
        # ####### if hunter eats prey #######
        #
        # if self.prey == self.hunter:
        #
        #     self.letter_color = HUNTER_COLOR
        #     self.reward = BE_EATEN_REWARD
        #     self.truncated = True
        #     self.done = True
        #     self.info['END'] = "Hunter ate prey"
        #
        # else:

        ########### Move prey ############

        if action == 0:  # UP
            self.prey[1] -= 1
        if action == 1:  # DOWN
            self.prey[1] += 1
        if action == 2:  # RIGHT
            self.prey[0] += 1
        if action == 3:  # LEFT
            self.prey[0] -= 1

        ############# if died #############

        # if self.prey == self.hunter:
        #
        #     self.letter_color = HUNTER_COLOR
        #     self.reward = BE_EATEN_REWARD
        #     self.truncated = True
        #     self.done = True
        #     self.info['END'] = "Hunter ate prey"

        if self._is_dead(self.prey):

            self.letter_color = (0, 0, 180)
            self.reward = COLLISION_REWARD
            self.truncated = True
            self.done = True
            self.info['END'] = "Prey collided"

        elif self.num_steps >= MAX_STEPS_LIMIT:

            self.letter_color = (0, 0, 180)
            self.reward = COLLISION_REWARD
            self.truncated = True
            self.done = True
            self.info['END'] = "Step limit excedeed"

        ############ if wins ##############

        elif self.goal == self.prey:
            self.done = True
            self.reward = WIN_REWARD
            self.letter_color = (0, 120, 0)
            self.info['END'] = "Prey arrived at goal"

        ############# Rewards #############

        else:
            # print(f"{self.prey} -- {self.state_log}")

            self.reward = 0
            if self.prey in self.state_log:
                self.reward = -0.01
            self.state_log.append(self.prey.copy())

            # if not self.min_dist_reached:
            #
            #     new_dist_to_goal = np.sum(np.abs(np.array(self.prey) - np.array(self.goal)))
            #
            #     if new_dist_to_goal < self.dist_to_goal:
            #         self.reward = 1 / new_dist_to_goal
            #
            #     self.min_dist_reached = new_dist_to_goal == 1

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
            # cv2.imwrite(f"frame_{self.num_steps}.png", board)
            cv2.waitKey(1)
            sleep(sleep_time)

        ############# Return ##############

        observation = self._get_observation()
        return observation, self.reward, self.done, self.truncated, self.info

    def reset(self, seed=None):

        """ Restaurar el entorno para empezar un nuevo episodio """

        # Reload environment distribution
        with open('maps.json', 'r', encoding='utf8') as f:
            selected_map = json.load(f)[MAP]

        # Obstacles
        self.obstacles = selected_map['obstacles']

        # Hunter
        self.hunter = [random.randint(1, 25), random.randint(1, 25)]
        while self.hunter in self.obstacles:
            self.hunter = [random.randint(1, 25), random.randint(1, 25)]

        manh_dist_hunter_prey = 0  # Minimal initial dist prey-hunter: 2

        # Initial state
        self.prey = self.hunter
        while self.prey in self.obstacles or manh_dist_hunter_prey < 2:
            self.prey = [random.randint(1, 25), random.randint(1, 25)]
            manh_dist_hunter_prey = np.sum(np.abs(np.array(self.prey) - np.array(self.hunter)))

        manh_dist_prey_goal = 0  # Minimal initial dist prey-goal: 2

        # Final state
        self.goal = self.prey
        while self.goal in self.obstacles or manh_dist_prey_goal < 2:
            self.goal = [random.randint(1, 25), random.randint(1, 25)]
            manh_dist_prey_goal = np.sum(np.abs(np.array(self.prey) - np.array(self.goal)))

        # Game Over
        self.truncated = False
        self.done = False

        # State log
        self.state_log = list()

        # Reward and number of steps
        self.reward, self.num_steps = 0, 0

        # Initial distance to goal
        self.dist_to_goal = np.sum(np.abs(np.array(self.prey) - np.array(self.goal)))
        self.min_dist_reached = False

        # Additional information
        self.info = {"END": ""}

        if self.render:

            # Generate board
            self.letter_color = (255, 144, 30)
            self.img = np.zeros((620, 594, 3), dtype=np.uint8)
            self._generate_board()

        ############# Return ##############

        observation = self._get_observation()
        return observation, self.info

    def set_render(self, b):
        """
        Render setter
        :param b: boolean
        """
        self.render = b

    def _is_dead(self, state):
        """
        Check if given state is terminal
        :param state: current state (2 item list)
        :return: True if terminal, False otherwise
        """
        # returns true if current state is a terminal state
        return state in self.obstacles or -1 in state or 27 in state

    def _get_observation(self):
        """
        Generate current prey observation
        :return: 7x7 matrix
            Prey      --> 2
            Goal      --> 1
            Nothing   --> 0
            Obstacle  --> -1
            Hunter    --> -2
        """

        observation = list()
        for i in range(self.prey[0] - 3, self.prey[0] + 3 + 1):
            row = list()
            for j in range(self.prey[1] - 3, self.prey[1] + 3 + 1):

                if [i, j] == self.prey:
                    row.append(2)
                # elif [i, j] == self.hunter:
                #     row.append(-2)
                elif [i, j] == self.goal:
                    row.append(1)
                elif [i, j] in self.obstacles or i < 0 or i > 27 or j < 0 or j > 27:
                    row.append(-1)
                else:
                    row.append(0)
            observation.append(row)
        return np.array(observation)

    def _get_possible_actions(self, state):
        """
        Get possible actions from given state
        :param state: current state (2 item list)
        :return: list of possible actions
            Up      --> 0
            Down    --> 1
            Right   --> 2
            Left    --> 3
        """
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
        """
        Draw fixed elements on board (obstacles and goal)
        """
        # Draw obstacles
        for obstacle in self.obstacles:
            cv2.rectangle(self.img, (obstacle[0] * 22 + 1, obstacle[1] * 22 + 1),
                          (obstacle[0] * 22 + 22 - 1, obstacle[1] * 22 + 22 - 1), OBSTACLES_COLOR, -1)

        # Draw goal
        cv2.rectangle(self.img, (self.goal[0] * 22 + 1, self.goal[1] * 22 + 1),
                      (self.goal[0] * 22 + 22 - 1, self.goal[1] * 22 + 22 - 1), GOAL_COLOR, -1)

    def _draw_step(self, board):
        """
        Draw moving elements on board (agents, view, steps and reward)
        :param board: Board with fixed elements (frame)
        :return: Board with all elements drawn (frame)
        """
        overlay = board.copy()

        # Draw prey vision
        i = (self.prey[0] - 3) * 22
        j = (self.prey[1] - 3) * 22
        cv2.rectangle(overlay, (i + 1, j + 1), (i + 153, j + 153), (150, 80, 80), -1)
        board = cv2.addWeighted(overlay, 0.3, board, 1 - 0.3, 0)

        # # Draw hunter vision
        # i = (self.hunter[0] - 1) * 22
        # j = (self.hunter[1] - 1) * 22
        # cv2.rectangle(overlay, (i + 1, j + 1), (i + 65, j + 65), (80, 80, 150), -1)
        # board = cv2.addWeighted(overlay, 0.3, board, 1 - 0.3, 0)

        # Draw prey
        cv2.circle(board, (self.prey[0] * 22 + 11, self.prey[1] * 22 + 11), 8, PREY_COLOR, -1)

        # Draw hunter
        # cv2.circle(board, (self.hunter[0] * 22 + 11, self.hunter[1] * 22 + 11), 8, HUNTER_COLOR, -1)

        # Draw steps
        cv2.putText(board, f"Steps: {self.num_steps}", (5, 612),
                    cv2.FONT_HERSHEY_PLAIN, 1, self.letter_color, 1, cv2.LINE_AA)

        # Draw current reward
        cv2.putText(board, f"Reward: {round(self.reward, 4)}", (120, 612),
                    cv2.FONT_HERSHEY_PLAIN, 1, self.letter_color, 1, cv2.LINE_AA)

        return board

    def close(self):
        pass
