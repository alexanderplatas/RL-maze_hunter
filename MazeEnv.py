import json
from time import sleep

import cv2
import gymnasium
import numpy as np
from gym import spaces
from gymnasium import spaces

MAX_STEPS_LIMIT = 500
LOSE_REWARD = -10
WIN_REWARD = 10
FPS = 10
PREY_COLOR = (255, 144, 30)
HUNTER_COLOR = (30, 144, 255)
OBSTACLES_COLOR = (100, 100, 100)
GOAL_COLOR = (0, 150, 0)
MAP = 'miniborders'


class MazeEnv(gymnasium.Env):

    def __init__(self, render=False):

        """ Inicialización del entorno """

        super(MazeEnv, self).__init__()

        # Action and observation spaces
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(low=-594, high=594, shape=(60,), dtype=np.float64)

        # Load environment distribution
        with open('maps.json', 'r', encoding='utf8') as f:
            map = json.load(f)[MAP]

        # Initial state
        self.prey = map['prey']
        self.hunter = map['hunter']

        # Final state
        self.goal = map['goal']

        # Obstacles
        self.obstacles = map['obstacles']

        # Visualize game
        self.render = render
        self.letter_color = (0, 120, 0)

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
            self.hunter[1] -= 1
        if action[1] == 1:  # DOWN
            self.hunter[1] += 1
        if action[1] == 2:  # RIGHT
            self.hunter[0] += 1
        if action[1] == 3:  # LEFT
            self.hunter[0] -= 1

        ####### if hunter eats prey #######

        if self.hunter == self.prey:
            self.done = True
            self.info['winner'] = 'Hunter'
            self.letter_color = HUNTER_COLOR

        else:

            ########### Move prey #############
            if action[0] == 0:  # UP
                self.prey[1] -= 1
            if action[0] == 1:  # DOWN
                self.prey[1] += 1
            if action[0] == 2:  # RIGHT
                self.prey[0] += 1
            if action[0] == 3:  # LEFT
                self.prey[0] -= 1

            self.num_steps += 1

            ######### if hunter dies ##########

            if self._is_dead(self.hunter):

                self.info['loser'] = 'Hunter'
                self.letter_color = (0, 0, 200)
                self.truncated = True
                self.done = True

            ########## if prey dies ###########

            elif self._is_dead(self.prey):

                self.info['loser'] = 'Prey'
                self.letter_color = (0, 0, 200)
                self.truncated = True
                self.done = True

            ######### if hunter wins ##########

            elif self.hunter == self.prey:
                self.done = True
                self.info['winner'] = 'Hunter'
                self.letter_color = HUNTER_COLOR

            ########## if prey wins ###########

            elif self.goal == self.prey:
                self.done = True
                self.info['winner'] = 'Prey'
                self.letter_color = PREY_COLOR

            ############# Rewards #############

            else:
                self.reward = 0

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

        # Reload environment distribution
        with open('maps.json', 'r', encoding='utf8') as f:
            map = json.load(f)[MAP]

        # Initial state
        self.prey = map['prey']
        self.hunter = map['hunter']

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
            self.letter_color = (0, 120, 0)
            self.img = np.zeros((620, 594, 3), dtype=np.uint8)
            self._generate_board()

        ############# Return ##############

        observation = self._get_observation()
        return observation, self.info

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
        Generate current prey and hunter observation
        :return: list of prey and hunter obsrtvation
            Prey      --> 2
            Goal      --> 1
            Nothing   --> 0
            Obstacle  --> -1
            Hunter    --> -2
        """
        ######## Prey observation #########

        prey_observation = list()
        for i in range(self.prey[0] - 3, self.prey[0] + 3 + 1):
            for j in range(self.prey[1] - 3, self.prey[1] + 3 + 1):

                if [i, j] == self.prey:
                    prey_observation.append(2)
                # elif [i, j] == self.hunter:
                #     prey_observation.append(-2)
                elif [i, j] == self.goal:
                    prey_observation.append(1)
                elif [i, j] in self.obstacles:
                    prey_observation.append(-1)
                else:
                    prey_observation.append(0)


        ####### Hunter observation ########

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
                else:
                    alrededor.append(0)
        hunter_observation = [final_x, final_y] + alrededor

        # returns the observation of current state
        return np.array(prey_observation + hunter_observation)

    def _generate_board(self):
        """
        Draw fixed elements on board (obstacles and goal)
        """
        # Draw obstacles (red)
        for obstacle in self.obstacles:
            cv2.rectangle(self.img, (obstacle[0] * 22 + 1, obstacle[1] * 22 + 1),
                          (obstacle[0] * 22 + 22 - 1, obstacle[1] * 22 + 22 - 1), OBSTACLES_COLOR, -1)

        # Draw goal (green)
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
