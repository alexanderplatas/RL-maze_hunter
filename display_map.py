import cv2
import numpy as np

""" PRUEBAS PARA CREAR NUEVOS MAPAS """

############ PARAMS ############

obstacles = [[0, 0], [0, 1], [0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 7], [0, 8], [0, 9], [0, 10], [0, 11], [0, 12], [0, 13], [0, 14], [0, 15], [0, 16], [0, 17], [0, 18], [0, 19], [0, 20], [0, 21], [0, 22], [0, 23], [0, 24], [0, 25], [0, 26], [1, 0], [2, 0], [3, 0], [4, 0], [5, 0], [6, 0], [7, 0], [8, 0], [9, 0], [10, 0], [11, 0], [12, 0], [13, 0], [14, 0], [15, 0], [16, 0], [17, 0], [18, 0], [19, 0], [20, 0], [21, 0], [22, 0], [23, 0], [24, 0], [25, 0], [26, 0], [26, 0], [26, 1], [26, 2], [26, 3], [26, 4], [26, 5], [26, 6], [26, 7], [26, 8], [26, 9], [26, 10], [26, 11], [26, 12], [26, 13], [26, 14], [26, 15], [26, 16], [26, 17], [26, 18], [26, 19], [26, 20], [26, 21], [26, 22], [26, 23], [26, 24], [26, 25], [26, 26], [0, 26], [1, 26], [2, 26], [3, 26], [4, 26], [5, 26], [6, 26], [7, 26], [8, 26], [9, 26], [10, 26], [11, 26], [12, 26], [13, 26], [14, 26], [15, 26], [16, 26], [17, 26], [18, 26], [19, 26], [20, 26], [21, 26], [22, 26], [23, 26], [24, 26], [25, 26],
             [2, 2], [3, 2], [4, 2], [5, 2], [6, 2], [8, 2], [9, 2], [10, 2], [11, 2], [12, 2], [14, 2], [15, 2], [16, 2], [17, 2], [18, 2], [20, 2], [21, 2], [22, 2], [23, 2], [24, 2], [2, 3], [3, 3], [4, 3], [5, 3], [6, 3], [8, 3], [9, 3], [10, 3], [11, 3], [12, 3], [14, 3], [15, 3], [16, 3], [17, 3], [18, 3], [20, 3], [21, 3], [22, 3], [23, 3], [24, 3], [2, 4], [3, 4], [4, 4], [5, 4], [6, 4], [8, 4], [9, 4], [10, 4], [11, 4], [12, 4], [14, 4], [15, 4], [16, 4], [17, 4], [18, 4], [20, 4], [21, 4], [22, 4], [23, 4], [24, 4], [2, 5], [3, 5], [4, 5], [5, 5], [6, 5], [8, 5], [9, 5], [10, 5], [11, 5], [12, 5], [14, 5], [15, 5], [16, 5], [17, 5], [18, 5], [20, 5], [21, 5], [22, 5], [23, 5], [24, 5], [2, 6], [3, 6], [4, 6], [5, 6], [6, 6], [8, 6], [9, 6], [10, 6], [11, 6], [12, 6], [14, 6], [15, 6], [16, 6], [17, 6], [18, 6], [20, 6], [21, 6], [22, 6], [23, 6], [24, 6], [2, 8], [3, 8], [4, 8], [5, 8], [6, 8], [8, 8], [9, 8], [10, 8], [11, 8], [12, 8], [14, 8], [15, 8], [16, 8], [17, 8], [18, 8], [20, 8], [21, 8], [22, 8], [23, 8], [24, 8], [2, 9], [3, 9], [4, 9], [5, 9], [6, 9], [8, 9], [9, 9], [10, 9], [11, 9], [12, 9], [14, 9], [15, 9], [16, 9], [17, 9], [18, 9], [20, 9], [21, 9], [22, 9], [23, 9], [24, 9], [2, 10], [3, 10], [4, 10], [5, 10], [6, 10], [8, 10], [9, 10], [10, 10], [11, 10], [12, 10], [14, 10], [15, 10], [16, 10], [17, 10], [18, 10], [20, 10], [21, 10], [22, 10], [23, 10], [24, 10], [2, 11], [3, 11], [4, 11], [5, 11], [6, 11], [8, 11], [9, 11], [10, 11], [11, 11], [12, 11], [14, 11], [15, 11], [16, 11], [17, 11], [18, 11], [20, 11], [21, 11], [22, 11], [23, 11], [24, 11], [2, 12], [3, 12], [4, 12], [5, 12], [6, 12], [8, 12], [9, 12], [10, 12], [11, 12], [12, 12], [14, 12], [15, 12], [16, 12], [17, 12], [18, 12], [20, 12], [21, 12], [22, 12], [23, 12], [24, 12], [2, 14], [3, 14], [4, 14], [5, 14], [6, 14], [8, 14], [9, 14], [10, 14], [11, 14], [12, 14], [14, 14], [15, 14], [16, 14], [17, 14], [18, 14], [20, 14], [21, 14], [22, 14], [23, 14], [24, 14], [2, 15], [3, 15], [4, 15], [5, 15], [6, 15], [8, 15], [9, 15], [10, 15], [11, 15], [12, 15], [14, 15], [15, 15], [16, 15], [17, 15], [18, 15], [20, 15], [21, 15], [22, 15], [23, 15], [24, 15], [2, 16], [3, 16], [4, 16], [5, 16], [6, 16], [8, 16], [9, 16], [10, 16], [11, 16], [12, 16], [14, 16], [15, 16], [16, 16], [17, 16], [18, 16], [20, 16], [21, 16], [22, 16], [23, 16], [24, 16], [2, 17], [3, 17], [4, 17], [5, 17], [6, 17], [8, 17], [9, 17], [10, 17], [11, 17], [12, 17], [14, 17], [15, 17], [16, 17], [17, 17], [18, 17], [20, 17], [21, 17], [22, 17], [23, 17], [24, 17], [2, 18], [3, 18], [4, 18], [5, 18], [6, 18], [8, 18], [9, 18], [10, 18], [11, 18], [12, 18], [14, 18], [15, 18], [16, 18], [17, 18], [18, 18], [20, 18], [21, 18], [22, 18], [23, 18], [24, 18], [2, 20], [3, 20], [4, 20], [5, 20], [6, 20], [8, 20], [9, 20], [10, 20], [11, 20], [12, 20], [14, 20], [15, 20], [16, 20], [17, 20], [18, 20], [20, 20], [21, 20], [22, 20], [23, 20], [24, 20], [2, 21], [3, 21], [4, 21], [5, 21], [6, 21], [8, 21], [9, 21], [10, 21], [11, 21], [12, 21], [14, 21], [15, 21], [16, 21], [17, 21], [18, 21], [20, 21], [21, 21], [22, 21], [23, 21], [24, 21], [2, 22], [3, 22], [4, 22], [5, 22], [6, 22], [8, 22], [9, 22], [10, 22], [11, 22], [12, 22], [14, 22], [15, 22], [16, 22], [17, 22], [18, 22], [20, 22], [21, 22], [22, 22], [23, 22], [24, 22], [2, 23], [3, 23], [4, 23], [5, 23], [6, 23], [8, 23], [9, 23], [10, 23], [11, 23], [12, 23], [14, 23], [15, 23], [16, 23], [17, 23], [18, 23], [20, 23], [21, 23], [22, 23], [23, 23], [24, 23], [2, 24], [3, 24], [4, 24], [5, 24], [6, 24], [8, 24], [9, 24], [10, 24], [11, 24], [12, 24], [14, 24], [15, 24], [16, 24], [17, 24], [18, 24], [20, 24], [21, 24], [22, 24], [23, 24], [24, 24]]
agent_state = [1, 1]
goal = [25, 25]

############ DESIGN ############

game_over = False
OBSTACLES_COLOR = (100, 100, 100)
GOAL_COLOR = (0, 150, 0)
AGENT_COLOR = (0, 150, 150)
num_steps = 10
letter_color = (255, 144, 30)

############# DRAW #############

# Draw board
board = np.zeros((620, 594, 3), dtype=np.uint8)

# Draw obstacles
for obstacle in obstacles:
    cv2.rectangle(board, (obstacle[0] * 22 + 1, obstacle[1] * 22 + 1),
                  (obstacle[0] * 22 + 22 - 1, obstacle[1] * 22 + 22 - 1), OBSTACLES_COLOR, -1)

# Draw goal
cv2.rectangle(board, (goal[0] * 22 + 1, goal[1] * 22 + 1),
              (goal[0] * 22 + 22 - 1, goal[1] * 22 + 22 - 1), GOAL_COLOR, -1)

# Draw agent
cv2.circle(board, (agent_state[0] * 22 + 11, agent_state[1] * 22 + 11), 8, AGENT_COLOR, -1)

# Draw steps
cv2.putText(board, f"Steps: {num_steps}", (5, 612),
            cv2.FONT_HERSHEY_PLAIN, 1, letter_color, 1, cv2.LINE_AA)

# Draw game over
if game_over:
    cv2.putText(board, 'GAME OVER', (480, 612), cv2.FONT_HERSHEY_PLAIN,
                1, letter_color, 1, cv2.LINE_AA)

cv2.imshow('MAZE HUNTER', board)

# Pulsar cualquier tecla para salir
q = cv2.waitKey(1)
while q == -1:
    q = cv2.waitKey(1)

cv2.destroyAllWindows()


