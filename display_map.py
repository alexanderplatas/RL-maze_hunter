import cv2
import json
import numpy as np

""" PRUEBAS PARA CREAR NUEVOS MAPAS """

############ PARAMS ############

# obstacles = [[0, 0], [0, 1], [0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 7], [0, 8], [0, 9], [0, 10], [0, 11], [0, 12], [0, 13], [0, 14], [0, 15], [0, 16], [0, 17], [0, 18], [0, 19], [0, 20], [0, 21], [0, 22], [0, 23], [0, 24], [0, 25], [0, 26], [1, 0], [2, 0], [3, 0], [4, 0], [5, 0], [6, 0], [7, 0], [8, 0], [9, 0], [10, 0], [11, 0], [12, 0], [13, 0], [14, 0], [15, 0], [16, 0], [17, 0], [18, 0], [19, 0], [20, 0], [21, 0], [22, 0], [23, 0], [24, 0], [25, 0], [26, 0], [26, 0], [26, 1], [26, 2], [26, 3], [26, 4], [26, 5], [26, 6], [26, 7], [26, 8], [26, 9], [26, 10], [26, 11], [26, 12], [26, 13], [26, 14], [26, 15], [26, 16], [26, 17], [26, 18], [26, 19], [26, 20], [26, 21], [26, 22], [26, 23], [26, 24], [26, 25], [26, 26], [0, 26], [1, 26], [2, 26], [3, 26], [4, 26], [5, 26], [6, 26], [7, 26], [8, 26], [9, 26], [10, 26], [11, 26], [12, 26], [13, 26], [14, 26], [15, 26], [16, 26], [17, 26], [18, 26], [19, 26], [20, 26], [21, 26], [22, 26], [23, 26], [24, 26], [25, 26]
# ,]
# agent_state = [1, 1]
# goal = [25, 25]


MAP = 'original'
with open('maps.json', 'r', encoding='utf8') as f:
    map = json.load(f)[MAP]
obstacles = map['obstacles']
hunter = map['hunter']
prey = map['prey']
goal = map['goal']


############ DESIGN ############

game_over = False
OBSTACLES_COLOR = (100, 100, 100)
GOAL_COLOR = (0, 150, 0)
AGENT_COLOR = (0, 150, 150)
num_steps = 0
letter_color = (50, 205, 154)
PREY_COLOR = (255, 144, 30)
HUNTER_COLOR = (30, 144, 255)

############# DRAW #############

# Draw board
board = np.zeros((620, 594, 3), dtype=np.uint8)
overlay = board.copy()

# Draw prey vision
i = (prey[0] - 2)*22
j = (prey[1] - 2)*22
alpha = 0.3
cv2.rectangle(overlay, (i+1, j+1), (i + 109, j + 109), (150, 80, 80), -1)
board = cv2.addWeighted(overlay, alpha, board, 1 - alpha, 0)

# Draw hunter vision
i = (hunter[0] - 1) * 22
j = (hunter[1] - 1) * 22
cv2.rectangle(overlay, (i + 1, j + 1), (i + 65, j + 65), (80, 80, 150), -1)
board = cv2.addWeighted(overlay, alpha, board, 1 - alpha, 0)

# Draw obstacles
for obstacle in obstacles:
    cv2.rectangle(board, (obstacle[0] * 22 + 1, obstacle[1] * 22 + 1),
                  (obstacle[0] * 22 + 22 - 1, obstacle[1] * 22 + 22 - 1), OBSTACLES_COLOR, -1)

# Draw goal
cv2.rectangle(board, (goal[0] * 22 + 1, goal[1] * 22 + 1),
              (goal[0] * 22 + 22 - 1, goal[1] * 22 + 22 - 1), GOAL_COLOR, -1)

# Draw agent
cv2.circle(board, (hunter[0] * 22 + 11, hunter[1] * 22 + 11), 8, HUNTER_COLOR, -1)
cv2.circle(board, (prey[0] * 22 + 11, prey[1] * 22 + 11), 8, PREY_COLOR, -1)


# Draw steps
cv2.putText(board, f"Steps: {num_steps}", (5, 612),
            cv2.FONT_HERSHEY_PLAIN, 1, letter_color, 1, cv2.LINE_AA)

# Draw current reward
cv2.putText(board, f"Reward: {round(0.0, 4)}", (120, 612),
            cv2.FONT_HERSHEY_PLAIN, 1, letter_color, 1, cv2.LINE_AA)

# Draw game over
if game_over:
    cv2.putText(board, 'GAME OVER', (480, 612), cv2.FONT_HERSHEY_PLAIN,
                1, letter_color, 1, cv2.LINE_AA)

i = (hunter[0] - 2) * 22
j = (hunter[1] - 2) * 22
cv2.imshow('MAZE HUNTER', board)

cv2.imwrite(f"original.png", board)

# Pulsar cualquier tecla para salir
q = cv2.waitKey(1)
while q == -1:
    q = cv2.waitKey(1)

cv2.destroyAllWindows()


