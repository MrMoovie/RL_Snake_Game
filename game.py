import pygame
from collections import namedtuple
import random
from enum import Enum
import numpy as np


pygame.init()

class Directions(Enum):
    Right = 1
    Left = 2
    Up = 3
    Down = 4

Point = namedtuple('Point', 'x,y')
BLOCK_SIZE = 20
SPEED = 1000

class SnakeGameAI:
    def __init__(self, w=640, h=480):
        self.w = w
        self.h = h

        self.display = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption("RL Snake game")
        self.clock = pygame.time.Clock()

        self.reset()

    def reset(self):
        self.direction = Directions.Right
        self.head = Point(self.w / 2, self.h / 2)
        self.snake = [self.head,
                      Point((self.head.x - BLOCK_SIZE), self.head.y),
                      Point((self.head.x - 2 * BLOCK_SIZE), self.head.y)]

        self.score = 0

        ###1
        self.hunter_direction = Directions.Down
        self.hunter = Point(0, 0)
        ###

        self.food = None
        self._place_food()
        self.track_iteration = 0

    def _place_food(self):
        x = random.randint(0, (self.w - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
        y = random.randint(0, (self.h - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
        self.food = Point(x,y)

        if self.food in self.snake:
            self._place_food()
        ###2
        # if self.food == self.hunter:
        #     self._place_food()
        ###

    ##part 2
    def play_step(self, action):
        self.track_iteration += 1
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

        old_distance = abs(self.head.x-self.food.x) + abs(self.head.y-self.food.y)

        self._move(action) ###
        self.snake.insert(0, self.head)

        new_distance = abs(self.head.x - self.food.x) + abs(self.head.y - self.food.y)

        reward = 0
        game_over = False
        if self._is_collision() or self.track_iteration > 100 * len(self.snake):
            game_over=True
            reward = -10
            return reward, game_over, self.score

        if self.head == self.food:
            self.score += 1
            reward = 10
            self._place_food()
        else: # wandering penalty
            self.snake.pop()
            if new_distance > old_distance:
                reward = -0.2
            else:
                reward = 0.1

        reward-=0.05

        self._update_ui()
        self.clock.tick(SPEED)

        return reward, game_over, self.score

    def _is_collision(self, pt=None):
        if pt is None:
            pt = self.head
            ##if pt is self.hunter:
                ##return True
        if pt.x > (self.w - BLOCK_SIZE) or pt.x < 0 or pt.y > (self.h-BLOCK_SIZE) or pt.y < 0:
            return True
        if pt in self.snake[1:]:
            return True
        return False

   ## def hunt(self):

    ###stage 3
    def _move(self, action):
        clock_wise = [Directions.Right, Directions.Down, Directions.Left, Directions.Up]
        idx = clock_wise.index(self.direction)

        if np.array_equal(action, [1, 0, 0]):
            new_dir = clock_wise[idx]  # straight
        elif np.array_equal(action, [0, 1, 0]):
            new_dir = clock_wise[(idx + 1) % 4]  # right turn
        else:
            new_dir = clock_wise[(idx - 1) % 4]  # left turn

        self.direction = new_dir

        x = self.head.x
        y = self.head.y
        if self.direction == Directions.Right:
            x += BLOCK_SIZE
        if self.direction == Directions.Left:
            x-=BLOCK_SIZE
        if self.direction == Directions.Up:
            y-=BLOCK_SIZE
        if self.direction == Directions.Down:
            y+=BLOCK_SIZE

        self.head = Point(x,y)

        #-----------------------------------------------------
    def get_image_state(self):
        grid_w = self.w // BLOCK_SIZE
        grid_h = self.h // BLOCK_SIZE

        state = np.zeros((3, grid_h, grid_w), dtype=np.float32)

        food_x = int(self.food.x // BLOCK_SIZE)
        food_y = int(self.food.y // BLOCK_SIZE)
        if 0 <= food_x < grid_w and 0 <= food_y < grid_h:
            state[0, food_y, food_x] = 1.0

        head_x = int(self.head.x // BLOCK_SIZE)
        head_y = int(self.head.y // BLOCK_SIZE)
        if 0 <= head_x < grid_w and 0 <= head_y < grid_h:
            state[1, head_y, head_x] = 1.0

        for pt in self.snake[1:]:
            pt_x = int(pt.x // BLOCK_SIZE)
            pt_y = int(pt.y // BLOCK_SIZE)
            if 0 <= pt_x < grid_w and 0 <= pt_y < grid_h:
                state[2, pt_y, pt_x] = 1.0

        return state

    #-----------------------------------------------------



    def _update_ui(self):
        self.display.fill((0,0,0))

        for pt in self.snake:
            pygame.draw.rect(self.display, (0,255,0), pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE))

        pygame.draw.rect(self.display, (255,0,0), pygame.Rect(self.food.x, self.food.y, BLOCK_SIZE, BLOCK_SIZE))
        pygame.display.flip()