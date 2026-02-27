import random
from collections import deque
import torch
from game import SnakeGameAI, Point, Directions, BLOCK_SIZE
from model import QNeuralNetwork, QTrainer
import numpy as np

MAX_MEMORY = 100_000
BATCH_SIZE = 1_000
LR = 0.001


class Agent:
    def __init__(self, state_dict = None, n_games=0):
        self.n_games = n_games
        self.epsilon = 0
        self.gamma = 0.9

        self.memory = deque(maxlen=MAX_MEMORY)

        self.model = QNeuralNetwork(11, 256, 3)
        self.trainer = QTrainer(self.model, self.gamma, LR)
        if state_dict is not None:
            self.model.load_state_dict(state_dict)

    def get_state(self, game):
        #look around
        head = game.head
        point_l = Point((head.x-BLOCK_SIZE), head.y)
        point_r = Point((head.x+BLOCK_SIZE), head.y)
        point_u = Point(head.x, (head.y-BLOCK_SIZE))
        point_d = Point(head.x, (head.y+BLOCK_SIZE))

        #current direction
        dir_l = game.direction == Directions.Left
        dir_r = game.direction == Directions.Right
        dir_u = game.direction == Directions.Up
        dir_d = game.direction == Directions.Down

        #current 11 params
        state = [
            # Danger Straight
            (dir_r and game._is_collision(point_r)) or
            (dir_l and game._is_collision(point_l)) or
            (dir_u and game._is_collision(point_u)) or
            (dir_d and game._is_collision(point_d)),

            # Danger Right
            (dir_u and game._is_collision(point_r)) or
            (dir_d and game._is_collision(point_l)) or
            (dir_l and game._is_collision(point_u)) or
            (dir_r and game._is_collision(point_d)),

            # Danger Left
            (dir_d and game._is_collision(point_r)) or
            (dir_u and game._is_collision(point_l)) or
            (dir_r and game._is_collision(point_u)) or
            (dir_l and game._is_collision(point_d)),

            # Move direction
            dir_l, dir_r, dir_u, dir_d,

            # Food location
            game.food.x < game.head.x,  # food left
            game.food.x > game.head.x,  # food right
            game.food.y < game.head.y,  # food up
            game.food.y > game.head.y  # food down
        ]

        return np.array(state, dtype=int)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE)
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)

    # Exploit is basing decisions on stored data
    # Explore forces the agent to try new things, so we won't stack in a local minimum
    def get_action(self, state):
            self.epsilon = 80 - self.n_games
            final_move = [0, 0, 0]

            # Explore
            if random.randint(0, 200) < self.epsilon:
                move = random.randint(0, 2)
                final_move[move] = 1

            # Exploit
            else:
                state_opt = torch.tensor(state, dtype=torch.float)
                pred = self.model(state_opt)

                move = torch.argmax(pred).item()
                final_move[move] = 1

            return final_move


